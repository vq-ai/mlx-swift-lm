// Copyright © 2026 Apple Inc.
//
// Reject-reforward rollback contract for Qwen3.5 MTP speculative decoding.
//
// `.rejectReforward` runs every verify capture-free (dodging the per-round capture tax the
// `.kernelScan`/`.lazyRescan` modes pay) and rebuilds a rejected round's state by restoring
// the RETAINED pre-verify gated-delta refs and re-forwarding the accepted prefix. Exactness
// rests on two facts, pinned here:
//
// 1. The gated-delta layer REPLACES `cache[0]`/`cache[1]` on update (MLX arrays are
//    immutable), so references retained before the verify ARE the pre-verify state.
// 2. The scan kernel consumes tokens sequentially in fp32 registers, so re-scanning the
//    kept prefix from the pre-state ends in the state the full verify scan passed through —
//    the same state a `.kernelScan` capture would have selected. (Identical math; measured
//    ~2e-7 apart because the re-forward re-projects at a different block length, switching
//    the matmul kernel config — the same scheduling-noise class as block-vs-token decode.)

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

// MARK: - Layer level

final class Qwen35RejectReforwardTests: XCTestCase {

    /// Tiny gated-delta config. `linear_key_head_dim` must be 32 (the Metal scan kernel
    /// requires Dk divisible by 32); everything else is minimal.
    private func makeNet() throws -> Qwen35GatedDeltaNet {
        let json = """
            {
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 4
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        return Qwen35GatedDeltaNet(config)
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Restore-refs + prefix re-forward must land on the state a `.kernelScan` capture
    /// would have selected at that index, and on the state a sequential T=1 decode of the
    /// same prefix reaches — for every possible accepted count.
    func testReforwardRebuildsCapturedAndSequentialState() throws {
        MLXRandom.seed(23)
        let net = try makeNet()
        let prime = MLXRandom.normal([1, 4, 16])  // non-trivial pre-verify state
        let K = 5
        let x = MLXRandom.normal([1, K, 16])  // the verify block

        let saved = Qwen35MTP.captureMode
        defer { Qwen35MTP.captureMode = saved }

        // Reference 1: .kernelScan captures from an identically-primed cache.
        Qwen35MTP.captureMode = .kernelScan
        let capCache = MambaCache()
        _ = net(prime, mask: nil, cache: capCache)
        capCache.captureVerify = true
        _ = net(x, mask: nil, cache: capCache)
        let capturedConv = capCache.capturedConv
        let capturedSSM = capCache.capturedSSM
        XCTAssertEqual(capturedSSM.count, K)
        eval(capturedConv + capturedSSM)

        // Reference 2: sequential T=1 decode from an identically-primed cache.
        let seqCache = MambaCache()
        _ = net(prime, mask: nil, cache: seqCache)
        var seqConv: [MLXArray] = []
        var seqSSM: [MLXArray] = []
        for t in 0 ..< K {
            _ = net(x[0..., t ..< (t + 1), 0...], mask: nil, cache: seqCache)
            seqConv.append(seqCache[0]!)
            seqSSM.append(seqCache[1]!)
        }
        eval(seqConv + seqSSM)

        // Reforward path: prime, RETAIN the refs, verify capture-free, then rebuild for
        // every accepted count by restoring the refs + one prefix re-forward.
        Qwen35MTP.captureMode = .rejectReforward
        let cache = MambaCache()
        _ = net(prime, mask: nil, cache: cache)
        let preConv = cache[0]!
        let preSSM = cache[1]!
        _ = net(x, mask: nil, cache: cache)  // captureVerify stays false — the fast path
        XCTAssertTrue(
            cache.capturedConv.isEmpty && cache.capturedSSM.isEmpty,
            "the reforward mode's verify must not arm capture")

        for acc in 0 ..< K {
            cache[0] = preConv
            cache[1] = preSSM
            _ = net(x[0..., 0 ..< (acc + 1), 0...], mask: nil, cache: cache)
            let convR = cache[0]!
            let ssmR = cache[1]!
            // vs the capture the shipped rollback would have restored. The scan itself is
            // prefix-exact (GatedDeltaTests pins tolerance 0 for prefix scans over SHARED
            // projections), but the re-forward recomputes the projections at a different
            // block length, which switches the matmul kernel config (gemv vs gemm) —
            // measured ~2e-7, the same scheduling-noise class the capture-vs-sequential
            // test tolerates at 1e-5.
            XCTAssertLessThan(
                maxAbsDiff(ssmR, capturedSSM[acc]), 1e-5,
                "re-forward SSM state at acc=\(acc) differs from the .kernelScan capture")
            XCTAssertLessThan(
                maxAbsDiff(convR, capturedConv[acc]), 1e-5,
                "re-forward conv state at acc=\(acc) differs from the .kernelScan capture")
            // vs plain sequential decode of the same prefix — the state plain greedy
            // decode would carry. Same tolerance and same reason as above.
            XCTAssertLessThan(
                maxAbsDiff(ssmR, seqSSM[acc]), 1e-5,
                "re-forward SSM state at acc=\(acc) diverged from sequential decode")
            XCTAssertLessThan(
                maxAbsDiff(convR, seqConv[acc]), 1e-5,
                "re-forward conv state at acc=\(acc) diverged from sequential decode")
            if acc == 0 {
                // A T-matched re-forward (T=1 vs the sequential T=1 step) runs the exact
                // same kernels on the exact same inputs — bit-exact, pinning that the
                // tolerance above covers ONLY the cross-T kernel-config switch.
                XCTAssertEqual(
                    maxAbsDiff(ssmR, seqSSM[acc]), 0,
                    "T=1 re-forward must be bit-exact vs the sequential T=1 step")
                XCTAssertEqual(
                    maxAbsDiff(convR, seqConv[acc]), 0,
                    "T=1 re-forward must be bit-exact vs the sequential T=1 step")
            }
        }
    }
}

// MARK: - Decode loop level

final class Qwen35RejectReforwardLoopTests: XCTestCase {

    /// Tiny hybrid target: layer 0 gated-delta (MambaCache), layer 1 full attention
    /// (KVCacheSimple) — both rollback flavors exercised in one stack.
    private static let tinyTextConfig = """
        {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32,
            "full_attention_interval": 2
        }
        """

    private func makeTarget() throws -> Qwen35Model {
        let json = """
            {"model_type": "qwen3_5", "text_config": \(Self.tinyTextConfig)}
            """
        let config = try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
        return Qwen35Model(config)
    }

    private func makeDrafter() throws -> Qwen35DraftModel {
        let json = """
            {
                "model_type": "qwen3_5_mtp",
                "block_size": 3,
                "tie_word_embeddings": true,
                "text_config": \(Self.tinyTextConfig)
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35DraftConfiguration.self, from: Data(json.utf8))
        return Qwen35DraftModel(config)
    }

    private let prompt: [Int] = [3, 7, 1, 9, 4, 2, 11, 5]

    private func run(
        _ mode: Qwen35CaptureMode, target: Qwen35Model, drafter: Qwen35DraftModel,
        prompt: [Int], maxTokens: Int, eosTokens: Set<Int> = [],
        session: Qwen35MTPSession
    ) -> Qwen35MTPResult {
        let saved = Qwen35MTP.captureMode
        defer { Qwen35MTP.captureMode = saved }
        Qwen35MTP.captureMode = mode
        return Qwen35MTP.generate(
            target: target, drafter: drafter, promptTokens: prompt,
            maxTokens: maxTokens, eosTokens: eosTokens, blockSize: 3, session: session)
    }

    /// Plain (non-speculative) greedy decode — the stream every mode must reproduce.
    private func plainGreedy(target: Qwen35Model, prompt: [Int], maxTokens: Int) -> [Int] {
        let cache = target.newCache(parameters: nil)
        var input = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
        var out: [Int] = []
        while out.count < maxTokens {
            let logits = target(input, cache: cache)
            let last = logits[0..., (logits.dim(1) - 1)..., 0...]
            let next = argMax(last, axis: -1).item(Int.self)
            out.append(next)
            input = MLXArray([Int32(next)]).reshaped([1, 1])
        }
        return out
    }

    private func offsets(_ caches: [KVCache]) -> [Int] { caches.map { $0.offset } }

    private func maxStateDiff(_ a: [KVCache], _ b: [KVCache]) -> Float {
        var worst: Float = 0
        for (ca, cb) in zip(a, b) {
            for (sa, sb) in zip(ca.state, cb.state) {
                worst = max(
                    worst,
                    abs(sa.asType(.float32) - sb.asType(.float32)).max().item(Float.self))
            }
        }
        return worst
    }

    /// Rejected rounds under `.rejectReforward` must leave tokens, session bookkeeping,
    /// cache offsets, and cache contents exactly where `.lazyRescan` leaves them — at one
    /// extra target forward per rejected round.
    func testRejectedRoundsMatchLazyRescanEndToEnd() throws {
        MLXRandom.seed(3)
        let target = try makeTarget()
        let drafter = try makeDrafter()
        let maxTokens = 24

        let sessA = Qwen35MTPSession()
        let resA = run(
            .lazyRescan, target: target, drafter: drafter, prompt: prompt,
            maxTokens: maxTokens, session: sessA)
        let sessB = Qwen35MTPSession()
        let resB = run(
            .rejectReforward, target: target, drafter: drafter, prompt: prompt,
            maxTokens: maxTokens, session: sessB)

        // The run must actually exercise the rejected-round rollback (a random-weight
        // drafter virtually never matches the target for a whole block).
        XCTAssertGreaterThan(
            resA.proposed, resA.accepted, "no rejections — the rollback path never ran")

        // Emitted stream: identical across modes AND identical to plain greedy decode.
        XCTAssertEqual(resA.tokens, resB.tokens)
        XCTAssertEqual(
            resB.tokens, plainGreedy(target: target, prompt: prompt, maxTokens: maxTokens),
            "the reforward mode broke greedy exactness")

        // Session bookkeeping parity (resume depends on this byte-for-byte).
        XCTAssertEqual(sessA.tokens, sessB.tokens)
        XCTAssertEqual(sessA.pendingFinalToken, sessB.pendingFinalToken)
        XCTAssertEqual(sessA.acceptLens, sessB.acceptLens)
        XCTAssertEqual(sessA.proposedLens, sessB.proposedLens)

        // Cache offsets: identical per layer, and the full-attention offset holds exactly
        // the session's consumed tokens ([.., bonus] + accepted drafts each round).
        XCTAssertEqual(offsets(sessA.caches), offsets(sessB.caches))
        let kvOffset = sessB.caches.compactMap { ($0 as? KVCacheSimple)?.offset }.first
        XCTAssertEqual(kvOffset, sessB.tokens.count)

        // Cost model: capture modes never re-forward; the reforward mode pays exactly one
        // extra target forward per rejected round.
        let rejectedRounds = zip(sessB.acceptLens, sessB.proposedLens)
            .filter { $0.0 < $0.1 }.count
        XCTAssertEqual(resA.rounds, resB.rounds)
        XCTAssertEqual(resA.targetForwards, resA.rounds)
        XCTAssertEqual(resB.targetForwards, resB.rounds + rejectedRounds)

        // Rebuilt cache contents match the capture-restored ones (numerics only).
        XCTAssertLessThan(maxStateDiff(sessA.caches, sessB.caches), 1e-4)
    }

    /// The mid-block EOS trim path shares the rebuild: after an EOS ends generation, the
    /// session must be left resumable with identical bookkeeping in both modes.
    func testEOSMidBlockRebuildMatchesLazyRescan() throws {
        let maxTokens = 24

        // Discover a greedy stream with a token whose FIRST occurrence is mid-stream (an
        // earlier duplicate would end generation at the duplicate instead, exercising
        // nothing). Tiny random models cycle fast, so scan a few seeds.
        var found:
            (target: Qwen35Model, drafter: Qwen35DraftModel, free: Qwen35MTPResult, eosIdx: Int)?
        for seed: UInt64 in [5, 6, 7, 11, 13, 17] {
            MLXRandom.seed(seed)
            let target = try makeTarget()
            let drafter = try makeDrafter()
            let free = run(
                .lazyRescan, target: target, drafter: drafter, prompt: prompt,
                maxTokens: maxTokens, session: Qwen35MTPSession())
            if let eosIdx = free.tokens.indices.dropFirst(1).first(where: { i in
                !free.tokens[..<i].contains(free.tokens[i])
            }) {
                found = (target, drafter, free, eosIdx)
                break
            }
        }
        guard let (target, drafter, free, eosIdx) = found else {
            XCTFail("no seed produced a stream with a fresh mid-stream token to use as EOS")
            return
        }
        let eos = free.tokens[eosIdx]
        let expected = Array(free.tokens.prefix(eosIdx))

        let sessA = Qwen35MTPSession()
        let resA = run(
            .lazyRescan, target: target, drafter: drafter, prompt: prompt,
            maxTokens: maxTokens, eosTokens: [eos], session: sessA)
        let sessB = Qwen35MTPSession()
        let resB = run(
            .rejectReforward, target: target, drafter: drafter, prompt: prompt,
            maxTokens: maxTokens, eosTokens: [eos], session: sessB)

        XCTAssertEqual(resA.tokens, expected)
        XCTAssertEqual(resB.tokens, expected)
        XCTAssertEqual(sessA.tokens, sessB.tokens)
        XCTAssertNil(sessA.pendingFinalToken)
        XCTAssertNil(sessB.pendingFinalToken)
        XCTAssertEqual(offsets(sessA.caches), offsets(sessB.caches))
        let kvOffset = sessB.caches.compactMap { ($0 as? KVCacheSimple)?.offset }.first
        XCTAssertEqual(kvOffset, sessB.tokens.count)
        XCTAssertLessThan(maxStateDiff(sessA.caches, sessB.caches), 1e-4)
    }

    /// Session resume (the agentic prefix-extension path) across a reforward-rolled-back
    /// step must behave exactly like the capture modes: same suffix prefill, same tokens.
    func testResumeAfterRejectReforwardMatchesLazyRescan() throws {
        MLXRandom.seed(9)
        let target = try makeTarget()
        let drafter = try makeDrafter()

        func twoStep(_ mode: Qwen35CaptureMode)
            -> (step1: [Int], step2: [Int], session: Qwen35MTPSession, res2: Qwen35MTPResult)
        {
            let session = Qwen35MTPSession()
            let r1 = run(
                mode, target: target, drafter: drafter, prompt: prompt,
                maxTokens: 10, session: session)
            // The agentic caller's next prompt: consumed tokens + the pending final token
            // + the new message.
            var prompt2 = session.tokens
            if let pending = session.pendingFinalToken { prompt2.append(pending) }
            prompt2 += [13, 6]
            let r2 = run(
                mode, target: target, drafter: drafter, prompt: prompt2,
                maxTokens: 10, session: session)
            return (r1.tokens, r2.tokens, session, r2)
        }

        let a = twoStep(.lazyRescan)
        let b = twoStep(.rejectReforward)

        XCTAssertEqual(a.step1, b.step1)
        XCTAssertEqual(a.step2, b.step2)
        XCTAssertGreaterThan(b.res2.resumedSuffix, 0, "step 2 must resume, not re-prefill")
        XCTAssertEqual(a.res2.resumedSuffix, b.res2.resumedSuffix)
        XCTAssertEqual(a.session.tokens, b.session.tokens)
        XCTAssertEqual(a.session.pendingFinalToken, b.session.pendingFinalToken)
        XCTAssertEqual(offsets(a.session.caches), offsets(b.session.caches))
    }
}
