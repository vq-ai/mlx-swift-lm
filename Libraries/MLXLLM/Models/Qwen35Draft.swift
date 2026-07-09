// Copyright © 2026 Apple Inc.
//
// Qwen3.5 MTP (multi-token-prediction) self-speculative drafter + greedy decode loop.
//
// The drafter is a small stateful network with its OWN KV cache, prefilled from the target's
// post-final-norm hidden states (it does NOT cross-attend the target's K/V — unlike Gemma4).
// It reuses the base `Qwen35DecoderLayer` (forced full-attention) and the target's token
// embeddings + LM head. Ports mlx-vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py.

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - Config

public struct Qwen35DraftConfiguration: Codable, Sendable {
    public var modelType: String = "qwen3_5_mtp"
    public var blockSize: Int = 3
    public var tieWordEmbeddings: Bool = true
    public var textConfig: Qwen35TextConfiguration
    public var quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case blockSize = "block_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case textConfig = "text_config"
        case quantization
    }
}

// MARK: - Drafter network

public final class Qwen35DraftModel: Module {
    public let blockSize: Int

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    // Bound from the target model (not owned, not part of this module's parameters).
    private var inputEmbed: Embedding?
    private var lmHead: Linear?
    private var embedScale: Float = 1.0

    // Transient per-stream decode state (its own KV cache + the seed carried between rounds).
    private var cache: [KVCache] = []
    private var seedToken: MLXArray?
    private var seedHidden: MLXArray?
    private(set) var roundAppended = 0

    public init(_ config: Qwen35DraftConfiguration) {
        self.blockSize = config.blockSize
        var textConfig = config.textConfig
        // All drafter layers are full-attention (no gated-delta): with interval 1, layer 0's
        // `(0+1) % 1 == 0` ⇒ isLinear == false ⇒ self-attention.
        textConfig.fullAttentionInterval = 1
        let mtpLayers = 1
        textConfig.hiddenLayers = mtpLayers

        _fc.wrappedValue = Linear(2 * textConfig.hiddenSize, textConfig.hiddenSize, bias: false)
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: textConfig.hiddenSize, eps: textConfig.rmsNormEps)
        _preFcNormHidden.wrappedValue = RMSNorm(
            dimensions: textConfig.hiddenSize, eps: textConfig.rmsNormEps)
        _layers.wrappedValue = (0 ..< mtpLayers).map { _ in
            Qwen35DecoderLayer(textConfig, layerIdx: 0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: textConfig.hiddenSize, eps: textConfig.rmsNormEps)
        super.init()
    }

    /// Bind the target's embeddings + LM head (looked up per stream, never retained as params).
    public func bind(target: Qwen35Model) {
        let inner = target.languageModel.model
        self.inputEmbed = inner.embedTokens
        self.embedScale = 1.0
        self.lmHead = target.languageModel.lmHead
    }

    /// Fresh per-stream state: a KV cache per drafter layer + cleared seed.
    public func reset(target: Qwen35Model) {
        bind(target: target)
        self.cache = layers.map { _ in KVCacheSimple() }
        self.seedToken = nil
        self.seedHidden = nil
        self.roundAppended = 0
    }

    private func lmHeadFn(_ h: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(h) }
        return inputEmbed!.asLinear(h)
    }

    /// concat(norm(tokenEmbed), norm(targetHidden)) → fc → full-attn layers (own cache) → norm.
    /// RoPE positions ride each layer cache's offset, which tracks the drafted position.
    private func forwardHidden(tokenEmbed: MLXArray, hidden: MLXArray) -> MLXArray {
        var h = concatenated(
            [preFcNormEmbedding(tokenEmbed), preFcNormHidden(hidden)], axis: -1)
        h = fc(h)
        for (layer, c) in zip(layers, cache) {
            let mask = createAttentionMask(h: h, cache: c)
            h = layer(h, attentionMask: mask, ssmMask: nil, cache: c)
        }
        return norm(h)
    }

    private func forwardTokens(_ tokens: MLXArray, hidden: MLXArray) -> MLXArray {
        let tokenEmbed = inputEmbed!(tokens) * embedScale
        return forwardHidden(tokenEmbed: tokenEmbed, hidden: hidden)
    }

    private func lastStep(_ h: MLXArray) -> MLXArray {
        h[0..., (h.dim(1) - 1) ..< h.dim(1), 0...]
    }

    private func setSeedFromHidden(_ hidden: MLXArray) {
        let logits = lmHeadFn(hidden)
        seedToken = argMax(logits, axis: -1).asType(.int32)  // [B, 1]
        seedHidden = hidden
    }

    /// Prefill the drafter cache by pairing the target's per-position hidden states with the
    /// left-shifted input tokens + the bonus, then seed the first draft from the last position.
    /// Chunked like the target's prefill: a single full-length forward materializes O(L²)
    /// attention scores in the drafter's full-attention layer (~1.3 GB at 6k ctx) — quadratic
    /// time and memory on long fresh starts. Only the LAST chunk's output seeds the draft.
    public func prefillFromTargetHidden(
        inputIds: MLXArray, hidden: MLXArray, bonusToken: Int, prefillStep: Int = 256
    ) {
        guard inputIds.dim(1) > 0 else { return }
        let bonus = MLXArray([Int32(bonusToken)]).reshaped([1, 1])
        let shifted = concatenated([inputIds[0..., 1...].asType(.int32), bonus], axis: 1)
        let total = shifted.dim(1)
        var start = 0
        var lastChunkOut: MLXArray? = nil
        while start < total {
            let end = min(start + prefillStep, total)
            let h = forwardTokens(
                shifted[0..., start ..< end], hidden: hidden[0..., start ..< end, 0...])
            eval(h)
            lastChunkOut = h
            start = end
        }
        if let h = lastChunkOut {
            setSeedFromHidden(lastStep(h))
        }
    }

    /// Draft `blockSize - 1` tokens greedily, starting from the carried seed.
    public func draftBlock(blockSize: Int? = nil) -> MLXArray {
        let effBlock = blockSize ?? self.blockSize
        roundAppended = 0
        guard let s = seedToken, let sh = seedHidden else {
            return MLXArray.zeros([1, 0], type: Int32.self)
        }
        var tok = s
        var hPrev = sh
        var tokens: [MLXArray] = [tok]
        seedToken = nil
        seedHidden = nil
        while tokens.count < effBlock - 1 {
            hPrev = forwardTokens(tok, hidden: hPrev)
            roundAppended += 1
            tok = argMax(lmHeadFn(hPrev), axis: -1).asType(.int32)  // [B, 1]
            tokens.append(tok)
        }
        return concatenated(tokens, axis: 1)  // [B, blockSize-1]
    }

    /// After the target verifies, trim the drafter cache to the accepted prefix and re-commit
    /// the newly-accepted tokens (+ bonus) paired with the target's verify hidden, re-seeding.
    public func acceptVerifiedTokens(
        verifyHidden: MLXArray, draftTokens: MLXArray, accepted: Int, newTokens: [Int]
    ) {
        let keepAppended = min(accepted, roundAppended)
        let trim = roundAppended - keepAppended
        if trim > 0 {
            for c in cache { _ = (c as? KVCacheSimple)?.trim(trim) }
        }
        var tokenChunks: [MLXArray] = []
        var hiddenChunks: [MLXArray] = []
        for draftIdx in keepAppended ..< accepted {
            tokenChunks.append(draftTokens[0..., draftIdx ..< (draftIdx + 1)])
            hiddenChunks.append(verifyHidden[0..., draftIdx ..< (draftIdx + 1), 0...])
        }
        if let last = newTokens.last {
            tokenChunks.append(MLXArray([Int32(last)]).reshaped([1, 1]))
            hiddenChunks.append(verifyHidden[0..., accepted ..< (accepted + 1), 0...])
        }
        if !tokenChunks.isEmpty {
            let toks = concatenated(tokenChunks, axis: 1).asType(.int32)
            let hids = concatenated(hiddenChunks, axis: 1)
            let h = forwardTokens(toks, hidden: hids)
            setSeedFromHidden(lastStep(h))
        }
        roundAppended = 0
    }

    /// Strip the `mtp.` prefix and apply the Qwen3.5 RMSNorm `+1` weight convention.
    ///
    /// The `+1` shift applies ONLY to original combined checkpoints (drafter keys under
    /// `mtp.`), whose norms are stored raw as `(1+w)` — the same guard the main model's
    /// sanitize uses. Split/trained drafter dirs (mlx-vlm split.py, mtp_train.py) store
    /// EFFECTIVE norm weights with unprefixed keys; shifting those again corrupts every
    /// norm (measured: v3 acc@1 0.83→0.76 agentic, 0.61→0.48 prose; stock 0.68→0.44).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let isOriginalCheckpoint = weights.keys.contains { $0.hasPrefix("mtp.") }
        let normSuffixes = [
            ".input_layernorm.weight", ".post_attention_layernorm.weight",
            ".q_norm.weight", ".k_norm.weight", "norm.weight",
            "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        ]
        var out = [String: MLXArray]()
        for (k0, v0) in weights {
            var k = k0
            if k.hasPrefix("mtp.") { k = String(k.dropFirst("mtp.".count)) }
            var v = v0
            if isOriginalCheckpoint, v.ndim == 1,
                normSuffixes.contains(where: { k.hasSuffix($0) }) {
                v = v + MLXArray(Float(1)).asType(v.dtype)
            }
            out[k] = v
        }
        return out
    }
}

// MARK: - Loading

extension Qwen35DraftModel {
    /// Load a Qwen3.5 MTP drafter checkpoint (config.json + model.safetensors) and bind it to
    /// the already-loaded target. Mirrors the framework's sanitize → quantize → update flow.
    public static func load(from directory: URL, target: Qwen35Model) throws -> Qwen35DraftModel {
        let configData = try Data(contentsOf: directory.appending(path: "config.json"))
        let config = try JSONDecoder().decode(Qwen35DraftConfiguration.self, from: configData)
        let model = Qwen35DraftModel(config)

        var weights = try loadArrays(url: directory.appending(path: "model.safetensors"))
        weights = model.sanitize(weights: weights)

        if let q = config.quantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] != nil ? q.asTuple : nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        model.bind(target: target)
        return model
    }
}

// MARK: - Target hidden return (Stage 4)

extension Qwen35TextModel {
    /// Run the text model and return BOTH the post-final-norm hidden states (the drafter's
    /// input) and the logits (for greedy verify), advancing `cache`.
    public func hiddenAndLogits(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> (hidden: MLXArray, logits: MLXArray) {
        let hidden = model(inputs, cache: cache)
        let logits = lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
        return (hidden, logits)
    }

    /// Hidden states only (no LM head), advancing `cache`. Prefill uses this: computing the
    /// full-vocab logits for every prompt position wastes an LM-head matmul + logits buffer
    /// per chunk when only the LAST position's logits are needed (the first bonus token).
    public func hidden(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model(inputs, cache: cache)
    }

    /// Apply the LM head to (a slice of) hidden states.
    public func logits(fromHidden hidden: MLXArray) -> MLXArray {
        lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }
}

extension Qwen35Model {
    public func hiddenAndLogits(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> (hidden: MLXArray, logits: MLXArray) {
        languageModel.hiddenAndLogits(inputs, cache: cache)
    }

    public func hidden(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel.hidden(inputs, cache: cache)
    }

    public func logits(fromHidden hidden: MLXArray) -> MLXArray {
        languageModel.logits(fromHidden: hidden)
    }
}

// MARK: - Greedy speculative decode loop (Stage 5)

/// Resumable decode state: the target's caches plus the EXACT token sequence they have
/// consumed. A follow-up `generate` whose prompt EXTENDS `tokens` prefills only the suffix —
/// the agentic loop re-sends the whole growing conversation every tool-call step, so resume
/// turns each step's prefill from O(conversation) into O(new tool result). The hybrid's SSM
/// states can't rewind, so resume is prefix-extension-only; any other prompt falls back to a
/// fresh prefill (and re-primes this session). The drafter's cache rides along implicitly
/// (same `Qwen35DraftModel` instance, not reset on resume) — pass the same drafter.
/// How the speculative verify prepares a rejected round's gated-delta rollback.
/// - `.kernelScan` / `.lazyRescan` ARM per-step state capture on EVERY verify forward (a
///   measured +60-115 ms/round on-device), so a rejection restores the captured state at the
///   accepted index without any re-forward.
/// - `.rejectReforward` runs every verify capture-free (the fast path) and pays only on
///   REJECTED rounds: restore the retained pre-verify gated-delta refs, trim the
///   full-attention caches, and re-forward the accepted prefix — rebuilding the states a
///   capture would have selected (scan-of-prefix ≡ prefix-of-scan; identical math, equal
///   up to kernel-shape scheduling like any block-vs-token forward). Memory-tight hosts
///   (iPhone) should use this.
/// - `.unsafeNoCapture` is a BENCH PROBE ONLY: the verify runs with capture disarmed
///   (measuring the true capture tax), and a rejected round keeps stale gated-delta states —
///   the emitted stream is NOT exact after the first rejection. Never ship it.
public enum Qwen35CaptureMode: Sendable {
    case kernelScan, lazyRescan, rejectReforward, unsafeNoCapture
}

public final class Qwen35MTPSession {
    public internal(set) var caches: [KVCache] = []
    public internal(set) var tokens: [Int] = []
    /// Set when the caches stopped matching `tokens` exactly.
    public internal(set) var invalidated = true
    /// The final emitted token NOT yet in `tokens`/caches (the last round's correction).
    /// nil after an EOS end (everything emitted is already consumed). A resuming caller builds
    /// its next prompt as `tokens + [pendingFinalToken] + new-message tokens`.
    public internal(set) var pendingFinalToken: Int?
    /// Accepted/proposed-draft histories carried ACROSS steps: tool-call steps are short
    /// (~9 rounds), so a per-generate history never reaches the adaptive controller's
    /// warm-up — carrying them lets blocks grow where acceptance supports it (formulaic
    /// tool calls hit 89-100%). Parallel arrays, one entry per speculation round.
    public internal(set) var acceptLens: [Int] = []
    public internal(set) var proposedLens: [Int] = []

    public init() {}

    /// True when `prompt` strictly extends the consumed tokens (resume = prefill the suffix).
    func canResume(prompt: [Int]) -> Bool {
        !invalidated && !tokens.isEmpty && prompt.count > tokens.count
            && Array(prompt.prefix(tokens.count)) == tokens
    }
}

public struct Qwen35MTPResult {
    public let tokens: [Int]
    public let proposed: Int
    public let accepted: Int
    /// Full-model target forwards in the decode loop (verify + any re-forwards). With the
    /// per-step rollback this is ~one per round; the device win tracks this dropping.
    public var targetForwards: Int = 0
    /// Number of speculation rounds (≈ target forwards). With `proposed`, gives the average
    /// block size actually used (useful to see what Adaptive settled on).
    public var rounds: Int = 0
    /// Resume telemetry: -1 = fresh prefill; otherwise the suffix length prefilled on resume.
    public var resumedSuffix: Int = -1
    /// Per-phase wall time across the decode loop, attributed at the three sync barriers:
    /// `draft` = draftBlock + its asArray sync (also realizes the previous round's re-commit
    /// graph), `verify` = target forward + its eval, `rollback` = cache rollback + its eval
    /// (partial rounds only). The remainder of decode time is walk/bookkeeping/streaming.
    public var draftSeconds: Double = 0
    public var verifySeconds: Double = 0
    public var rollbackSeconds: Double = 0

    // MARK: Per-round memory telemetry (retention-tax instrumentation)
    //
    // Sampled at the SAME sync barriers as the phase clocks above, as pure reads of the
    // Metal allocator's counters (`Memory.activeMemory`/`cacheMemory`/`peakMemory` are
    // plain size_t loads — no lock, no eval, no GPU sync; the per-round peak reset takes
    // only the allocator mutex). All values in MB; 0 = not measured (loop never ran).
    //
    // "Fresh" is the wiring-hypothesis money metric: MLX recycles buffers through a pool,
    // and `active + cache` stays CONSTANT while requests are served from the pool (reuse
    // moves bytes cache→active, free moves them back). The total only GROWS when the
    // allocator must create a new MTLBuffer — exactly the fresh wiring the retention-tax
    // hypothesis blames for the state-retaining modes' verify cost — and only SHRINKS when
    // buffers are truly released to the system (cache purge under pressure / GC inside
    // malloc). Positive inter-sample deltas of the total are attributed to the phase that
    // forced them.

    /// Active (in-use) MLX memory right before the draft phase, avg/max over rounds.
    public var memActiveBeforeDraftAvgMB: Double = 0
    public var memActiveBeforeDraftMaxMB: Double = 0
    /// Active memory right after the verify eval barrier.
    public var memActiveAfterVerifyAvgMB: Double = 0
    public var memActiveAfterVerifyMaxMB: Double = 0
    /// Active memory after the rollback block (sampled every round, rollback or not).
    public var memActiveAfterRollbackAvgMB: Double = 0
    public var memActiveAfterRollbackMaxMB: Double = 0
    /// Buffer-pool (recyclable) memory at the same three points, avg over rounds.
    public var memCacheBeforeDraftAvgMB: Double = 0
    public var memCacheAfterVerifyAvgMB: Double = 0
    public var memCacheAfterRollbackAvgMB: Double = 0
    /// Per-round high-water mark of active memory. Requires
    /// ``Qwen35MTP/memoryTelemetryResetsPeak`` (resets the program-global peak counter at
    /// each round start); 0 when disabled.
    public var memPeakRoundAvgMB: Double = 0
    public var memPeakRoundMaxMB: Double = 0
    /// Decode-loop high-water mark minus active at loop entry — the headroom the loop
    /// itself needs beyond the standing state (weights + caches + retained refs).
    public var memPeakDecodeGrowthMB: Double = 0
    /// Fresh bytes obtained from Metal per round (positive Δ(active+cache) between
    /// consecutive samples). Rounds that recycle perfectly report ~0; the wiring
    /// hypothesis predicts ~verify-workspace-sized values EVERY round in the
    /// state-retaining modes.
    public var memFreshRoundAvgMB: Double = 0
    public var memFreshRoundMaxMB: Double = 0
    /// The verify phase's share of the round's fresh bytes (after-draft → after-verify).
    public var memFreshVerifyRoundAvgMB: Double = 0
    public var memFreshVerifyRoundMaxMB: Double = 0
    /// Total fresh bytes across the decode loop (== `memFreshRoundAvgMB` × `rounds`).
    public var memFreshTotalMB: Double = 0
    /// Bytes truly released back to the system during decode (cache purges under memory
    /// pressure / GC inside malloc). Nonzero means the pool is torn down mid-loop, forcing
    /// later rounds to wire fresh buffers again — the churn signature.
    public var memReleasedTotalMB: Double = 0
}

/// Per-round allocator-counter sampler behind ``Qwen35MTPResult``'s memory telemetry.
///
/// Every read is a plain load of the Metal allocator's counters — no lock, no eval, no GPU
/// sync (`GPU.resetPeakMemory()` alone takes the allocator mutex; still trivially cheap).
/// Samples land at the same sync barriers as the phase clocks, so the counters reflect
/// exactly what the just-synced phase left allocated.
private struct Qwen35DecodeMemorySampler {
    private static let bytesPerMB = 1024.0 * 1024.0

    private let resetPeakPerRound: Bool
    private var rounds = 0
    private let activeAtLoopStartMB: Double
    private var decodePeakMB: Double
    private var lastTotalMB: Double

    // Per-point accumulators, indexed by Point.rawValue.
    private var activeSumMB = [0.0, 0.0, 0.0]
    private var activeMaxMB = [0.0, 0.0, 0.0]
    private var cacheSumMB = [0.0, 0.0, 0.0]

    private var peakRoundSumMB = 0.0
    private var peakRoundMaxMB = 0.0

    private var freshRoundMB = 0.0
    private var freshVerifyRoundMB = 0.0
    private var freshRoundSumMB = 0.0
    private var freshRoundMaxMB = 0.0
    private var freshVerifySumMB = 0.0
    private var freshVerifyMaxMB = 0.0
    private var releasedTotalMB = 0.0

    enum Point: Int { case beforeDraft = 0, afterVerify = 1, afterRollback = 2 }

    /// The two allocator counters, in MB. Plain loads — safe at any point in the loop.
    private static func countersMB() -> (active: Double, cache: Double) {
        (Double(Memory.activeMemory) / bytesPerMB, Double(Memory.cacheMemory) / bytesPerMB)
    }

    init(resetPeakPerRound: Bool) {
        self.resetPeakPerRound = resetPeakPerRound
        let (active, cache) = Self.countersMB()
        self.activeAtLoopStartMB = active
        self.decodePeakMB = active
        self.lastTotalMB = active + cache
    }

    /// Start a round: reset the (program-global) peak counter so ``roundEnd()`` reads a
    /// per-round high-water mark, then take the before-draft sample.
    mutating func roundBegin() {
        rounds += 1
        freshRoundMB = 0
        freshVerifyRoundMB = 0
        if resetPeakPerRound { GPU.resetPeakMemory() }
        sample(.beforeDraft)
    }

    /// After the draft sync barrier — feeds only the fresh-delta chain (the draft phase
    /// also realizes the previous round's drafter re-commit graph), isolating the verify
    /// phase's fresh bytes from the draft's.
    mutating func afterDraftSync() {
        let (active, cache) = Self.countersMB()
        advanceTotal(active: active, cache: cache, verifyPhase: false)
    }

    mutating func sample(_ point: Point) {
        let (active, cache) = Self.countersMB()
        let i = point.rawValue
        activeSumMB[i] += active
        activeMaxMB[i] = max(activeMaxMB[i], active)
        cacheSumMB[i] += cache
        decodePeakMB = max(decodePeakMB, active)
        advanceTotal(active: active, cache: cache, verifyPhase: point == .afterVerify)
    }

    private mutating func advanceTotal(active: Double, cache: Double, verifyPhase: Bool) {
        let total = active + cache
        let delta = total - lastTotalMB
        lastTotalMB = total
        if delta > 0 {
            freshRoundMB += delta
            if verifyPhase { freshVerifyRoundMB += delta }
        } else {
            releasedTotalMB -= delta
        }
    }

    /// Close the round (after the after-rollback sample): fold the per-round peak and
    /// fresh accumulators.
    mutating func roundEnd() {
        if resetPeakPerRound {
            let peak = Double(Memory.peakMemory) / Self.bytesPerMB
            peakRoundSumMB += peak
            peakRoundMaxMB = max(peakRoundMaxMB, peak)
            decodePeakMB = max(decodePeakMB, peak)
        }
        freshRoundSumMB += freshRoundMB
        freshRoundMaxMB = max(freshRoundMaxMB, freshRoundMB)
        freshVerifySumMB += freshVerifyRoundMB
        freshVerifyMaxMB = max(freshVerifyMaxMB, freshVerifyRoundMB)
    }

    /// Write the accumulated telemetry into the result. No-op when no round completed.
    func write(into result: inout Qwen35MTPResult) {
        guard rounds > 0 else { return }
        let n = Double(rounds)
        result.memActiveBeforeDraftAvgMB = activeSumMB[0] / n
        result.memActiveBeforeDraftMaxMB = activeMaxMB[0]
        result.memActiveAfterVerifyAvgMB = activeSumMB[1] / n
        result.memActiveAfterVerifyMaxMB = activeMaxMB[1]
        result.memActiveAfterRollbackAvgMB = activeSumMB[2] / n
        result.memActiveAfterRollbackMaxMB = activeMaxMB[2]
        result.memCacheBeforeDraftAvgMB = cacheSumMB[0] / n
        result.memCacheAfterVerifyAvgMB = cacheSumMB[1] / n
        result.memCacheAfterRollbackAvgMB = cacheSumMB[2] / n
        if resetPeakPerRound {
            result.memPeakRoundAvgMB = peakRoundSumMB / n
            result.memPeakRoundMaxMB = peakRoundMaxMB
        }
        result.memPeakDecodeGrowthMB = max(0, decodePeakMB - activeAtLoopStartMB)
        result.memFreshRoundAvgMB = freshRoundSumMB / n
        result.memFreshRoundMaxMB = freshRoundMaxMB
        result.memFreshVerifyRoundAvgMB = freshVerifySumMB / n
        result.memFreshVerifyRoundMaxMB = freshVerifyMaxMB
        result.memFreshTotalMB = freshRoundSumMB
        result.memReleasedTotalMB = releasedTotalMB
    }
}

/// Draws the per-round sampling outcomes for the sampled speculative verify.
///
/// All heavy tensors ([K, V] transforms, gumbel noise) stay lazy inside the caller's verify
/// evaluation; only the tiny outcome vectors ([K] floats/ints) are materialized. The
/// transforms mirror the standard sampled path EXACTLY (RepetitionContext's CTRL-style
/// penalty, then `TopPSampler`'s top-p-on-untempered-logprobs, then temperature inside the
/// categorical) so speculative decode samples from the same distribution as non-speculative.
private struct Qwen35SampledVerifier {
    let sampling: Qwen35MTP.Sampling
    let randomState: MLXRandom.RandomState

    init(sampling: Qwen35MTP.Sampling) {
        self.sampling = sampling
        self.randomState = sampling.seed.map { MLXRandom.RandomState(seed: $0) }
            ?? MLXRandom.RandomState()
    }

    /// The final log-distribution rows after penalty → top-p → temperature.
    /// `logits`: [rows, V] float32; `ringRows`: per-row repetition-context token ids.
    private func finalLogprobs(_ logits: MLXArray, ringRows: [[Int]]) -> MLXArray {
        var logits = logits
        if let penalty = sampling.repetitionPenalty, penalty != 0, penalty != 1,
            ringRows.contains(where: { !$0.isEmpty }) {
            // Rows may have different context lengths (in-block drafts accrue) — pad each
            // row's id list to a common width by REPEATING its first id (a duplicate index
            // in putAlong just rewrites the same penalized value; the transform is
            // idempotent per token, so repeats are harmless).
            let width = ringRows.map(\.count).max() ?? 0
            let padded = ringRows.map { row -> [UInt32] in
                let pad = Array(repeating: UInt32(row.first ?? 0), count: width - row.count)
                return pad + row.map { UInt32($0) }
            }
            let idx = MLXArray(padded.flatMap { $0 }).reshaped([ringRows.count, width])
            var sel = takeAlong(logits, idx, axis: -1)
            sel = MLX.where(sel .< 0, sel * penalty, sel / penalty)
            logits = putAlong(logits, idx, values: sel, axis: -1)
        }
        var logprobs = logSoftmax(logits, axis: -1)
        if sampling.topP < 1 {
            // Same math as TopPSampler.applyTopP: mask the low-probability tail on the
            // UNTEMPERED logprobs, in sorted order, scatter back.
            let sortedIndices = argSort(logprobs, axis: -1)
            let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
            let cumulativeProbs = cumsum(exp(sortedLogprobs), axis: -1)
            let filtered = MLX.where(
                cumulativeProbs .> (1 - sampling.topP), sortedLogprobs,
                MLXArray(-Float.infinity))
            logprobs = putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
        }
        return logSoftmax(logprobs * (1 / sampling.temperature), axis: -1)
    }

    /// Sample one token from a single-position logits row (the post-prefill bonus).
    func sampleToken(logits: MLXArray, ring: [Int]) -> Int {
        let logp = finalLogprobs(logits.reshaped([1, logits.dim(-1)]).asType(.float32),
                                 ringRows: [ring])
        let sampled = withRandomState(randomState) {
            argMax(logp + MLXRandom.gumbel(logp.shape, type: Float.self), axis: -1)
        }
        return sampled.item(Int.self)
    }

    /// Per-round outcomes for ``Qwen35MTP/speculativeWalkSampled``.
    ///
    /// `verifyLogits`: [1, K, V] (row i = target's distribution after consuming
    /// verifyInput[0...i]); `draftTokens`: the K-1 drafts; `ringRows[i]`: repetition
    /// context for row i (base ring + drafts[0..<i]).
    func outcomes(
        verifyLogits: MLXArray, draftTokens: [Int], ringRows: [[Int]]
    ) -> (pDraft: [Float], uniforms: [Float], residual: [Int], final: Int) {
        let k = verifyLogits.dim(1)
        precondition(draftTokens.count == k - 1 && ringRows.count == k)
        let logp = finalLogprobs(
            verifyLogits.squeezed(axis: 0).asType(.float32), ringRows: ringRows)  // [K, V]

        let (pDraftArr, residualArr, finalArr, uniformsArr) = withRandomState(randomState) {
            let draftRows = logp[0 ..< max(k - 1, 0), 0...]
            let draftIdx = MLXArray(draftTokens.map { UInt32($0) }).reshaped([k - 1, 1])
            let pDraft = exp(takeAlong(draftRows, draftIdx, axis: -1)).squeezed(axes: [-1])
            // Residual distribution per draft row: zero out the draft token, renormalize —
            // gumbel-argmax over the masked logprobs IS a sample from that residual.
            let masked = putAlong(
                draftRows, draftIdx,
                values: broadcast(MLXArray(-Float.infinity), to: [k - 1, 1]), axis: -1)
            let residual = argMax(
                masked + MLXRandom.gumbel(masked.shape, type: Float.self), axis: -1)
            let lastRow = logp[(k - 1)..., 0...]
            let final = argMax(
                lastRow + MLXRandom.gumbel(lastRow.shape, type: Float.self), axis: -1)
            let uniforms = MLXRandom.uniform(low: Float(0), high: 1, [max(k - 1, 0)])
            return (pDraft, residual, final, uniforms)
        }
        // One materialization for all four tiny outputs (K-sized).
        eval(pDraftArr, residualArr, finalArr, uniformsArr)
        return (
            pDraft: pDraftArr.asArray(Float.self),
            uniforms: uniformsArr.asArray(Float.self),
            residual: residualArr.asArray(Int32.self).map { Int($0) },
            final: finalArr.item(Int.self)
        )
    }
}

extension Qwen35MTP {
    /// Verify capture strategy — memory-tight hosts (iPhone) should use `.rejectReforward`.
    /// nonisolated(unsafe): set once at startup before any generation.
    public nonisolated(unsafe) static var captureMode: Qwen35CaptureMode = .kernelScan

    /// Whether the decode loop's memory telemetry resets the program-global
    /// `GPU.peakMemory` counter at each round start so `Qwen35MTPResult.memPeakRound*`
    /// report true per-round high-water marks. Costs one allocator-mutex acquisition per
    /// round but CLOBBERS the global peak for any outer observer — hosts that track a
    /// process-wide peak across a generate should disable it (the per-round peak fields
    /// then stay 0). nonisolated(unsafe): set once at startup before any generation.
    public nonisolated(unsafe) static var memoryTelemetryResetsPeak = true

    /// MTP self-speculative decode. With `sampling == nil` it is exact GREEDY: the emitted
    /// token ids are identical to plain greedy decode of the target. With `sampling` set it
    /// is exact SAMPLED: the emitted stream is a true sample from the same
    /// temperature/top-p/repetition-penalty distribution the standard (non-speculative)
    /// sampled decode produces — speculation changes speed, never the distribution.
    /// Returns the generated tokens (excluding the prompt) plus draft proposal/acceptance
    /// counts.
    public static func generate(
        target: Qwen35Model, drafter: Qwen35DraftModel,
        promptTokens: [Int], maxTokens: Int, eosTokens: Set<Int>,
        blockSize: Int? = nil,
        prefillStep: Int = 256,
        adaptiveCeiling: Int = 6,
        sampling: Sampling? = nil,
        session: Qwen35MTPSession? = nil,
        onTokens: (([Int]) -> Bool)? = nil
    ) -> Qwen35MTPResult {
        let verifier = sampling.flatMap { $0.temperature > 0 ? Qwen35SampledVerifier(sampling: $0) : nil }
        let ringSize = sampling?.repetitionContextSize ?? 0
        // Repetition ring: the last `ringSize` tokens of the TEXT stream (prompt + generated),
        // exactly what RepetitionContext tracks on the standard path.
        var ring: [Int] = ringSize > 0 ? Array(promptTokens.suffix(ringSize)) : []
        func ringAppend(_ tokens: [Int]) {
            guard ringSize > 0 else { return }
            ring.append(contentsOf: tokens)
            if ring.count > ringSize { ring.removeFirst(ring.count - ringSize) }
        }
        // Resume when the prompt strictly extends the session's consumed tokens: keep the
        // caches (and the drafter's) and prefill only the suffix. Otherwise start fresh
        // (and re-prime the session so the NEXT step can resume).
        let resuming = session?.canResume(prompt: promptTokens) ?? false
        if !resuming {
            // Release the stale session's ~350 MB of caches BEFORE allocating fresh ones —
            // holding both across the prefill doubles the peak and trips jetsam on-device.
            session?.caches = []
            session?.tokens = []
        }
        let targetCache = resuming ? session!.caches : target.newCache(parameters: nil)
        let suffixStart = resuming ? session!.tokens.count : 0
        let suffixTokens = Array(promptTokens[suffixStart...])

        // Prefill the target over the (suffix of the) prompt IN CHUNKS — a single full forward
        // spikes activation memory and OOM-kills memory-tight devices (iPhone: ~400 MB
        // headroom) on long RAG prompts. Collect the per-position hidden states (the drafter
        // is prefilled from them).
        let suffixArr = MLXArray(suffixTokens.map { Int32($0) }).reshaped([1, suffixTokens.count])
        var hiddenChunks: [MLXArray] = []
        var prefilled = 0
        while prefilled < suffixTokens.count {
            let endIdx = min(prefilled + prefillStep, suffixTokens.count)
            // Hidden states only — the LM head runs ONCE below, on the final position. Computing
            // full-vocab logits for every prompt position wasted an LM-head matmul (~0.6 GFLOP
            // per position) plus a [chunk, vocab] logits buffer per chunk.
            let h = target.hidden(suffixArr[0..., prefilled ..< endIdx], cache: targetCache)
            eval(h)
            hiddenChunks.append(h)
            GPU.clearCache()
            prefilled = endIdx
        }
        let suffixHidden = concatenated(hiddenChunks, axis: 1)
        eval(suffixHidden)
        let lastHidden = suffixHidden[0..., (suffixHidden.dim(1) - 1)..., 0...]
        var bonus: Int
        if let verifier {
            bonus = verifier.sampleToken(logits: target.logits(fromHidden: lastHidden), ring: ring)
        } else {
            bonus = argMax(target.logits(fromHidden: lastHidden), axis: -1).item(Int.self)
        }
        ringAppend([bonus])

        var output: [Int] = [bonus]
        if !resuming {
            drafter.reset(target: target)
        }
        // On resume this EXTENDS the drafter cache: pairing starts at the suffix (the drafter
        // already holds pairs up to the session's last token) and re-seeds from the new end.
        drafter.prefillFromTargetHidden(
            inputIds: suffixArr, hidden: suffixHidden, bonusToken: bonus,
            prefillStep: prefillStep)

        // The session now represents exactly the full prompt.
        session?.caches = targetCache
        session?.tokens = promptTokens
        session?.invalidated = false

        var proposed = 0
        var accepted = 0
        var targetForwards = 0  // full-model forwards in the decode loop (verify + re-forwards)
        var rounds = 0
        // Per-phase wall clocks (see Qwen35MTPResult) — each phase ends at its sync barrier,
        // so lazy graph cost lands on the phase that forces it.
        var draftSeconds = 0.0
        var verifySeconds = 0.0
        var rollbackSeconds = 0.0
        // Per-round allocator-counter sampling at the same barriers as the phase clocks
        // (see Qwen35MTPResult's memory fields) — pure counter reads, trivial overhead.
        var memSampler = Qwen35DecodeMemorySampler(
            resetPeakPerRound: Qwen35MTP.memoryTelemetryResetsPeak)
        // Carried across steps via the session (parallel per-round histories).
        var acceptLens: [Int] = session?.acceptLens ?? []
        var proposedLens: [Int] = session?.proposedLens ?? []
        // Adaptive grows from the drafter's trained depth toward `adaptiveCeiling`; each unit
        // of ceiling costs ~layers × 8.4 MB of verify-capture transient — cap it on-device.
        // Stream the first bonus token; `onTokens` returning false requests cancellation.
        var cancelled = (onTokens?([bonus]) == false)

        while !cancelled && output.count < maxTokens && !eosTokens.contains(bonus) {
            // Effective block this round: a fixed `blockSize` if requested, else Adaptive —
            // a staircase from the drafter's trained depth toward the ceiling, stepping on
            // the recent full-block hit rate (see `effectiveBlockSize`).
            let remaining = maxTokens - output.count
            let effBlock: Int
            if let bs = blockSize, bs > 0 {
                effBlock = min(bs, remaining)
            } else {
                effBlock = effectiveBlockSize(
                    requestedBlock: adaptiveCeiling, configuredBlock: drafter.blockSize,
                    acceptLens: acceptLens, proposedLens: proposedLens,
                    remainingBudget: remaining)
            }
            rounds += 1
            memSampler.roundBegin()
            let tDraft = CFAbsoluteTimeGetCurrent()
            let draftArr = drafter.draftBlock(blockSize: effBlock)  // [1, effBlock-1]
            let draftTokens = draftArr.asArray(Int32.self).map { Int($0) }
            draftSeconds += CFAbsoluteTimeGetCurrent() - tDraft
            memSampler.afterDraftSync()
            proposed += draftTokens.count

            // Verify [bonus, draft_0 .. draft_{K-2}] in one forward (advances target cache by K).
            let verifyInput = [bonus] + draftTokens
            let verifyArr = MLXArray(verifyInput.map { Int32($0) })
                .reshaped([1, verifyInput.count])
            // Read the true pre-verify offset from a full-attention cache (the gated-delta
            // MambaCache does not track `offset` via `advance`).
            let preVerifyOffset = targetCache.compactMap { ($0 as? KVCacheSimple)?.offset }.first ?? 0
            // .kernelScan/.lazyRescan arm per-step capture on the gated-delta caches so a
            // rejection rolls back without a re-forward — at a capture tax on EVERY verify.
            // .rejectReforward and the probe run the verify capture-free (the fast path).
            let capture: Bool
            switch Qwen35MTP.captureMode {
            case .kernelScan, .lazyRescan: capture = true
            case .rejectReforward, .unsafeNoCapture: capture = false
            }
            for c in targetCache { (c as? MambaCache)?.captureVerify = capture }
            // .rejectReforward: retain the pre-verify gated-delta state refs, one pair per
            // Mamba cache. MLX arrays are immutable and the layer REPLACES cache[0]/cache[1]
            // on update, so holding the references is a free, already-materialized snapshot —
            // only consulted when this round rejects (see `rollbackVerifiedCaches`).
            let preVerifyStates: [(conv: MLXArray?, ssm: MLXArray?)] =
                Qwen35MTP.captureMode == .rejectReforward
                ? targetCache.compactMap { $0 as? MambaCache }.map { ($0[0], $0[1]) } : []
            targetForwards += 1
            let tVerify = CFAbsoluteTimeGetCurrent()
            let (verifyHidden, verifyLogits) = target.hiddenAndLogits(verifyArr, cache: targetCache)
            let budget = maxTokens - output.count
            let acc: Int
            let newToksRaw: [Int]
            if let verifier {
                // Sampled verify: reduce the [1, K, vocab] logits to the tiny per-round
                // outcome vectors inside the same evaluation as the hidden states (the
                // captured per-step states stay lazy, as in the greedy path). Row i is only
                // consumed when drafts 0..<i were all accepted, so including the in-block
                // draft prefix in its repetition ring matches sequential decode exactly.
                let ringRows = (0 ..< verifyInput.count).map { i -> [Int] in
                    let row = ring + draftTokens.prefix(i)
                    return Array(row.suffix(max(ringSize, 1)))
                }
                let outcome = verifier.outcomes(
                    verifyLogits: verifyLogits, draftTokens: draftTokens, ringRows: ringRows)
                eval(verifyHidden)
                verifySeconds += CFAbsoluteTimeGetCurrent() - tVerify
                (acc, newToksRaw) = speculativeWalkSampled(
                    draftTokens: draftTokens, pDraft: outcome.pDraft,
                    uniforms: outcome.uniforms, residualSamples: outcome.residual,
                    finalSample: outcome.final, budget: budget)
            } else {
                // ONE sync per verify: fold the argmax into the same evaluation as the hidden
                // states, so the full [1, K, vocab] logits are reduced inside the graph and the
                // captured per-step states stay lazy — only the SELECTED rollback state is
                // materialized below (on full acceptance the captures are discarded without
                // ever being computed).
                let targetPredsArr = argMax(verifyLogits, axis: -1).asType(.int32)
                eval(verifyHidden, targetPredsArr)
                verifySeconds += CFAbsoluteTimeGetCurrent() - tVerify
                let targetPreds = targetPredsArr.asArray(Int32.self).map { Int($0) }
                (acc, newToksRaw) = speculativeWalk(
                    draftTokens: draftTokens, targetTokens: targetPreds, budget: budget)
            }
            memSampler.sample(.afterVerify)
            accepted += acc
            acceptLens.append(acc)
            proposedLens.append(draftTokens.count)
            session?.acceptLens = acceptLens
            session?.proposedLens = proposedLens
            // Stop at the first EOS among the committed tokens. EOS can be an accepted *draft*
            // mid-block (not just the round's last token), so the `bonus`-only check misses it —
            // which is why generation ran past `<|im_end|>`. Emit up to (not including) EOS.
            var newToks = newToksRaw
            var hitEOS = false
            if let eosIdx = newToks.firstIndex(where: { eosTokens.contains($0) }) {
                newToks = Array(newToks.prefix(eosIdx))
                hitEOS = true
            }
            output.append(contentsOf: newToks)
            ringAppend(newToks)
            // Stream this round's committed tokens (accepted drafts + the correction).
            if !newToks.isEmpty, onTokens?(newToks) == false { cancelled = true }
            if hitEOS {
                // Roll the caches back to just before the EOS token — the same machinery as a
                // rejection — so the session STAYS resumable: tool-call steps routinely end
                // fence + im_end in one round, and the next step's delta re-supplies the
                // turn-close explicitly. The drafter skipped its round update (stale).
                let keep = newToks.count  // tokens kept from the verify block, before EOS
                if draftTokens.count - keep > 0 {
                    let tRollback = CFAbsoluteTimeGetCurrent()
                    targetForwards += rollbackVerifiedCaches(
                        target: target, targetCache: targetCache,
                        preVerifyStates: preVerifyStates, preVerifyOffset: preVerifyOffset,
                        verifyInput: verifyInput, keep: keep)
                    rollbackSeconds += CFAbsoluteTimeGetCurrent() - tRollback
                }
                memSampler.sample(.afterRollback)
                memSampler.roundEnd()
                for c in targetCache {
                    guard let mamba = c as? MambaCache else { continue }
                    mamba.captureVerify = false
                    mamba.capturedConv = []
                    mamba.capturedSSM = []
                }
                // Align the drafter exactly like a rejection at `keep` (newToks are all drafts
                // here — the correction never survives an EOS trim), so it keeps its full-turn
                // context for the next step. Empty newTokens: the next step's suffix prefill
                // re-seeds it.
                drafter.acceptVerifiedTokens(
                    verifyHidden: verifyHidden, draftTokens: draftArr,
                    accepted: keep, newTokens: [])
                session?.tokens += [bonus] + draftTokens.prefix(keep)
                session?.pendingFinalToken = nil
                break
            }

            // Roll back the target cache to EXACTLY [bonus] + accepted drafts — the correction
            // is the NEXT bonus and stays out of the cache. On full acceptance the cache already
            // holds [bonus, all drafts]; otherwise `rollbackVerifiedCaches` applies the
            // mode's rollback (captured-state restore, accepted-prefix re-forward, or the
            // probe's offset-only fixup).
            if acc < draftTokens.count {
                let tRollback = CFAbsoluteTimeGetCurrent()
                targetForwards += rollbackVerifiedCaches(
                    target: target, targetCache: targetCache,
                    preVerifyStates: preVerifyStates, preVerifyOffset: preVerifyOffset,
                    verifyInput: verifyInput, keep: acc)
                rollbackSeconds += CFAbsoluteTimeGetCurrent() - tRollback
            }
            memSampler.sample(.afterRollback)
            memSampler.roundEnd()
            // Clear capture state for the next round.
            for c in targetCache {
                guard let mamba = c as? MambaCache else { continue }
                mamba.captureVerify = false
                mamba.capturedConv = []
                mamba.capturedSSM = []
            }
            // The drafter's own cache update only affects draft QUALITY (acceptance), never the
            // emitted tokens — the target verify above guarantees exactness regardless.
            drafter.acceptVerifiedTokens(
                verifyHidden: verifyHidden, draftTokens: draftArr,
                accepted: acc, newTokens: newToks)

            // The caches now hold exactly [.. round bonus + accepted drafts] — mirror that in
            // the session; the round's correction becomes the next bonus and stays out (pending).
            session?.tokens += [bonus] + draftTokens.prefix(acc)
            session?.pendingFinalToken = newToks.last

            bonus = newToks.last ?? bonus
        }

        var result = Qwen35MTPResult(
            tokens: Array(output.prefix(maxTokens)), proposed: proposed, accepted: accepted,
            targetForwards: targetForwards, rounds: rounds,
            resumedSuffix: resuming ? suffixTokens.count : -1,
            draftSeconds: draftSeconds, verifySeconds: verifySeconds,
            rollbackSeconds: rollbackSeconds)
        memSampler.write(into: &result)
        return result
    }

    /// Roll the target caches back so they hold exactly `[round bonus] + the first `keep`
    /// drafts` of the verify block, offset `preVerifyOffset + keep + 1`. Shared by the
    /// rejected-round and mid-block-EOS paths; branches on the capture mode:
    ///
    /// - `.kernelScan` / `.lazyRescan`: gated-delta caches restore the captured per-step
    ///   conv + SSM state at index `keep` (the state after [bonus] + kept drafts);
    ///   full-attention caches `trim` the rejected tail. Only the SELECTED state is
    ///   materialized — one per gated-delta layer instead of all K — cutting it loose from
    ///   the verify graph before the captures are cleared; states past `keep` are never
    ///   computed. Empty captures = the `.unsafeNoCapture` probe: fix the offset only and
    ///   keep the (stale) states — timing stays honest, the stream does not.
    /// - `.rejectReforward`: restore the retained pre-verify refs, trim the FULL verify
    ///   block off the full-attention caches, and re-forward the kept prefix
    ///   `[bonus] + drafts.prefix(keep)` capture-free. The scan kernel consumes tokens
    ///   sequentially in fp32 registers, so scan-of-prefix ≡ prefix-of-scan — the rebuild
    ///   recomputes the same states/KV a capture would have selected (identical math;
    ///   measured ~1e-7 apart from re-projecting at a different block length, the same
    ///   scheduling-noise class as block-vs-token decode — see Qwen35RejectReforwardTests).
    ///   Its hidden output is NOT needed — the caller keeps using `verifyHidden` slices
    ///   for the kept prefix.
    ///
    /// Returns the number of extra full-model target forwards performed (1 on
    /// `.rejectReforward`, else 0) so the caller can count it in `targetForwards`.
    static func rollbackVerifiedCaches(
        target: Qwen35Model, targetCache: [KVCache],
        preVerifyStates: [(conv: MLXArray?, ssm: MLXArray?)],
        preVerifyOffset: Int, verifyInput: [Int], keep: Int
    ) -> Int {
        let keepOffset = preVerifyOffset + keep + 1
        if Qwen35MTP.captureMode == .rejectReforward {
            var mambaIdx = 0
            for c in targetCache {
                if let mamba = c as? MambaCache {
                    let pre = preVerifyStates[mambaIdx]
                    mambaIdx += 1
                    mamba[0] = pre.conv
                    mamba[1] = pre.ssm
                } else if let base = c as? BaseKVCache, base.isTrimmable {
                    _ = base.trim(verifyInput.count)  // the re-forward re-writes the kept prefix
                }
            }
            let prefix = verifyInput.prefix(keep + 1).map { Int32($0) }
            let prefixArr = MLXArray(Array(prefix)).reshaped([1, prefix.count])
            let h = target.hidden(prefixArr, cache: targetCache)
            eval(h)
            // `ArraysCache.advance` does not track `offset` — pin it for cache/session
            // parity with the capture modes' rollback.
            for c in targetCache {
                guard let mamba = c as? MambaCache else { continue }
                (mamba as BaseKVCache).offset = keepOffset
            }
            return 1
        }
        let drop = verifyInput.count - 1 - keep  // rejected drafts to drop from the verify block
        var selectedStates: [MLXArray] = []
        for c in targetCache {
            if let mamba = c as? MambaCache, !mamba.capturedConv.isEmpty {
                mamba[0] = mamba.capturedConv[keep]
                mamba[1] = mamba.capturedSSM[keep]
                (mamba as BaseKVCache).offset = keepOffset
                selectedStates.append(mamba.capturedConv[keep])
                selectedStates.append(mamba.capturedSSM[keep])
            } else if let mamba = c as? MambaCache {
                (mamba as BaseKVCache).offset = keepOffset
            } else if let base = c as? BaseKVCache, base.isTrimmable {
                _ = base.trim(drop)
            }
        }
        if !selectedStates.isEmpty { eval(selectedStates) }
        return 0
    }
}
