// Copyright © 2026 Apple Inc.
//
// Speculative-verify capture contract for the Qwen3.5 gated-delta layer.
//
// A speculative rejection rolls the MambaCache back to `capturedConv[acc]` /
// `capturedSSM[acc]` — the state after consuming verify tokens 0...acc. Exactness of the
// emitted token stream rests on those captures matching the state the block forward
// implies at that position. Two capture modes must agree:
//
// - `.kernelScan`: the capture-variant kernel writes every post-token state straight from
//   the scan registers.
// - `.lazyRescan`: independent LAZY prefix scans (one kernel per index) recompute the
//   state from the block's own projected inputs; only the selected index is ever
//   materialized. (Previously a sequential per-token chain — the prefix-scan form must be
//   bit-identical to it AND to `.kernelScan`, since the scan consumes tokens sequentially
//   in fp32 either way.)

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class Qwen35VerifyCaptureTests: XCTestCase {

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

    /// Run `net` over `x` token-by-token (the plain decode path, no capture), recording the
    /// cache's conv + SSM state after each token — the ground truth a rollback must restore.
    private func sequentialStates(
        net: Qwen35GatedDeltaNet, x: MLXArray
    ) -> (conv: [MLXArray], ssm: [MLXArray]) {
        let cache = MambaCache()
        var conv: [MLXArray] = []
        var ssm: [MLXArray] = []
        for t in 0 ..< x.dim(1) {
            _ = net(x[0..., t ..< (t + 1), 0...], mask: nil, cache: cache)
            conv.append(cache[0]!)
            ssm.append(cache[1]!)
        }
        eval(conv + ssm)
        return (conv, ssm)
    }

    /// Run one block forward with `captureVerify` in the given mode, returning the captures
    /// and the cache's final state.
    private func capturedStates(
        net: Qwen35GatedDeltaNet, x: MLXArray, mode: Qwen35CaptureMode
    ) -> (conv: [MLXArray], ssm: [MLXArray], finalConv: MLXArray, finalSSM: MLXArray) {
        let saved = Qwen35MTP.captureMode
        Qwen35MTP.captureMode = mode
        defer { Qwen35MTP.captureMode = saved }

        let cache = MambaCache()
        cache.captureVerify = true
        _ = net(x, mask: nil, cache: cache)
        eval(cache.capturedConv + cache.capturedSSM)
        return (cache.capturedConv, cache.capturedSSM, cache[0]!, cache[1]!)
    }

    func testLazyRescanCapturesMatchSequentialDecode() throws {
        MLXRandom.seed(7)
        let net = try makeNet()
        let T = 5
        let x = MLXRandom.normal([1, T, 16])

        let truth = sequentialStates(net: net, x: x)
        let cap = capturedStates(net: net, x: x, mode: .lazyRescan)

        XCTAssertEqual(cap.conv.count, T)
        XCTAssertEqual(cap.ssm.count, T)
        for t in 0 ..< T {
            // Block forward vs per-token forwards: identical math per position in fp32;
            // tolerance covers kernel-shape-dependent op scheduling only.
            XCTAssertLessThan(
                maxAbsDiff(cap.conv[t], truth.conv[t]), 1e-5,
                "capturedConv[\(t)] diverged from the sequential-decode conv state")
            XCTAssertLessThan(
                maxAbsDiff(cap.ssm[t], truth.ssm[t]), 1e-5,
                "capturedSSM[\(t)] diverged from the sequential-decode SSM state")
        }
        // The final captured state IS the block's committed cache state (full acceptance).
        XCTAssertEqual(maxAbsDiff(cap.ssm[T - 1], cap.finalSSM), 0)
        XCTAssertEqual(maxAbsDiff(cap.conv[T - 1], cap.finalConv), 0)
    }

    /// The two capture modes consume the same block inputs and carry the recurrence in
    /// fp32, so their captures must agree bit-for-bit — a rollback must not depend on
    /// which mode the host selected.
    func testLazyRescanMatchesKernelScanExactly() throws {
        MLXRandom.seed(11)
        let net = try makeNet()
        let T = 5
        let x = MLXRandom.normal([1, T, 16])

        let lazy = capturedStates(net: net, x: x, mode: .lazyRescan)
        let kernel = capturedStates(net: net, x: x, mode: .kernelScan)

        XCTAssertEqual(lazy.ssm.count, kernel.ssm.count)
        for t in 0 ..< T {
            XCTAssertEqual(
                maxAbsDiff(lazy.ssm[t], kernel.ssm[t]), 0,
                "capturedSSM[\(t)] differs between .lazyRescan and .kernelScan")
            XCTAssertEqual(
                maxAbsDiff(lazy.conv[t], kernel.conv[t]), 0,
                "capturedConv[\(t)] differs between .lazyRescan and .kernelScan")
        }
        XCTAssertEqual(maxAbsDiff(lazy.finalSSM, kernel.finalSSM), 0)
    }
}
