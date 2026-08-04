// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

public class GatedDeltaTests: XCTestCase {

    private struct Inputs {
        let q, k, v, a, b, aLog, dtBias: MLXArray
    }

    /// Build deterministic bf16 inputs shaped for the GDN entry points.
    /// Hk/Hv/Dk/Dv stay tiny so the kernel dispatches but the test runs in ms.
    private func makeInputs(
        B: Int = 1, T: Int = 16, Hk: Int = 2, Dk: Int = 32,
        Hv: Int = 4, Dv: Int = 16, seed: UInt64 = 42
    ) -> Inputs {
        MLXRandom.seed(seed)
        let dtype = DType.bfloat16
        let q = MLXRandom.normal([B, T, Hk, Dk]).asType(dtype)
        let k = MLXRandom.normal([B, T, Hk, Dk]).asType(dtype)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(dtype)
        let a = MLXRandom.normal([B, T, Hv]).asType(dtype)
        let b = MLXRandom.normal([B, T, Hv]).asType(dtype)
        let aLog = (MLXRandom.normal([Hv]) * MLXArray(0.1)).asType(dtype)
        let dtBias = MLXRandom.normal([Hv]).asType(dtype)
        return Inputs(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias)
    }

    /// Multi-chunk prefill must match single-chunk prefill at the same total T.
    ///
    /// Regression for the fp32-state fix. Pre-fix, `gatedDeltaKernel` wrote
    /// `state_out` as `InT` (bf16) and `gatedDeltaUpdate` defaulted state to
    /// `q.dtype` (bf16). When a second-chunk prefill reloaded that state, it
    /// arrived bf16-quantized; the fp32 scratch recurrence then ran from a
    /// degraded starting point. With this test's inputs the cross-chunk drift
    /// vs a single full-length prefill is >10 max abs. Post-fix, state crosses
    /// the chunk boundary as fp32 and the two paths match within bf16 input
    /// quantization noise.
    func testGatedDeltaMultiChunkMatchesSingleChunk() throws {
        let T = 16
        let inputs = makeInputs(T: T)

        let (ySingle, _) = gatedDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v,
            a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias
        )
        eval(ySingle)

        let mid = T / 2
        let (y1, state1) = gatedDeltaUpdate(
            q: inputs.q[0..., ..<mid], k: inputs.k[0..., ..<mid], v: inputs.v[0..., ..<mid],
            a: inputs.a[0..., ..<mid], b: inputs.b[0..., ..<mid],
            aLog: inputs.aLog, dtBias: inputs.dtBias
        )
        let (y2, _) = gatedDeltaUpdate(
            q: inputs.q[0..., mid...], k: inputs.k[0..., mid...], v: inputs.v[0..., mid...],
            a: inputs.a[0..., mid...], b: inputs.b[0..., mid...],
            aLog: inputs.aLog, dtBias: inputs.dtBias,
            state: state1
        )
        let yMulti = concatenated([y1, y2], axis: 1)
        eval(yMulti)

        let diff = abs(ySingle.asType(.float32) - yMulti.asType(.float32)).max()
        eval(diff)
        let maxDiff = diff.item(Float.self)

        // Pre-fix: bf16-cast at chunk boundary diverges by >10 max abs.
        // Post-fix: fp32 state across the boundary leaves only bf16 input noise.
        XCTAssertLessThan(
            maxDiff, 1e-2,
            "Multi-chunk GDN prefill diverged from single-chunk by \(maxDiff) max abs. "
                + "Cross-chunk state must persist in fp32; bf16 cast loses precision."
        )
    }

    /// A one-shot prefix scan (T = t+1) must end in EXACTLY the state a chained per-token
    /// scan reaches after t+1 tokens, and the full-length prefix must equal the whole-scan
    /// final state.
    ///
    /// This is the substitution the speculative-verify `.lazyRescan` capture relies on:
    /// rollback states are computed as independent prefix scans (one kernel per layer)
    /// instead of a sequential per-token chain. Both paths carry the recurrence in fp32
    /// (registers in the kernel, stored state between chained calls), so they must agree
    /// bit-for-bit — the emitted speculative tokens depend on it.
    func testPrefixScanMatchesPerTokenChainExactly() throws {
        let T = 6
        let inputs = makeInputs(T: T)

        // Chained per-token reference: state after each token, threading fp32 state.
        var running: MLXArray? = nil
        var chainStates: [MLXArray] = []
        for t in 0 ..< T {
            let (_, st) = gatedDeltaUpdate(
                q: inputs.q[0..., t ..< (t + 1)], k: inputs.k[0..., t ..< (t + 1)],
                v: inputs.v[0..., t ..< (t + 1)],
                a: inputs.a[0..., t ..< (t + 1)], b: inputs.b[0..., t ..< (t + 1)],
                aLog: inputs.aLog, dtBias: inputs.dtBias, state: running
            )
            running = st
            chainStates.append(st)
        }
        eval(chainStates)

        // Whole-scan final state (what the verify forward computes for the full block).
        let (_, finalState) = gatedDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v, a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias
        )

        for t in 0 ..< T {
            let (_, prefixState) = gatedDeltaUpdate(
                q: inputs.q[0..., ..<(t + 1)], k: inputs.k[0..., ..<(t + 1)],
                v: inputs.v[0..., ..<(t + 1)],
                a: inputs.a[0..., ..<(t + 1)], b: inputs.b[0..., ..<(t + 1)],
                aLog: inputs.aLog, dtBias: inputs.dtBias
            )
            let diff = abs(prefixState.asType(.float32) - chainStates[t].asType(.float32))
                .max().item(Float.self)
            XCTAssertEqual(
                diff, 0,
                "Prefix scan T=\(t + 1) diverged from the per-token chain by \(diff) max abs. "
                    + "Rollback states must be bit-identical to the sequential scan."
            )
        }

        let finalDiff = abs(finalState.asType(.float32) - chainStates[T - 1].asType(.float32))
            .max().item(Float.self)
        XCTAssertEqual(
            finalDiff, 0,
            "Whole-scan final state diverged from the per-token chain by \(finalDiff) max abs."
        )
    }

    /// Training must bypass the inference-only Metal scan, whose primitive has no VJP.
    func testDifferentiableExecutionSupportsGradient() throws {
        let inputs = makeInputs(T: 2, Hk: 1, Dk: 32, Hv: 1, Dv: 2)

        let gradient = GatedDeltaExecutionContext.withDifferentiableOperations {
            grad { query in
                gatedDeltaUpdate(
                    q: query, k: inputs.k, v: inputs.v,
                    a: inputs.a, b: inputs.b,
                    aLog: inputs.aLog, dtBias: inputs.dtBias
                ).0.sum()
            }(inputs.q)
        }
        eval(gradient)

        XCTAssertEqual(gradient.shape, inputs.q.shape)
        XCTAssertTrue(
            gradient.asArray(Float.self).allSatisfy { $0.isFinite },
            "Differentiable gated-delta fallback produced a non-finite gradient."
        )
    }

    /// The fused training primitive must remain gradient-equivalent to the transparent ops scan.
    /// This covers GQA head reduction and the recurrent-state cotangent, not just the output path.
    func testFusedTrainingGradientMatchesOpsReference() throws {
        let inputs = makeInputs(T: 5, Hk: 1, Dk: 32, Hv: 2, Dv: 4)
        let state = MLXRandom.normal([1, 2, 4, 32]).asType(.float32)
        let g = computeGatedDeltaG(inputs.aLog, inputs.a, inputs.dtBias)
        let beta = sigmoid(inputs.b).asType(.float32)
        let primals = [
            state, inputs.q.asType(.float32), inputs.k.asType(.float32),
            inputs.v.asType(.float32), g, beta,
        ]

        func loss(_ output: (MLXArray, MLXArray)) -> [MLXArray] {
            [
                (output.0.asType(.float32).square().mean()
                    + MLXArray(0.1) * output.1.square().mean())
            ]
        }

        let (_, reference) = vjp(
            { values in
                loss(
                    gatedDeltaOps(
                        q: values[1], k: values[2], v: values[3],
                        g: values[4], beta: values[5], state: values[0]))
            },
            primals: primals,
            cotangents: [MLXArray(1.0)]
        )
        let (_, optimized) = vjp(
            { values in
                loss(
                    gatedDeltaTrainingUpdate(
                        q: values[1], k: values[2], v: values[3],
                        g: values[4], beta: values[5], state: values[0]))
            },
            primals: primals,
            cotangents: [MLXArray(1.0)]
        )
        eval(reference + optimized)

        for (index, values) in zip(reference, optimized).enumerated() {
            let maximumError = abs(values.0.asType(.float32) - values.1.asType(.float32))
                .max().item(Float.self)
            XCTAssertLessThan(
                maximumError, 5e-3,
                "Fused gated-delta training gradient \(index) diverged by \(maximumError)."
            )
        }
    }

    /// A realistic trace prefix must not materialize one graph node chain per token. The old
    /// fallback crossed Metal's 499k live-resource ceiling in the end-to-end trainer.
    func testFusedTrainingGradientHandlesLongSequence() throws {
        MLXRandom.seed(73)
        let q = (MLXRandom.normal([1, 4_096, 1, 32]) * MLXArray(0.01)).asType(.float32)
        let k = (MLXRandom.normal([1, 4_096, 1, 32]) * MLXArray(0.01)).asType(.float32)
        let v = (MLXRandom.normal([1, 4_096, 1, 2]) * MLXArray(0.01)).asType(.float32)
        let g = MLXArray.full([1, 4_096, 1], values: MLXArray(0.99))
        let beta = MLXArray.full([1, 4_096, 1], values: MLXArray(0.1))
        let gradient = grad { query in
            return gatedDeltaTrainingUpdate(
                q: query, k: k, v: v, g: g, beta: beta
            ).0.square().mean()
        }(q)
        eval(gradient)

        XCTAssertEqual(gradient.shape, q.shape)
        XCTAssertTrue(gradient.asArray(Float.self).allSatisfy(\.isFinite))
    }

}
