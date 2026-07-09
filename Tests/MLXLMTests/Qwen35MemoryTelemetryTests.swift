// Copyright © 2026 Apple Inc.
//
// Memory-telemetry contract for the Qwen3.5 MTP decode loop.
//
// The retention-tax investigation needs per-round allocator counters next to the per-phase
// clocks: every state-RETAINING capture mode pays ~2x the verify time of the retain-nothing
// probe, and the working hypothesis is fresh-buffer wiring — the retained pre-verify
// gated-delta refs deny the buffer pool those buffers, so every verify creates (and wires)
// fresh Metal buffers instead of recycling. These tests pin the telemetry contract:
//
// 1. All memory fields on `Qwen35MTPResult` default to zero.
// 2. A real generate over a tiny model populates them (pure counter reads at the same
//    three sync barriers as the phase clocks).
// 3. The per-round peak fields honor `Qwen35MTP.memoryTelemetryResetsPeak`.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class Qwen35MemoryTelemetryTests: XCTestCase {

    /// Tiny hybrid target: layer 0 gated-delta (MambaCache), layer 1 full attention
    /// (KVCacheSimple) — same stack as Qwen35RejectReforwardLoopTests. The gated-delta
    /// kernel requires `linear_key_head_dim` divisible by 32.
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

    private func generate(
        target: Qwen35Model, drafter: Qwen35DraftModel, maxTokens: Int = 24
    ) -> Qwen35MTPResult {
        Qwen35MTP.generate(
            target: target, drafter: drafter, promptTokens: prompt,
            maxTokens: maxTokens, eosTokens: [], blockSize: 3,
            session: Qwen35MTPSession())
    }

    // MARK: - 1. Defaults

    /// A result that never went through the decode loop reports zero for every memory
    /// field — the bench treats 0 as "not measured".
    func testResultMemoryFieldsDefaultToZero() {
        let r = Qwen35MTPResult(tokens: [], proposed: 0, accepted: 0)

        XCTAssertEqual(r.memActiveBeforeDraftAvgMB, 0)
        XCTAssertEqual(r.memActiveBeforeDraftMaxMB, 0)
        XCTAssertEqual(r.memActiveAfterVerifyAvgMB, 0)
        XCTAssertEqual(r.memActiveAfterVerifyMaxMB, 0)
        XCTAssertEqual(r.memActiveAfterRollbackAvgMB, 0)
        XCTAssertEqual(r.memActiveAfterRollbackMaxMB, 0)
        XCTAssertEqual(r.memCacheBeforeDraftAvgMB, 0)
        XCTAssertEqual(r.memCacheAfterVerifyAvgMB, 0)
        XCTAssertEqual(r.memCacheAfterRollbackAvgMB, 0)
        XCTAssertEqual(r.memPeakRoundAvgMB, 0)
        XCTAssertEqual(r.memPeakRoundMaxMB, 0)
        XCTAssertEqual(r.memPeakDecodeGrowthMB, 0)
        XCTAssertEqual(r.memFreshRoundAvgMB, 0)
        XCTAssertEqual(r.memFreshRoundMaxMB, 0)
        XCTAssertEqual(r.memFreshVerifyRoundAvgMB, 0)
        XCTAssertEqual(r.memFreshVerifyRoundMaxMB, 0)
        XCTAssertEqual(r.memFreshTotalMB, 0)
        XCTAssertEqual(r.memReleasedTotalMB, 0)
    }

    // MARK: - 2. A real generate populates the telemetry

    func testGeneratePopulatesMemoryTelemetry() throws {
        MLXRandom.seed(3)
        let target = try makeTarget()
        let drafter = try makeDrafter()

        let saved = Qwen35MTP.captureMode
        defer { Qwen35MTP.captureMode = saved }
        Qwen35MTP.captureMode = .rejectReforward  // the shipping iPhone mode

        let res = generate(target: target, drafter: drafter)
        XCTAssertGreaterThan(res.rounds, 0, "the decode loop never ran — nothing to sample")

        // Active memory is sampled at all three barriers; a loaded model + live caches
        // guarantee it is non-zero, and a max of samples can never undercut their mean.
        for (avg, max, point) in [
            (res.memActiveBeforeDraftAvgMB, res.memActiveBeforeDraftMaxMB, "beforeDraft"),
            (res.memActiveAfterVerifyAvgMB, res.memActiveAfterVerifyMaxMB, "afterVerify"),
            (res.memActiveAfterRollbackAvgMB, res.memActiveAfterRollbackMaxMB, "afterRollback"),
        ] {
            XCTAssertGreaterThan(avg, 0, "active avg not populated at \(point)")
            XCTAssertGreaterThanOrEqual(max, avg, "active max < avg at \(point)")
        }

        // Cache samples are valid counters (may legitimately be 0 right after a purge).
        XCTAssertGreaterThanOrEqual(res.memCacheBeforeDraftAvgMB, 0)
        XCTAssertGreaterThanOrEqual(res.memCacheAfterVerifyAvgMB, 0)
        XCTAssertGreaterThanOrEqual(res.memCacheAfterRollbackAvgMB, 0)

        // Per-round peak: with the default reset-per-round flag the counter is a true
        // per-round high-water mark of active memory.
        XCTAssertTrue(Qwen35MTP.memoryTelemetryResetsPeak, "default flag changed?")
        XCTAssertGreaterThan(res.memPeakRoundAvgMB, 0)
        XCTAssertGreaterThanOrEqual(res.memPeakRoundMaxMB, res.memPeakRoundAvgMB)
        XCTAssertGreaterThanOrEqual(res.memPeakDecodeGrowthMB, 0)

        // Fresh-allocation estimate: the prefill clears the buffer pool, so round 1 MUST
        // obtain new buffers from Metal — the fresh total cannot be zero.
        XCTAssertGreaterThan(res.memFreshTotalMB, 0, "no fresh allocations recorded")
        XCTAssertGreaterThanOrEqual(res.memFreshRoundMaxMB, res.memFreshRoundAvgMB)
        XCTAssertGreaterThanOrEqual(res.memFreshVerifyRoundMaxMB, res.memFreshVerifyRoundAvgMB)
        XCTAssertGreaterThanOrEqual(res.memFreshVerifyRoundAvgMB, 0)
        // The verify phase's fresh bytes are a subset of the round's.
        XCTAssertGreaterThanOrEqual(res.memFreshRoundAvgMB, res.memFreshVerifyRoundAvgMB)
        // Total/avg consistency: total == avg × rounds.
        XCTAssertEqual(
            res.memFreshTotalMB, res.memFreshRoundAvgMB * Double(res.rounds),
            accuracy: 1e-6)

        XCTAssertGreaterThanOrEqual(res.memReleasedTotalMB, 0)
    }

    // MARK: - 3. Peak-reset flag

    /// With per-round peak resets disabled the program-global peak counter is left alone
    /// and the per-round peak fields stay 0; everything else still populates.
    func testPeakResetFlagOffZeroesRoundPeaks() throws {
        MLXRandom.seed(5)
        let target = try makeTarget()
        let drafter = try makeDrafter()

        let savedFlag = Qwen35MTP.memoryTelemetryResetsPeak
        defer { Qwen35MTP.memoryTelemetryResetsPeak = savedFlag }
        Qwen35MTP.memoryTelemetryResetsPeak = false

        let res = generate(target: target, drafter: drafter)
        XCTAssertGreaterThan(res.rounds, 0)

        XCTAssertEqual(res.memPeakRoundAvgMB, 0)
        XCTAssertEqual(res.memPeakRoundMaxMB, 0)
        // Actives still sampled; decode growth falls back to sampled actives (clamped ≥ 0).
        XCTAssertGreaterThan(res.memActiveBeforeDraftAvgMB, 0)
        XCTAssertGreaterThan(res.memActiveAfterVerifyAvgMB, 0)
        XCTAssertGreaterThanOrEqual(res.memPeakDecodeGrowthMB, 0)
    }
}
