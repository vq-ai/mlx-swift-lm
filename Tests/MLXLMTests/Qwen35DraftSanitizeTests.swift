// Copyright © 2026 Apple Inc.
//
// Guards the drafter checkpoint-format distinction in `Qwen35DraftModel.sanitize`:
//
// - ORIGINAL combined checkpoints carry the drafter under an `mtp.` prefix and store
//   RMSNorm weights in the raw Qwen3.5 `(1+w)` convention → sanitize must strip the
//   prefix AND apply the `+1` shift.
// - SPLIT/trained drafter dirs (mlx-vlm split.py, mtp_train.py) store EFFECTIVE norm
//   weights with unprefixed keys → sanitize must pass them through UNCHANGED. Pre-fix,
//   the shift was applied unconditionally, double-shifting every deployed drafter's
//   norms (measured: v3 acc@1 0.83→0.76 agentic / 0.61→0.48 prose; stock 0.68→0.44).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class Qwen35DraftSanitizeTests: XCTestCase {

    private func makeDrafter() throws -> Qwen35DraftModel {
        // Minimum-viable config: tiny dims keep module init cheap; every text_config
        // field has a decode default, so only the shape-critical ones are pinned.
        let json = """
            {
                "model_type": "qwen3_5_mtp",
                "block_size": 3,
                "tie_word_embeddings": false,
                "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "linear_num_value_heads": 2,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 4,
                    "linear_conv_kernel_dim": 4,
                    "vocab_size": 32,
                    "full_attention_interval": 1
                }
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35DraftConfiguration.self, from: Data(json.utf8))
        return Qwen35DraftModel(config)
    }

    /// Split-format checkpoints (unprefixed keys) already store effective norm weights —
    /// sanitize must NOT shift them again.
    func testSplitFormatNormsPassThroughUnchanged() throws {
        let model = try makeDrafter()
        let norm = MLXArray([Float](repeating: 0.5, count: 8))
        let fc = MLXArray.zeros([8, 16])
        let weights: [String: MLXArray] = [
            "pre_fc_norm_embedding.weight": norm,
            "pre_fc_norm_hidden.weight": norm,
            "norm.weight": norm,
            "layers.0.input_layernorm.weight": norm,
            "layers.0.self_attn.q_norm.weight": norm,
            "fc.weight": fc,
        ]

        let out = model.sanitize(weights: weights)

        for key in [
            "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight",
            "layers.0.input_layernorm.weight", "layers.0.self_attn.q_norm.weight",
        ] {
            let v = try XCTUnwrap(out[key], "missing \(key)")
            XCTAssertEqual(
                v.mean().item(Float.self), 0.5, accuracy: 1e-6,
                "split-format norm \(key) must pass through unshifted")
        }
        XCTAssertNotNil(out["fc.weight"])
    }

    /// Original combined checkpoints (`mtp.` prefix, raw `(1+w)` convention) get the
    /// prefix stripped and the `+1` shift applied — the pre-existing behavior.
    func testMTPPrefixedNormsAreShiftedAndStripped() throws {
        let model = try makeDrafter()
        let norm = MLXArray([Float](repeating: 0.5, count: 8))
        let weights: [String: MLXArray] = [
            "mtp.pre_fc_norm_embedding.weight": norm,
            "mtp.norm.weight": norm,
            "mtp.layers.0.input_layernorm.weight": norm,
            "mtp.fc.weight": MLXArray.zeros([8, 16]),
        ]

        let out = model.sanitize(weights: weights)

        for key in [
            "pre_fc_norm_embedding.weight", "norm.weight",
            "layers.0.input_layernorm.weight",
        ] {
            let v = try XCTUnwrap(out[key], "missing \(key)")
            XCTAssertEqual(
                v.mean().item(Float.self), 1.5, accuracy: 1e-6,
                "original-format norm \(key) must be shifted by +1")
        }
        // 2-D weights are never shifted; the prefix is stripped.
        XCTAssertNotNil(out["fc.weight"])
        XCTAssertEqual(
            out["fc.weight"]!.sum().item(Float.self), 0, accuracy: 1e-6)
        XCTAssertTrue(out.keys.allSatisfy { !$0.hasPrefix("mtp.") })
    }
}
