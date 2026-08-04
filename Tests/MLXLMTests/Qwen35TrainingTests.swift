// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35TrainingTests: XCTestCase {
    func testChunkedCausalAttentionMatchesStockValuesAndGradients() {
        MLXRandom.seed(81)
        let queries = (MLXRandom.normal([1, 4, 7, 8]) * MLXArray(0.1)).asType(.float32)
        let keys = (MLXRandom.normal([1, 2, 7, 8]) * MLXArray(0.1)).asType(.float32)
        let values = (MLXRandom.normal([1, 2, 7, 8]) * MLXArray(0.1)).asType(.float32)
        let primals = [queries, keys, values]
        let cotangent = MLXRandom.normal([1, 4, 7, 8]).asType(.float32)
        let scale = Float(1 / sqrt(8.0))

        let (stock, stockGradients) = vjp(
            { arrays in
                [
                    MLXFast.scaledDotProductAttention(
                        queries: arrays[0], keys: arrays[1], values: arrays[2],
                        scale: scale, mask: .causal)
                ]
            },
            primals: primals,
            cotangents: [cotangent]
        )
        let (chunked, chunkedGradients) = vjp(
            { arrays in
                [
                    qwen35MemoryEfficientAttention(
                        queries: arrays[0], keys: arrays[1], values: arrays[2],
                        scale: scale, causal: true, queryChunkSize: 3)
                ]
            },
            primals: primals,
            cotangents: [cotangent]
        )
        eval(stock + chunked + stockGradients + chunkedGradients)

        XCTAssertLessThan(maximumDifference(stock[0], chunked[0]), 2e-5)
        for (stockGradient, chunkedGradient) in zip(stockGradients, chunkedGradients) {
            XCTAssertLessThan(maximumDifference(stockGradient, chunkedGradient), 3e-5)
        }
    }

    func testGatedCheckpointMatchesUncheckpointedValuesAndGradients() {
        MLXRandom.seed(82)
        let weight = (MLXRandom.normal([6, 4]) * MLXArray(0.1)).asType(.float32)
        let input = (MLXRandom.normal([3, 6]) * MLXArray(0.1)).asType(.float32)
        let primals = [weight, input]
        let cotangent = MLXRandom.normal([3, 4]).asType(.float32)
        let body: ([MLXArray]) -> [MLXArray] = { arrays in
            [tanh(matmul(arrays[1], arrays[0]))]
        }

        let (plain, plainGradients) = vjp(
            body, primals: primals, cotangents: [cotangent])
        let (checkpointed, checkpointedGradients) = vjp(
            { arrays in qwen35GatedCheckpoint(arrays, body: body) },
            primals: primals,
            cotangents: [cotangent]
        )
        eval(plain + checkpointed + plainGradients + checkpointedGradients)

        XCTAssertLessThan(maximumDifference(plain[0], checkpointed[0]), 1e-6)
        for (plainGradient, checkpointedGradient) in zip(
            plainGradients, checkpointedGradients)
        {
            XCTAssertLessThan(maximumDifference(plainGradient, checkpointedGradient), 1e-6)
        }
    }

    func testTrainingExecutionContextIsScoped() {
        XCTAssertNil(Qwen35TrainingExecutionContext.configuration)
        Qwen35TrainingExecutionContext.withMemoryEfficientOperations(
            configuration: .init(
                attentionQueryChunkSize: 17,
                minimumChunkedAttentionTokens: 33,
                gradientCheckpointing: true)
        ) {
            XCTAssertEqual(
                Qwen35TrainingExecutionContext.configuration?.attentionQueryChunkSize, 17)
        }
        XCTAssertNil(Qwen35TrainingExecutionContext.configuration)
    }

    func testTinyLoRAModelBackpropagatesThroughOptimizedTrainingStack() throws {
        let json = """
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 16,
                    "num_hidden_layers": 2,
                    "intermediate_size": 32,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "linear_num_value_heads": 2,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 32,
                    "linear_value_head_dim": 4,
                    "linear_conv_kernel_dim": 4,
                    "vocab_size": 64,
                    "full_attention_interval": 2,
                    "tie_word_embeddings": false
                }
            }
            """
        let configuration = try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
        let model = Qwen35Model(configuration)
        _ = try LoRAContainer.from(
            model: model,
            configuration: .init(numLayers: 2, loraParameters: .init(rank: 2)))
        let tokens = MLXArray((0 ..< 40).map { Int32($0 % 64) })[.newAxis, 0...]
        let valueAndGradient = valueAndGrad(model: model as Module) { module, input in
            let qwen = module as! Qwen35Model
            return Qwen35TrainingExecutionContext.withMemoryEfficientOperations(
                configuration: .init(
                    attentionQueryChunkSize: 8,
                    minimumChunkedAttentionTokens: 16,
                    gradientCheckpointing: true)
            ) {
                GatedDeltaExecutionContext.withDifferentiableOperations {
                    [qwen(input, cache: nil).asType(.float32).square().mean()]
                }
            }
        }

        let (loss, gradients) = valueAndGradient(model, tokens)
        let flattened = gradients.flattened().map(\.1)
        eval(loss + flattened)

        XCTAssertTrue(loss[0].item(Float.self).isFinite)
        XCTAssertFalse(flattened.isEmpty)
        XCTAssertTrue(
            flattened.allSatisfy { $0.asArray(Float.self).allSatisfy(\.isFinite) })
    }

    private func maximumDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
    }
}
