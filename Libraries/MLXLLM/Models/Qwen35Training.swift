// Copyright © 2026 Apple Inc.

import MLX

/// Memory-bounded execution settings for gradient-bearing Qwen3.5/Qwen3.6 work.
/// Inference remains on the existing optimized path unless this scope is explicitly entered.
public struct Qwen35TrainingConfiguration: Equatable, Sendable {
    public var attentionQueryChunkSize: Int
    public var minimumChunkedAttentionTokens: Int
    public var gradientCheckpointing: Bool

    public init(
        attentionQueryChunkSize: Int = 512,
        minimumChunkedAttentionTokens: Int = 1_024,
        gradientCheckpointing: Bool = true
    ) {
        precondition(attentionQueryChunkSize > 0)
        precondition(minimumChunkedAttentionTokens > 0)
        self.attentionQueryChunkSize = attentionQueryChunkSize
        self.minimumChunkedAttentionTokens = minimumChunkedAttentionTokens
        self.gradientCheckpointing = gradientCheckpointing
    }
}

/// Selects the long-context training path without changing normal inference behavior.
public enum Qwen35TrainingExecutionContext {
    @TaskLocal static var configuration: Qwen35TrainingConfiguration?

    public static func withMemoryEfficientOperations<Result>(
        configuration: Qwen35TrainingConfiguration = .init(),
        _ operation: () throws -> Result
    ) rethrows -> Result {
        try $configuration.withValue(configuration, operation: operation)
    }

    public static func withMemoryEfficientOperations<Result>(
        configuration: Qwen35TrainingConfiguration = .init(),
        _ operation: () async throws -> Result
    ) async rethrows -> Result {
        try await $configuration.withValue(configuration, operation: operation)
    }
}

/// Checkpoint whose recomputation is gated on the incoming cotangent. This prevents the lazy
/// scheduler from constructing several layer-sized recompute working sets at the same time.
func qwen35GatedCheckpoint(
    _ inputs: [MLXArray],
    body: @escaping ([MLXArray]) -> [MLXArray]
) -> [MLXArray] {
    let operation = CustomFunction {
        Forward(body)
        VJP { primals, cotangents in
            let gatedPrimals = depends(inputs: primals, dependencies: cotangents)
            return vjp(body, primals: gatedPrimals, cotangents: cotangents).1
        }
    }
    return operation(inputs)
}

/// Causal attention with a chunked forward and analytic, query-chunked reverse pass. It keeps
/// one `queryChunkSize x sequenceLength` score tile live rather than the full quadratic tensor.
func qwen35MemoryEfficientAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    causal: Bool,
    queryChunkSize: Int
) -> MLXArray {
    precondition(queryChunkSize > 0)
    let operation = CustomFunction {
        Forward { primals in
            [
                qwen35ChunkedAttentionForward(
                    queries: primals[0], keys: primals[1], values: primals[2],
                    scale: scale, causal: causal, queryChunkSize: queryChunkSize)
            ]
        }
        VJP { primals, cotangents in
            qwen35ChunkedAttentionVJP(
                primals: primals, cotangent: cotangents[0], scale: scale,
                causal: causal, queryChunkSize: queryChunkSize)
        }
    }
    return operation([queries, keys, values])[0]
}

private func qwen35ChunkedAttentionForward(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    causal: Bool,
    queryChunkSize: Int
) -> MLXArray {
    let batch = queries.dim(0)
    let queryHeads = queries.dim(1)
    let sequenceLength = queries.dim(2)
    let headDimension = queries.dim(3)
    let keyHeads = keys.dim(1)
    let repeats = queryHeads / keyHeads
    precondition(queryHeads.isMultiple(of: keyHeads))

    let queryGroups = queries.reshaped(batch, keyHeads, repeats, sequenceLength, headDimension)
    let keyGroups = expandedDimensions(keys, axis: 2)
    let valueGroups = expandedDimensions(values, axis: 2)
    var outputs: [MLXArray] = []
    outputs.reserveCapacity((sequenceLength + queryChunkSize - 1) / queryChunkSize)

    for start in stride(from: 0, to: sequenceLength, by: queryChunkSize) {
        let end = min(start + queryChunkSize, sequenceLength)
        let keyEnd = causal ? min(((end + 4_095) / 4_096) * 4_096, sequenceLength) : sequenceLength
        var queryChunk = queryGroups[0..., 0..., 0..., start ..< end, 0...]
        if let previous = outputs.last {
            queryChunk = depends(input: queryChunk, dependencies: [previous])
        }
        var scores =
            scale
            * matmul(
                queryChunk,
                keyGroups[0..., 0..., 0..., ..<keyEnd, 0...].transposed(0, 1, 2, 4, 3))
        if causal {
            scores = qwen35ApplyCausalMask(scores, rowStart: start, rowEnd: end, keyEnd: keyEnd)
        }
        let probabilities = softmax(scores, axis: -1, precise: true).asType(queries.dtype)
        outputs.append(
            matmul(probabilities, valueGroups[0..., 0..., 0..., ..<keyEnd, 0...]))
    }
    return concatenated(outputs, axis: 3).reshaped(
        batch, queryHeads, sequenceLength, headDimension)
}

private func qwen35ChunkedAttentionVJP(
    primals: [MLXArray],
    cotangent: MLXArray,
    scale: Float,
    causal: Bool,
    queryChunkSize: Int
) -> [MLXArray] {
    let queries = primals[0]
    let keys = primals[1]
    let values = primals[2]
    let batch = queries.dim(0)
    let queryHeads = queries.dim(1)
    let sequenceLength = queries.dim(2)
    let headDimension = queries.dim(3)
    let keyHeads = keys.dim(1)
    let repeats = queryHeads / keyHeads

    let workType = queries.dtype
    let queryGroups = queries.reshaped(batch, keyHeads, repeats, sequenceLength, headDimension)
    let keyGroups = expandedDimensions(keys, axis: 2)
    let valueGroups = expandedDimensions(values, axis: 2)
    let outputGradients =
        cotangent
        .reshaped(batch, keyHeads, repeats, sequenceLength, headDimension)
        .asType(workType)
    var keyGradient = MLXArray.zeros(
        [batch, keyHeads, 1, sequenceLength, headDimension], dtype: .float32)
    var valueGradient = MLXArray.zeros(
        [batch, keyHeads, 1, sequenceLength, headDimension], dtype: .float32)
    var queryGradients: [MLXArray] = []
    queryGradients.reserveCapacity((sequenceLength + queryChunkSize - 1) / queryChunkSize)

    for start in stride(from: 0, to: sequenceLength, by: queryChunkSize) {
        let end = min(start + queryChunkSize, sequenceLength)
        let keyEnd = causal ? min(((end + 4_095) / 4_096) * 4_096, sequenceLength) : sequenceLength
        let gated = depends(
            inputs: [
                queryGroups[0..., 0..., 0..., start ..< end, 0...],
                keyGroups[0..., 0..., 0..., ..<keyEnd, 0...],
                valueGroups[0..., 0..., 0..., ..<keyEnd, 0...],
            ],
            dependencies: [keyGradient, valueGradient]
        )
        let queryChunk = gated[0]
        let keyChunk = gated[1]
        let valueChunk = gated[2]
        var scores = scale * matmul(queryChunk, keyChunk.transposed(0, 1, 2, 4, 3))
        if causal {
            scores = qwen35ApplyCausalMask(scores, rowStart: start, rowEnd: end, keyEnd: keyEnd)
        }
        let probabilities = softmax(scores, axis: -1, precise: true).asType(workType)
        let outputChunk = outputGradients[0..., 0..., 0..., start ..< end, 0...]
        let probabilityGradient = matmul(outputChunk, valueChunk.transposed(0, 1, 2, 4, 3))
        let scoreGradient =
            probabilities
            * (probabilityGradient
                - (probabilityGradient * probabilities).sum(axis: -1, keepDims: true))
            * scale
        let queryGradient = matmul(scoreGradient, keyChunk)
        var keyContribution = matmul(
            scoreGradient.transposed(0, 1, 2, 4, 3), queryChunk
        ).sum(axis: 2, keepDims: true).asType(.float32)
        var valueContribution = matmul(
            probabilities.transposed(0, 1, 2, 4, 3), outputChunk
        ).sum(axis: 2, keepDims: true).asType(.float32)
        if keyEnd < sequenceLength {
            let padding = MLXArray.zeros(
                [batch, keyHeads, 1, sequenceLength - keyEnd, headDimension], dtype: .float32)
            keyContribution = concatenated([keyContribution, padding], axis: 3)
            valueContribution = concatenated([valueContribution, padding], axis: 3)
        }
        keyGradient = keyGradient + keyContribution
        valueGradient = valueGradient + valueContribution
        queryGradients.append(queryGradient)
    }

    let queryGradient = concatenated(queryGradients, axis: 3)
        .reshaped(batch, queryHeads, sequenceLength, headDimension)
    return [
        queryGradient.asType(queries.dtype),
        keyGradient[0..., 0..., 0, 0..., 0...].asType(keys.dtype),
        valueGradient[0..., 0..., 0, 0..., 0...].asType(values.dtype),
    ]
}

private func qwen35ApplyCausalMask(
    _ scores: MLXArray,
    rowStart: Int,
    rowEnd: Int,
    keyEnd: Int
) -> MLXArray {
    let rows = arange(rowStart, rowEnd)[.newAxis, 0..., .newAxis]
    let columns = arange(keyEnd)[.newAxis, .newAxis, 0...]
    let mask = columns .<= rows
    let negative = MLXArray(scores.dtype == .float16 ? -65_504 : -1e30).asType(scores.dtype)
    return MLX.where(mask, scores, negative)
}
