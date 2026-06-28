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
    public func prefillFromTargetHidden(inputIds: MLXArray, hidden: MLXArray, bonusToken: Int) {
        guard inputIds.dim(1) > 0 else { return }
        let bonus = MLXArray([Int32(bonusToken)]).reshaped([1, 1])
        let shifted = concatenated([inputIds[0..., 1...].asType(.int32), bonus], axis: 1)
        let h = forwardTokens(shifted, hidden: hidden[0..., ..<shifted.dim(1), 0...])
        setSeedFromHidden(lastStep(h))
    }

    /// Draft `blockSize - 1` tokens greedily, starting from the carried seed.
    public func draftBlock() -> MLXArray {
        roundAppended = 0
        guard let s = seedToken, let sh = seedHidden else {
            return MLXArray.zeros([1, 0], type: Int32.self)
        }
        var tok = s
        var hPrev = sh
        var tokens: [MLXArray] = [tok]
        seedToken = nil
        seedHidden = nil
        while tokens.count < blockSize - 1 {
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
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
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
            if v.ndim == 1, normSuffixes.contains(where: { k.hasSuffix($0) }) {
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
}

extension Qwen35Model {
    public func hiddenAndLogits(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> (hidden: MLXArray, logits: MLXArray) {
        languageModel.hiddenAndLogits(inputs, cache: cache)
    }
}

// MARK: - Greedy speculative decode loop (Stage 5)

public struct Qwen35MTPResult {
    public let tokens: [Int]
    public let proposed: Int
    public let accepted: Int
}

extension Qwen35MTP {
    /// Greedy MTP self-speculative decode. Exact: the emitted token ids are identical to plain
    /// greedy decode of the target. Returns the generated tokens (excluding the prompt) plus
    /// draft proposal/acceptance counts.
    public static func generate(
        target: Qwen35Model, drafter: Qwen35DraftModel,
        promptTokens: [Int], maxTokens: Int, eosTokens: Set<Int>,
        onTokens: (([Int]) -> Bool)? = nil
    ) -> Qwen35MTPResult {
        let targetCache = target.newCache(parameters: nil)

        // Prefill the target over the prompt; first bonus = greedy argmax of the last position.
        let promptArr = MLXArray(promptTokens.map { Int32($0) }).reshaped([1, promptTokens.count])
        let (promptHidden, promptLogits) = target.hiddenAndLogits(promptArr, cache: targetCache)
        eval(promptHidden, promptLogits)
        var bonus = argMax(promptLogits[0..., (promptTokens.count - 1)..., 0...], axis: -1)
            .item(Int.self)

        var output: [Int] = [bonus]
        drafter.reset(target: target)
        drafter.prefillFromTargetHidden(
            inputIds: promptArr, hidden: promptHidden, bonusToken: bonus)

        var proposed = 0
        var accepted = 0
        // Stream the first bonus token; `onTokens` returning false requests cancellation.
        var cancelled = (onTokens?([bonus]) == false)

        while !cancelled && output.count < maxTokens && !eosTokens.contains(bonus) {
            let draftArr = drafter.draftBlock()  // [1, blockSize-1]
            let draftTokens = draftArr.asArray(Int32.self).map { Int($0) }
            proposed += draftTokens.count

            // Verify [bonus, draft_0 .. draft_{K-2}] in one forward (advances target cache by K).
            let verifyInput = [bonus] + draftTokens
            let verifyArr = MLXArray(verifyInput.map { Int32($0) })
                .reshaped([1, verifyInput.count])
            let snap = snapshotCaches(targetCache)
            let (verifyHidden, verifyLogits) = target.hiddenAndLogits(verifyArr, cache: targetCache)
            eval(verifyHidden, verifyLogits)
            let targetPreds = argMax(verifyLogits, axis: -1).asArray(Int32.self).map { Int($0) }

            let budget = maxTokens - output.count
            let (acc, newToks) = speculativeWalk(
                draftTokens: draftTokens, targetTokens: targetPreds, budget: budget)
            accepted += acc
            output.append(contentsOf: newToks)
            // Stream this round's committed tokens (accepted drafts + the correction).
            if onTokens?(newToks) == false { cancelled = true }

            // Target-cache invariant: at round start the current `bonus` is NOT in the cache;
            // the verify forward added [bonus, drafts]. Keep exactly [bonus] + accepted drafts
            // (the correction is the NEXT bonus and must stay out). On full acceptance the cache
            // already holds [bonus, all drafts] — nothing to do; otherwise roll back and re-commit.
            if acc < draftTokens.count {
                restoreCaches(targetCache, to: snap)
                let keep = [bonus] + newToks.dropLast()  // bonus + accepted drafts (drop correction)
                let keepArr = MLXArray(keep.map { Int32($0) }).reshaped([1, keep.count])
                let (keepHidden, _) = target.hiddenAndLogits(keepArr, cache: targetCache)
                eval(keepHidden)
            }
            // The drafter's own cache update only affects draft QUALITY (acceptance), never the
            // emitted tokens — the target verify above guarantees exactness regardless.
            drafter.acceptVerifiedTokens(
                verifyHidden: verifyHidden, draftTokens: draftArr,
                accepted: acc, newTokens: newToks)

            bonus = newToks.last ?? bonus
        }

        return Qwen35MTPResult(
            tokens: Array(output.prefix(maxTokens)), proposed: proposed, accepted: accepted)
    }
}
