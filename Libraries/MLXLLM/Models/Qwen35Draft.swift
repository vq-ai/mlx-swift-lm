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
public final class Qwen35MTPSession {
    public internal(set) var caches: [KVCache] = []
    public internal(set) var tokens: [Int] = []
    /// Set when the caches stopped matching `tokens` exactly (EOS cut a verify block short).
    public internal(set) var invalidated = true

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
}

extension Qwen35MTP {
    /// Greedy MTP self-speculative decode. Exact: the emitted token ids are identical to plain
    /// greedy decode of the target. Returns the generated tokens (excluding the prompt) plus
    /// draft proposal/acceptance counts.
    public static func generate(
        target: Qwen35Model, drafter: Qwen35DraftModel,
        promptTokens: [Int], maxTokens: Int, eosTokens: Set<Int>,
        blockSize: Int? = nil,
        prefillStep: Int = 256,
        session: Qwen35MTPSession? = nil,
        onTokens: (([Int]) -> Bool)? = nil
    ) -> Qwen35MTPResult {
        // Resume when the prompt strictly extends the session's consumed tokens: keep the
        // caches (and the drafter's) and prefill only the suffix. Otherwise start fresh
        // (and re-prime the session so the NEXT step can resume).
        let resuming = session?.canResume(prompt: promptTokens) ?? false
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
        var bonus = argMax(target.logits(fromHidden: lastHidden), axis: -1).item(Int.self)

        var output: [Int] = [bonus]
        if !resuming {
            drafter.reset(target: target)
        }
        // On resume this EXTENDS the drafter cache: pairing starts at the suffix (the drafter
        // already holds pairs up to the session's last token) and re-seeds from the new end.
        drafter.prefillFromTargetHidden(
            inputIds: suffixArr, hidden: suffixHidden, bonusToken: bonus)

        // The session now represents exactly the full prompt.
        session?.caches = targetCache
        session?.tokens = promptTokens
        session?.invalidated = false

        var proposed = 0
        var accepted = 0
        var targetForwards = 0  // full-model forwards in the decode loop (verify + re-forwards)
        var rounds = 0
        var acceptLens: [Int] = []  // accepted drafts per round — drives the Adaptive block size
        let adaptiveCeiling = 6     // Adaptive grows from the drafter's trained depth toward this
        // Stream the first bonus token; `onTokens` returning false requests cancellation.
        var cancelled = (onTokens?([bonus]) == false)

        while !cancelled && output.count < maxTokens && !eosTokens.contains(bonus) {
            // Effective block this round: a fixed `blockSize` if requested, else Adaptive —
            // grow from the drafter's trained depth toward the ceiling only while recent rounds
            // keep fully accepting the base depth (see `effectiveBlockSize`).
            let remaining = maxTokens - output.count
            let effBlock: Int
            if let bs = blockSize, bs > 0 {
                effBlock = min(bs, remaining)
            } else {
                effBlock = effectiveBlockSize(
                    requestedBlock: adaptiveCeiling, configuredBlock: drafter.blockSize,
                    acceptLens: acceptLens, remainingBudget: remaining)
            }
            rounds += 1
            let draftArr = drafter.draftBlock(blockSize: effBlock)  // [1, effBlock-1]
            let draftTokens = draftArr.asArray(Int32.self).map { Int($0) }
            proposed += draftTokens.count

            // Verify [bonus, draft_0 .. draft_{K-2}] in one forward (advances target cache by K).
            let verifyInput = [bonus] + draftTokens
            let verifyArr = MLXArray(verifyInput.map { Int32($0) })
                .reshaped([1, verifyInput.count])
            // Enable per-step capture on the gated-delta caches so a rejection rolls back
            // without a re-forward. Read the true pre-verify offset from a full-attention
            // cache (the gated-delta MambaCache does not track `offset` via `advance`).
            let preVerifyOffset = targetCache.compactMap { ($0 as? KVCacheSimple)?.offset }.first ?? 0
            for c in targetCache { (c as? MambaCache)?.captureVerify = true }
            targetForwards += 1
            let (verifyHidden, verifyLogits) = target.hiddenAndLogits(verifyArr, cache: targetCache)
            // ONE sync per verify: fold the argmax into the same evaluation as the hidden states,
            // so the full [1, K, vocab] logits are reduced inside the graph and the captured
            // per-step states stay lazy — only the SELECTED rollback state is materialized below
            // (on full acceptance the captures are discarded without ever being computed).
            let targetPredsArr = argMax(verifyLogits, axis: -1).asType(.int32)
            eval(verifyHidden, targetPredsArr)
            let targetPreds = targetPredsArr.asArray(Int32.self).map { Int($0) }

            let budget = maxTokens - output.count
            let (acc, newToksRaw) = speculativeWalk(
                draftTokens: draftTokens, targetTokens: targetPreds, budget: budget)
            accepted += acc
            acceptLens.append(acc)
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
            // Stream this round's committed tokens (accepted drafts + the correction).
            if !newToks.isEmpty, onTokens?(newToks) == false { cancelled = true }
            if hitEOS {
                // The verify block was cut short of the rollback bookkeeping — the caches no
                // longer match a clean token prefix, so this session can't be resumed. (EOS
                // ends the turn's final step; the next turn re-prefills anyway.)
                session?.invalidated = true
                break
            }

            // Roll back the target cache to EXACTLY [bonus] + accepted drafts — the correction
            // is the NEXT bonus and stays out of the cache. On full acceptance the cache already
            // holds [bonus, all drafts]; otherwise, with NO re-forward: full-attention caches
            // `trim` to the accepted offset, gated-delta caches restore the captured per-step
            // conv + SSM state at index `acc` (the state after [bonus]+accepted drafts).
            if acc < draftTokens.count {
                let keepOffset = preVerifyOffset + acc + 1
                let drop = draftTokens.count - acc  // rejected drafts to drop from the verify block
                var selectedStates: [MLXArray] = []
                for c in targetCache {
                    if let mamba = c as? MambaCache {
                        mamba[0] = mamba.capturedConv[acc]
                        mamba[1] = mamba.capturedSSM[acc]
                        (mamba as BaseKVCache).offset = keepOffset
                        selectedStates.append(mamba.capturedConv[acc])
                        selectedStates.append(mamba.capturedSSM[acc])
                    } else if let base = c as? BaseKVCache, base.isTrimmable {
                        _ = base.trim(drop)
                    }
                }
                // Materialize ONLY the selected rollback state (index `acc`) — one state per
                // gated-delta layer instead of all K — cutting it loose from the verify graph
                // before the captures are cleared. States past `acc` are never computed.
                if !selectedStates.isEmpty { eval(selectedStates) }
            }
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
            // the session (the round's correction becomes the next bonus and stays out).
            session?.tokens += [bonus] + draftTokens.prefix(acc)

            bonus = newToks.last ?? bonus
        }

        return Qwen35MTPResult(
            tokens: Array(output.prefix(maxTokens)), proposed: proposed, accepted: accepted,
            targetForwards: targetForwards, rounds: rounds,
            resumedSuffix: resuming ? suffixTokens.count : -1)
    }
}
