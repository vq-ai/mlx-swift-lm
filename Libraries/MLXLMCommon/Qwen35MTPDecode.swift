// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Qwen3.5 MTP (multi-token-prediction) self-speculative decoding.
///
/// Unlike the Gemma4-style `MTPSpeculativeTokenIterator` (stateless drafter that
/// cross-attends the target's shared K/V), Qwen3.5 MTP uses a *stateful* drafter with its
/// own KV cache, prefilled from the target's hidden states. This namespace holds the
/// architecture-agnostic pieces (the greedy accept/reject walk, and later the decode loop).
public enum Qwen35MTP {

    /// Exact-greedy speculative-decoding walk.
    ///
    /// Accept drafted tokens up to the first mismatch with the target's greedy choice, then
    /// take the target's bonus token at that position. Ports `_speculative_walk` from
    /// mlx-vlm/speculative/common.py.
    ///
    /// - Parameters:
    ///   - draftTokens: the `K-1` tokens proposed by the drafter for this round.
    ///   - targetTokens: the target's greedy argmax at each of the `K` verify positions
    ///     (length `draftTokens.count + 1`); element `i` is the target's prediction given
    ///     the first `i` drafted tokens, so `targetTokens[accepted]` is the correction/bonus.
    ///   - budget: maximum number of new tokens to emit this round.
    /// - Returns: `accepted` (count of drafts that matched) and `newTokens` (the accepted
    ///   drafts followed by the target's bonus, truncated to `budget`).
    public static func speculativeWalk(
        draftTokens: [Int], targetTokens: [Int], budget: Int
    ) -> (accepted: Int, newTokens: [Int]) {
        var accepted = draftTokens.count
        for i in draftTokens.indices {
            if i >= targetTokens.count || draftTokens[i] != targetTokens[i] {
                accepted = i
                break
            }
        }
        var newTokens = Array(draftTokens.prefix(accepted))
        if accepted < targetTokens.count {
            newTokens.append(targetTokens[accepted])
        }
        return (accepted, Array(newTokens.prefix(max(0, budget))))
    }

    /// Sampling configuration for MTP speculative decoding.
    ///
    /// With sampling active, verification switches from exact-greedy token matching to the
    /// standard rejection rule for a *deterministic* draft distribution: draft token `d_i`
    /// is accepted with probability `p_target(d_i)` (after temperature / top-p / repetition
    /// transforms); on rejection the correction is sampled from the residual distribution
    /// (`p_target` with `d_i` zeroed, renormalized). The emitted stream is then an exact
    /// sample from the same distribution the standard (non-speculative) sampled decode
    /// produces — speculation changes speed, never the distribution.
    public struct Sampling: Sendable {
        /// Softmax temperature; must be > 0 (use `nil` Sampling for greedy).
        public var temperature: Float
        /// Nucleus threshold; 1 disables the top-p mask.
        public var topP: Float
        /// CTRL-style repetition penalty (>1 penalizes); `nil` disables.
        public var repetitionPenalty: Float?
        /// Ring size for the repetition context (matches `RepetitionContext`).
        public var repetitionContextSize: Int
        /// Seed for reproducible generation (trajectory capture / benchmarks); `nil` = random.
        public var seed: UInt64?

        public init(
            temperature: Float, topP: Float = 1.0,
            repetitionPenalty: Float? = nil, repetitionContextSize: Int = 200,
            seed: UInt64? = nil
        ) {
            self.temperature = temperature
            self.topP = topP
            self.repetitionPenalty = repetitionPenalty
            self.repetitionContextSize = repetitionContextSize
            self.seed = seed
        }
    }

    /// Sampled speculative-decoding walk (deterministic drafts, exact target distribution).
    ///
    /// Pure function so the accept/reject math is unit-testable without models: the caller
    /// supplies per-position acceptance probabilities and pre-drawn samples.
    ///
    /// - Parameters:
    ///   - draftTokens: the `K-1` tokens proposed by the drafter for this round.
    ///   - pDraft: `p_target(d_i)` for each draft position (length `draftTokens.count`).
    ///   - uniforms: pre-drawn U(0,1) per draft position; `u_i < p_i` accepts `d_i`.
    ///   - residualSamples: per position, a sample from the residual distribution
    ///     (target with the draft token excluded) — used when position `i` rejects.
    ///   - finalSample: a sample from the last verify row's full distribution — the
    ///     bonus/correction when every draft is accepted.
    ///   - budget: maximum number of new tokens to emit this round.
    /// - Returns: `accepted` and `newTokens`, same contract as ``speculativeWalk``.
    public static func speculativeWalkSampled(
        draftTokens: [Int], pDraft: [Float], uniforms: [Float],
        residualSamples: [Int], finalSample: Int, budget: Int
    ) -> (accepted: Int, newTokens: [Int]) {
        precondition(pDraft.count == draftTokens.count && uniforms.count == draftTokens.count)
        precondition(residualSamples.count == draftTokens.count)
        for i in draftTokens.indices where uniforms[i] >= pDraft[i] {
            let newTokens = Array(draftTokens.prefix(i)) + [residualSamples[i]]
            return (i, Array(newTokens.prefix(max(0, budget))))
        }
        let newTokens = draftTokens + [finalSample]
        return (draftTokens.count, Array(newTokens.prefix(max(0, budget))))
    }

    /// Adaptive MTP block size — a marginal-depth staircase controller.
    ///
    /// Replaces the ported mlx-vlm `_effective_mtp_block_size` gate (≥65% full-base-block
    /// hits → jump straight to the ceiling), which never grew in practice: 65% full-block
    /// acceptance at depth 3 needs ~0.81 per-depth chain acceptance, far above the compute
    /// break-even. Drafting one token deeper costs ~0.17-0.2 of a target forward (the MTP
    /// head is small but shares the target's LM head), so the marginal depth pays for
    /// itself while the chance the whole current block is accepted stays ≥ ~0.25.
    ///
    /// Rule: over the recent rounds, measure the FULL-BLOCK hit rate (accepted == proposed
    /// drafts). ≥30% → grow ONE step (those rounds would each have had a shot at an extra
    /// token); <20% → shrink one step (the deepest draft is mostly wasted compute); the
    /// hysteresis band between avoids thrash and settles the depth where the marginal
    /// full-block rate straddles break-even. Tool-call segments (hit rates 0.89-1.0) climb
    /// to the ceiling in a few rounds; low-acceptance prose stays at the trained depth.
    ///
    /// `acceptLens`/`proposedLens` are the per-round accepted/proposed draft histories
    /// (parallel arrays, carried across steps via the session); `remainingBudget` caps the
    /// block to the tokens left to emit.
    public static func effectiveBlockSize(
        requestedBlock: Int, configuredBlock: Int,
        acceptLens: [Int], proposedLens: [Int], remainingBudget: Int
    ) -> Int {
        let blockTotal = min(requestedBlock, remainingBudget)
        let configured = min(configuredBlock, blockTotal)
        if blockTotal <= configured || configured <= 1 { return blockTotal }
        guard acceptLens.count >= 8, acceptLens.count == proposedLens.count else {
            return configured
        }
        // Short window: react fast at regime changes (tool-call <-> prose).
        let window = 12
        var hits = 0
        var informative = 0
        for (a, p) in zip(acceptLens.suffix(window), proposedLens.suffix(window)) where p > 0 {
            informative += 1
            if a >= p { hits += 1 }
        }
        guard informative >= 6 else { return configured }
        let hitRate = Double(hits) / Double(informative)
        // The staircase steps from the depth the last round actually ran (budget-clamped
        // end-of-generation rounds are pulled back into [configured, blockTotal]).
        let current = min(max((proposedLens.last ?? configured - 1) + 1, configured), blockTotal)
        if hitRate >= 0.30 { return min(current + 1, blockTotal) }
        if hitRate < 0.20 { return max(current - 1, configured) }
        return current
    }

    /// A full snapshot of the target's cache stack, taken before a verify forward so the
    /// cache can be restored exactly when speculation is (partly) rejected.
    ///
    /// This is the key to avoiding per-step gated-delta state surgery: rather than trimming
    /// the recurrent SSM state in place (which can't be sliced like attention KV), we keep a
    /// whole-cache snapshot. On full acceptance we discard it (no rollback — the fast path);
    /// on rejection we restore it and re-commit only the accepted tokens.
    public struct CacheSnapshot {
        let perCache: [(state: [MLXArray], offset: Int)]
    }

    /// Capture the state of every cache — full-attention KV *and* gated-delta recurrent
    /// state — plus its offset. The arrays are `eval`'d so a later in-place KV write (which
    /// `KVCacheSimple.update` does) can't mutate the captured data.
    public static func snapshotCaches(_ caches: [any KVCache]) -> CacheSnapshot {
        let per = caches.map { (state: $0.state, offset: $0.offset) }
        eval(per.flatMap { $0.state })
        return CacheSnapshot(perCache: per)
    }

    /// Restore every cache to a prior ``CacheSnapshot`` — both its arrays and its offset.
    /// (`KVCacheSimple.state`'s setter restores offset from the keys length, but
    /// `MambaCache`'s does not, so we set offset explicitly via `BaseKVCache`.)
    public static func restoreCaches(_ caches: [any KVCache], to snapshot: CacheSnapshot) {
        precondition(
            caches.count == snapshot.perCache.count,
            "restoreCaches: cache count \(caches.count) != snapshot \(snapshot.perCache.count)")
        for (cache, snap) in zip(caches, snapshot.perCache) {
            // All Qwen3.5 caches are BaseKVCache subclasses; the class cast makes the
            // settable `state`/`offset` reachable (the `any KVCache` existential is not
            // class-bound, so its setters aren't callable on a `let`). `state`'s setter is
            // dynamically dispatched to the concrete KVCacheSimple/MambaCache override.
            guard let base = cache as? BaseKVCache else { continue }
            base.state = snap.state
            base.offset = snap.offset
        }
    }
}
