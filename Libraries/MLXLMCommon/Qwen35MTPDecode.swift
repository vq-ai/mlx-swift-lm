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

    /// Adaptive MTP block size — ported from mlx-vlm `_effective_mtp_block_size`
    /// (speculative/mtp.py). Grows from the drafter's configured (trained) depth toward
    /// `requestedBlock` only when the base depth's drafts are fully accepted often enough
    /// (≥65% of recent rounds); otherwise the extra autoregressive draft steps just pay for
    /// tokens that won't be accepted, so it stays at the base. `acceptLens` is the per-round
    /// accepted-draft history; `remainingBudget` caps the block to the tokens left to emit.
    public static func effectiveBlockSize(
        requestedBlock: Int, configuredBlock: Int, acceptLens: [Int], remainingBudget: Int
    ) -> Int {
        let blockTotal = min(requestedBlock, remainingBudget)
        let configured = min(configuredBlock, blockTotal)
        if blockTotal <= configured || configured <= 1 { return blockTotal }
        if acceptLens.count < 8 { return configured }
        let recent = acceptLens.suffix(12)  // short window: react fast at regime changes (tool-call <-> prose)
        let configuredDraftCount = configured - 1
        let hits = recent.filter { $0 >= configuredDraftCount }.count
        let hitRate = Double(hits) / Double(recent.count)
        return hitRate < 0.65 ? configured : blockTotal
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
