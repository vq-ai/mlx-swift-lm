// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Stage 1: the pure greedy speculative-decoding walk — accept drafted tokens up to the
/// first mismatch with the target's greedy choice, then take the target's bonus there.
/// Ports `_speculative_walk` from mlx-vlm/speculative/common.py.
@Suite
struct Qwen35MTPDecodeTests {

    @Test func allAccepted() {
        // Both drafts match the target → accept all, plus the target's bonus token.
        let r = Qwen35MTP.speculativeWalk(
            draftTokens: [5, 6], targetTokens: [5, 6, 7], budget: 10)
        #expect(r.accepted == 2)
        #expect(r.newTokens == [5, 6, 7])
    }

    @Test func partialAccept() {
        // draft[0]=5 matches, draft[1]=9 != target[1]=6 → accept 1, then target's 6.
        let r = Qwen35MTP.speculativeWalk(
            draftTokens: [5, 9], targetTokens: [5, 6, 7], budget: 10)
        #expect(r.accepted == 1)
        #expect(r.newTokens == [5, 6])
    }

    @Test func allRejected() {
        // First draft already mismatches → accept 0, emit only the target's first token.
        let r = Qwen35MTP.speculativeWalk(
            draftTokens: [9, 9], targetTokens: [5, 6, 7], budget: 10)
        #expect(r.accepted == 0)
        #expect(r.newTokens == [5])
    }

    @Test func emptyDraft() {
        // No drafts (e.g. blockSize 1) → just the target's single token.
        let r = Qwen35MTP.speculativeWalk(
            draftTokens: [], targetTokens: [7], budget: 10)
        #expect(r.accepted == 0)
        #expect(r.newTokens == [7])
    }

    @Test func budgetTruncates() {
        // All accepted would emit 3 tokens, but the budget caps it at 2.
        let r = Qwen35MTP.speculativeWalk(
            draftTokens: [5, 6], targetTokens: [5, 6, 7], budget: 2)
        #expect(r.accepted == 2)
        #expect(r.newTokens == [5, 6])
    }

    // MARK: - Adaptive block size (ported from _effective_mtp_block_size)

    @Test func adaptiveStaysAtBaseWithShortHistory() {
        // <8 rounds of history → don't grow yet, stay at the configured base.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3, acceptLens: [2, 2, 2], remainingBudget: 100)
        #expect(bs == 3)
    }

    @Test func adaptiveGrowsWhenBaseFullyAccepted() {
        // Recent rounds consistently accept the base draft-count (3-1=2) → grow to ceiling.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: Array(repeating: 2, count: 10), remainingBudget: 100)
        #expect(bs == 6)
    }

    @Test func adaptiveStaysAtBaseWhenAcceptanceLow() {
        // Base rarely fully accepted (<65%) → stay at base, don't waste draft steps.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: Array(repeating: 0, count: 10), remainingBudget: 100)
        #expect(bs == 3)
    }

    @Test func adaptiveHitRateBoundary() {
        // The controller looks at the last 12 rounds (short window: reacts fast at tool-call ↔
        // prose regime changes). 8/12 ≥ 65% → grow; 7/12 < 65% → stay. Older history (the
        // leading zeros) must be ignored.
        let grow = [Int](repeating: 0, count: 10) + [Int](repeating: 2, count: 8)
        #expect(Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3, acceptLens: grow, remainingBudget: 100) == 6)
        let stay = [Int](repeating: 0, count: 5) + [Int](repeating: 2, count: 7)
        #expect(Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3, acceptLens: stay, remainingBudget: 100) == 3)
    }

    @Test func adaptiveRespectsRemainingBudget() {
        // Budget caps the block even when acceptance is high.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: Array(repeating: 2, count: 10), remainingBudget: 4)
        #expect(bs == 4)
    }

    @Test func fixedBlockNoRoomToGrow() {
        // requested == configured (a fixed size) → just return it (capped by budget).
        #expect(Qwen35MTP.effectiveBlockSize(
            requestedBlock: 3, configuredBlock: 3, acceptLens: [], remainingBudget: 100) == 3)
    }

    // MARK: - Stage 2: cache snapshot / restore (exact rollback on rejection)

    @Test func kvCacheSnapshotRestoreRoundTrip() {
        // Full-attention KV cache writes in place, so the snapshot must survive later appends.
        let cache = KVCacheSimple()
        let k1 = MLXArray.ones([1, 2, 3, 4])
        let v1 = MLXArray.ones([1, 2, 3, 4]) * 2
        _ = cache.update(keys: k1, values: v1)
        #expect(cache.offset == 3)

        let snap = Qwen35MTP.snapshotCaches([cache])

        // A verify forward appends 2 more tokens.
        _ = cache.update(keys: MLXArray.ones([1, 2, 2, 4]) * 9, values: MLXArray.ones([1, 2, 2, 4]) * 9)
        #expect(cache.offset == 5)

        // Rollback must rewind to exactly the snapshot — offset and the first 3 tokens.
        Qwen35MTP.restoreCaches([cache], to: snap)
        #expect(cache.offset == 3)
        let s = cache.state
        #expect(s[0].dim(2) == 3)
        #expect((s[0] .== k1).all().item(Bool.self))
        #expect((s[1] .== v1).all().item(Bool.self))
    }

    @Test func mambaCacheSnapshotRestoreRoundTrip() {
        // Gated-delta recurrent state can't be trimmed — snapshot/restore is the only rollback.
        let cache = MambaCache()
        cache.state = [MLXArray.ones([1, 4, 4]), MLXArray.ones([1, 4, 4]) * 3]
        (cache as BaseKVCache).offset = 2

        let snap = Qwen35MTP.snapshotCaches([cache])

        // A verify forward overwrites the recurrent state and advances offset.
        cache.state = [MLXArray.zeros([1, 4, 4]), MLXArray.zeros([1, 4, 4])]
        (cache as BaseKVCache).offset = 7

        Qwen35MTP.restoreCaches([cache], to: snap)
        #expect(cache.offset == 2)
        let s = cache.state
        #expect((s[0] .== MLXArray.ones([1, 4, 4])).all().item(Bool.self))
        #expect((s[1] .== (MLXArray.ones([1, 4, 4]) * 3)).all().item(Bool.self))
    }
}
