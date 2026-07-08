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

    // MARK: - Adaptive block size (marginal-depth staircase controller)

    @Test func adaptiveStaysAtBaseWithShortHistory() {
        // <8 rounds of history → don't grow yet, stay at the configured base.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: [2, 2, 2], proposedLens: [2, 2, 2], remainingBudget: 100)
        #expect(bs == 3)
    }

    @Test func adaptiveGrowsOneStepNotToCeiling() {
        // Sustained full acceptance at the base block (2/2 drafts) → grow ONE step (4),
        // never jump straight to the ceiling — deeper depths must earn their own hits.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: Array(repeating: 2, count: 12),
            proposedLens: Array(repeating: 2, count: 12), remainingBudget: 100)
        #expect(bs == 4)
    }

    @Test func adaptiveClimbsFromTheDepthLastRun() {
        // The staircase steps from the last round's depth: rounds at block 4 (3 drafts)
        // fully accepted → grow to 5.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: [2, 2, 2, 2] + Array(repeating: 3, count: 8),
            proposedLens: [2, 2, 2, 2] + Array(repeating: 3, count: 8), remainingBudget: 100)
        #expect(bs == 5)
    }

    @Test func adaptiveCapsAtCeiling() {
        // Already at the ceiling with perfect acceptance → stay there.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: Array(repeating: 5, count: 12),
            proposedLens: Array(repeating: 5, count: 12), remainingBudget: 100)
        #expect(bs == 6)
    }

    @Test func adaptiveHoldsInHysteresisBand() {
        // Full-block hit rate 3/12 = 0.25 ∈ [0.20, 0.30) at block 4 → hold the depth
        // (marginal cost ≈ marginal yield; no thrash).
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: [3, 3, 3] + Array(repeating: 1, count: 9),
            proposedLens: Array(repeating: 3, count: 12), remainingBudget: 100)
        #expect(bs == 4)
    }

    @Test func adaptiveShrinksOnLowHitRate() {
        // Hit rate < 20% at block 4 → the deepest draft is wasted compute; step back to 3.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: Array(repeating: 0, count: 12),
            proposedLens: Array(repeating: 3, count: 12), remainingBudget: 100)
        #expect(bs == 3)
    }

    @Test func adaptiveNeverShrinksBelowConfigured() {
        // Total rejection at the base block → still no lower than the configured depth.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: Array(repeating: 0, count: 12),
            proposedLens: Array(repeating: 2, count: 12), remainingBudget: 100)
        #expect(bs == 3)
    }

    @Test func adaptiveRespectsRemainingBudget() {
        // Budget caps the block even when acceptance is high.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: Array(repeating: 2, count: 12),
            proposedLens: Array(repeating: 2, count: 12), remainingBudget: 4)
        #expect(bs == 4)
    }

    @Test func adaptiveIgnoresZeroDraftRounds() {
        // Rounds that proposed no drafts carry no acceptance signal; if fewer than 6
        // informative rounds remain in the window, stay at the base.
        let bs = Qwen35MTP.effectiveBlockSize(
            requestedBlock: 6, configuredBlock: 3,
            acceptLens: Array(repeating: 0, count: 8) + [2, 2],
            proposedLens: Array(repeating: 0, count: 8) + [2, 2], remainingBudget: 100)
        #expect(bs == 3)
    }

    @Test func fixedBlockNoRoomToGrow() {
        // requested == configured (a fixed size) → just return it (capped by budget).
        #expect(Qwen35MTP.effectiveBlockSize(
            requestedBlock: 3, configuredBlock: 3,
            acceptLens: [], proposedLens: [], remainingBudget: 100) == 3)
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

/// The pure sampled speculative-decoding walk — deterministic drafts verified by the
/// rejection rule (accept draft i with probability p_target(draft_i); on rejection the
/// correction is the pre-drawn residual sample; on full acceptance the final sample).
@Suite
struct Qwen35MTPSampledWalkTests {

    @Test func allAcceptedTakesFinalSample() {
        // u_i < p_i everywhere → all drafts accepted + the final-row sample.
        let r = Qwen35MTP.speculativeWalkSampled(
            draftTokens: [5, 6], pDraft: [0.9, 0.8], uniforms: [0.5, 0.5],
            residualSamples: [11, 12], finalSample: 7, budget: 10)
        #expect(r.accepted == 2)
        #expect(r.newTokens == [5, 6, 7])
    }

    @Test func rejectionTakesResidualAtThatRow() {
        // Row 0 accepts (0.3 < 0.9); row 1 rejects (0.9 >= 0.2) → residual sample of row 1.
        let r = Qwen35MTP.speculativeWalkSampled(
            draftTokens: [5, 6], pDraft: [0.9, 0.2], uniforms: [0.3, 0.9],
            residualSamples: [11, 12], finalSample: 7, budget: 10)
        #expect(r.accepted == 1)
        #expect(r.newTokens == [5, 12])
    }

    @Test func firstRowRejection() {
        let r = Qwen35MTP.speculativeWalkSampled(
            draftTokens: [5, 6], pDraft: [0.1, 0.9], uniforms: [0.5, 0.1],
            residualSamples: [11, 12], finalSample: 7, budget: 10)
        #expect(r.accepted == 0)
        #expect(r.newTokens == [11])
    }

    @Test func budgetTruncates() {
        let r = Qwen35MTP.speculativeWalkSampled(
            draftTokens: [5, 6], pDraft: [0.9, 0.8], uniforms: [0.1, 0.1],
            residualSamples: [11, 12], finalSample: 7, budget: 2)
        #expect(r.accepted == 2)
        #expect(r.newTokens == [5, 6])
    }

    @Test func emptyDraftEmitsFinalSample() {
        // blockSize 1: no drafts — the walk reduces to plain sampling of one token.
        let r = Qwen35MTP.speculativeWalkSampled(
            draftTokens: [], pDraft: [], uniforms: [],
            residualSamples: [], finalSample: 42, budget: 10)
        #expect(r.accepted == 0)
        #expect(r.newTokens == [42])
    }

    @Test func boundaryEqualUniformRejects() {
        // u == p rejects (accept iff u < p) — pins the comparison direction.
        let r = Qwen35MTP.speculativeWalkSampled(
            draftTokens: [5], pDraft: [0.5], uniforms: [0.5],
            residualSamples: [11], finalSample: 7, budget: 10)
        #expect(r.accepted == 0)
        #expect(r.newTokens == [11])
    }
}
