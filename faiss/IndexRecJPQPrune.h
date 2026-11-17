/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/invlists/RecJPQInvertedLists.h>

namespace faiss {

/// Statistics collected during search(), for the caller to inspect.
/// When several queries are searched in one call, the counters accumulate
/// over the queries of the call.
struct RecJPQPruneStats {
    /// number of main-loop iterations executed
    size_t iterations = 0;
    /// number of unique items whose score was computed
    size_t items_scored = 0;
    /// number of posting-list entries visited (before deduplication)
    size_t postings_traversed = 0;
    /// number of sub-item ids processed
    size_t sub_items_processed = 0;
    /// final value of the sigma upper bound (last query of the call)
    float final_sigma = 0;
    /// final value of the theta threshold (last query of the call)
    float final_theta = 0;
};

/** Search parameters for IndexRecJPQPrune.
 *
 * Exactness guarantees (see IndexRecJPQPrune::search):
 *  - exact_mode = true (default): the search runs until the exact pruning
 *    condition sigma <= theta holds, so the result is guaranteed to be the
 *    exact top-K under the documented tie-breaking policy. ub_inflation
 *    must be 1.0 and max_iterations is ignored (unlimited).
 *  - exact_mode = false: the search may stop early as soon as
 *    sigma <= theta * ub_inflation (ub_inflation >= 1.0), or after
 *    max_iterations main-loop iterations (>= 0). The result is then only a
 *    best-effort top-K over the items scored so far; check
 *    IndexRecJPQPrune::last_search_complete after the call.
 */
struct SearchParametersRecJPQPrune : SearchParameters {
    /// number of sub-item ids processed per main-loop iteration (BS), >= 1
    size_t batch_size = 8;
    /// safety limit on the main-loop iterations; < 0 means unlimited.
    /// Ignored in exact mode.
    int64_t max_iterations = -1;
    /// multiplier applied to the pruning threshold in approximate mode:
    /// the search stops as soon as sigma <= theta * ub_inflation.
    /// Must be >= 1.0; 1.0 (default) keeps the search exact.
    float ub_inflation = 1.0f;
    /// when true (default), the search runs to exact termination
    bool exact_mode = true;
    /// if non-null, filled with the statistics of the search
    RecJPQPruneStats* stats = nullptr;
};

/** Index of RecJPQ-compressed item embeddings, searched with the
 * RecJPQPrune dynamic pruning algorithm.
 *
 * Reference: "Efficient Recommendation with Millions of Items by Dynamic
 * Pruning of Sub-Item Embeddings", Petrov, Macdonald and Tonellotto,
 * SIGIR 2025 (https://arxiv.org/abs/2505.00560).
 *
 * Each of the N items is represented by M codes in [0, B) (one per split);
 * the item embedding is the concatenation of the M sub-item embeddings
 * looked up from the codebooks G[m][b] in R^d_split, with d_split = d / M.
 * The score of an item for a query phi is the sum over the splits of the
 * dot product of the sub-item embedding with the corresponding sub-query:
 *
 *     score(i) = sum_m S[m][item_codes[i][m]]
 *     S[m][b]  = dot(G[m][b], phi_m)
 *
 * This is the PQTopK scoring function. The search implements Algorithm 1 of
 * the paper: sub-item ids are processed per split in descending order of
 * their S score; at every iteration the split whose next unprocessed
 * sub-item has the highest score contributes a batch of batch_size
 * sub-item ids, whose posting lists yield the candidate items; the new
 * candidates are scored with the S lookup table and merged into a running
 * top-K. The search terminates when the upper bound on the score of any
 * unscored item
 *
 *     sigma = sum_m S[m][Q[m][P[m]]]   (-inf for exhausted splits)
 *
 * drops to the score theta of the K-th best item found so far; under this
 * condition no unscored item can reach the top-K.
 *
 * The index is built from pre-computed codes and codebooks (typically from
 * an existing IndexPQ), not from dense vectors: see build(). add() and
 * add_with_ids() therefore throw; add_sa_codes() accepts RecJPQ codes.
 *
 * Results are returned in descending score order. Ties in score are broken
 * deterministically by discovery order: among items with an equal score,
 * the item that first entered a candidate batch (earlier iteration, then
 * earlier position within the batch) ranks first. This policy makes the
 * pruning condition sigma <= theta exact, since an unscored item tying the
 * threshold necessarily loses the tie-break against an already scored one.
 *
 * Only METRIC_INNER_PRODUCT is supported: higher scores are better.
 *
 * Thread safety: const methods (search, reconstruct, ...) may be called
 * concurrently from several threads; each search thread uses its own
 * scratch buffers, including the deduplication state. Non-const methods
 * (build, reset) must not run concurrently with anything else.
 *
 * Differences from the paper and its TF reference
 * (pruned_score_batch.py):
 *  - the reference compares sigmoid(sigma) > sigmoid(theta) * ub_inflation;
 *    here the same stopping rule is applied directly as
 *    sigma <= theta * ub_inflation (identical at ub_inflation = 1.0);
 *  - the reference scores only the top of each posting batch; here all new
 *    candidates of a batch are scored. Both are exact: every candidate is
 *    scored at most once per query (deduplicated), so the total work is the
 *    same;
 *  - the reference stores a dense, -1-padded inverted index; here the CSR
 *    lists (RecJPQInvertedLists) store no padding;
 *  - B > 256 is not supported: item codes are stored as uint8_t.
 *
 * See also: benchs/bench_recjpq_prune.cpp (benchmark against an exhaustive
 * PQTopK baseline) and demos/demo_recjpq_prune.cpp (minimal usage).
 */
struct IndexRecJPQPrune : Index {
    /// number of splits
    size_t M = 0;
    /// number of sub-item ids per split, in [1, 256]
    size_t B = 0;
    /// dimensionality of one split (d / M)
    size_t d_split = 0;

    /** row-major N * M item codes; the code at split m of item i is at
     * item_codes[i * M + m] */
    std::vector<uint8_t> item_codes;
    /** external ids, one per item (size ntotal) */
    std::vector<idx_t> ids;
    /** codebooks: G[m][b][j] at codebooks[(m * B + b) * d_split + j] */
    std::vector<float> codebooks;

    /** CSR inverted lists over the item codes */
    RecJPQInvertedLists invlists;

    /// default search parameters, used when search() gets no params
    size_t batch_size = 8;
    int64_t max_iterations = -1;
    float ub_inflation = 1.0f;
    bool exact_mode = true;
    /// true when ids[i] == i for all i (cache maintained by build(),
    /// add_sa_codes() and reset(); used to shortcut id lookups)
    bool ids_sequential = true;

    /// true after the most recent search() call ran every query to exact
    /// termination; false when an approximate stop (ub_inflation > 1.0 or
    /// the max_iterations limit) triggered first
    mutable std::atomic<bool> last_search_complete{true};

    /** Construct an empty index.
     *
     * @param d      embedding dimension, a positive multiple of M
     * @param M      number of splits, >= 1
     * @param B      number of sub-item ids per split, in [1, 256]
     * @param metric must be METRIC_INNER_PRODUCT
     */
    explicit IndexRecJPQPrune(
            size_t d,
            size_t M,
            size_t B,
            MetricType metric = METRIC_INNER_PRODUCT);

    /** Default constructor, only used by read_index() before the header
     * is parsed; do not use directly. */
    IndexRecJPQPrune() : Index(0, METRIC_INNER_PRODUCT) {}

    /** Populate the index from RecJPQ codes and codebooks.
     *
     * The index must be empty (call reset() first after a previous build).
     * The supplied arrays are copied; the caller keeps ownership of them.
     *
     * @param n          number of items
     * @param item_codes N*M uint8 codes, the code at split m of item i at
     *                   item_codes[i * M + m], each in [0, B)
     * @param codebooks  M*B*d_split floats, G[m][b][j] at
     *                   codebooks[(m * B + b) * d_split + j]
     * @param ids        optional external ids, size n; when null the
     *                   sequential ids 0..n-1 are used. Duplicate ids are
     *                   allowed.
     */
    void build(
            size_t n,
            const uint8_t* item_codes,
            const float* codebooks,
            const idx_t* ids = nullptr);

    /** Compute the M x B sub-item score matrix (PQTopK lookup table) for
     * the query x, row-major: S[m][b] = dot(G[m][b], x[m*d_split..]).
     * @param S  output, size M * B, caller-allocated
     */
    void compute_sub_item_scores(const float* x, float* S) const;

    /** Score a single item with the precomputed sub-item score matrix:
     * score(i) = sum_m S[m][item_codes[i * M + m]].
     * @param S  output of compute_sub_item_scores
     */
    float score_item(size_t i, const float* S) const;

    /** @param x  dense query vectors, not supported: the index stores
     *            pre-computed RecJPQ codes, not dense vectors
     */
    void add(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    /** Add items from RecJPQ codes (standalone codec interface).
     *
     * The codebooks must already be present (they are set by build()); the
     * item codes are appended to the index and the inverted lists are
     * rebuilt over all items, so repeated calls are O(N) each.
     *
     * @param codes  n*M uint8 codes, as for build()
     * @param xids   optional external ids, size n; when null the sequential
     *               ids ntotal .. ntotal + n - 1 are used
     */
    void add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids)
            override;

    /** size of the standalone codec: M bytes per item (one uint8 code per
     * split) */
    size_t sa_code_size() const override;

    /** Decode RecJPQ codes back into (approximate) full embeddings.
     * @param bytes  n*M uint8 codes
     * @param x      output, size n * d
     */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// drops all items; the codebooks and dimensions are kept
    void reset() override;

    /** Reconstruct the (approximate) embedding of the item with external
     * id key. */
    void reconstruct(idx_t key, float* recons) const override;
    void reconstruct_batch(
            idx_t n,
            const idx_t* keys,
            float* recons) const override;
    /** Reconstruct the items with external ids i0 .. i0 + ni - 1
     * (Faiss convention for external ids). */
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    /** Reconstruct the items at internal positions i0 .. i0 + ni - 1. */
    void reconstruct_positions(idx_t i0, idx_t ni, float* recons) const;

    /// size in bytes of the index data (codes, ids, codebooks, lists)
    size_t memory_footprint() const;

private:
    /// per-thread scratch used by search(); never shared between threads
    struct ThreadState {
        /// sub-item score matrix, M * B
        std::vector<float> S;
        /// sorted sub-item ids per split (descending score), M * B
        std::vector<uint8_t> Q;
        /// current position in Q per split, M
        std::vector<size_t> P;
        /// scratch for the per-split sub-item id sort
        std::vector<size_t> order;

        struct Entry {
            float score;
            idx_t pos; ///< internal item position
            uint32_t seq; ///< discovery sequence (tie-break)
        };
        /// running top-K result set, sorted by (score desc, seq asc)
        std::vector<Entry> topk;
        /// merge buffer: topk union current candidate batch
        std::vector<Entry> merge;
        /// candidate batch of the current iteration (internal positions)
        std::vector<idx_t> cand_pos;

        /// deduplication: visited[pos] is the generation of the last query
        /// that scored the item at internal position pos
        std::vector<uint32_t> visited;
        uint32_t generation = 0;
    };

    /// compare two top-K entries: higher score first, then earlier discovery
    /// sequence first; the sequence is unique per item within a query, so
    /// this is a total order
    static bool entry_before(
            const ThreadState::Entry& a,
            const ThreadState::Entry& b);

    /// scratch of the calling thread, created on first use; the shared_ptr
    /// keeps the state alive even if the cache evicts it from another thread
    std::shared_ptr<ThreadState> get_thread_state() const;

    /// run the RecJPQPrune algorithm for one query; returns true when the
    /// search ran to exact termination
    bool search_one(
            const float* xq,
            idx_t k,
            float* distances,
            idx_t* labels,
            size_t batch_size,
            int64_t max_iterations,
            float ub_inflation,
            bool exact_mode,
            RecJPQPruneStats* stats,
            ThreadState& ts) const;

    /// external id -> internal position; throws when the id is unknown
    idx_t position_of_id(idx_t key) const;
    /// recompute ids_sequential for the current ids
    void recompute_ids_sequential();

    mutable std::mutex thread_state_mutex;
    mutable std::unordered_map<
            std::thread::id,
            std::shared_ptr<ThreadState>>
            thread_states;

    /// lazily built external id -> internal position map (reconstruct)
    mutable std::mutex id_map_mutex;
    mutable std::unordered_map<idx_t, idx_t> id_to_pos;
    mutable bool id_map_built = false;
};

} // namespace faiss
