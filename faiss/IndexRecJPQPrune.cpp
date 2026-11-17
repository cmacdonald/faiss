/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRecJPQPrune.h>

#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef FINTEGER
#define FINTEGER int
#endif

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

} // namespace

bool IndexRecJPQPrune::entry_before(
        const ThreadState::Entry& a,
        const ThreadState::Entry& b) {
    // higher score first, then earlier discovery sequence first; the
    // sequence is unique per item within a query, so this is a total order
    return a.score > b.score || (a.score == b.score && a.seq < b.seq);
}

IndexRecJPQPrune::IndexRecJPQPrune(
        size_t d,
        size_t M,
        size_t B,
        MetricType metric)
        : Index(d, metric), M(M), B(B) {
    FAISS_THROW_IF_NOT_MSG(d > 0, "IndexRecJPQPrune: d must be > 0");
    FAISS_THROW_IF_NOT_MSG(M > 0, "IndexRecJPQPrune: M must be > 0");
    FAISS_THROW_IF_NOT_MSG(B >= 1, "IndexRecJPQPrune: B must be >= 1");
    FAISS_THROW_IF_NOT_MSG(
            B <= 256,
            "IndexRecJPQPrune: B must be <= 256 (item codes are uint8)");
    FAISS_THROW_IF_NOT_MSG(
            d % M == 0, "IndexRecJPQPrune: d must be a multiple of M");
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_INNER_PRODUCT,
            "IndexRecJPQPrune: only METRIC_INNER_PRODUCT is supported");
    d_split = d / M;
}

void IndexRecJPQPrune::recompute_ids_sequential() {
    ids_sequential = true;
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] != (idx_t)i) {
            ids_sequential = false;
            return;
        }
    }
}

void IndexRecJPQPrune::build(
        size_t n,
        const uint8_t* item_codes,
        const float* codebooks,
        const idx_t* ids) {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0,
            "IndexRecJPQPrune::build: index is not empty; call reset() first");
    FAISS_THROW_IF_NOT_MSG(
            n == 0 || item_codes != nullptr,
            "IndexRecJPQPrune::build: item_codes must not be null when "
            "n > 0");
    FAISS_THROW_IF_NOT_MSG(
            codebooks != nullptr,
            "IndexRecJPQPrune::build: codebooks must not be null");

    if (n > 0) {
        this->item_codes.assign(item_codes, item_codes + n * M);
    } else {
        this->item_codes.clear();
    }
    this->codebooks.assign(codebooks, codebooks + M * B * d_split);
    if (ids != nullptr) {
        this->ids.assign(ids, ids + n);
    } else {
        this->ids.resize(n);
        std::iota(this->ids.begin(), this->ids.end(), (idx_t)0);
    }
    recompute_ids_sequential();

    invlists.build(this->item_codes.data(), n, M, B);
    ntotal = (idx_t)n;

    {
        std::lock_guard<std::mutex> lk(id_map_mutex);
        id_to_pos.clear();
        id_map_built = false;
    }
}

void IndexRecJPQPrune::add(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_MSG(
            "IndexRecJPQPrune does not support add(): it stores "
            "pre-computed RecJPQ codes, not dense vectors. Build the index "
            "from codes and codebooks with IndexRecJPQPrune::build(), for "
            "example from an existing IndexPQ: codes with pq.compute() and "
            "codebooks with pq.centroids. To append codes use "
            "add_sa_codes().");
}

void IndexRecJPQPrune::add_with_ids(
        idx_t /*n*/,
        const float* /*x*/,
        const idx_t* /*xids*/) {
    FAISS_THROW_MSG(
            "IndexRecJPQPrune does not support add_with_ids(): it stores "
            "pre-computed RecJPQ codes, not dense vectors. Build the index "
            "with IndexRecJPQPrune::build() or append codes with "
            "add_sa_codes().");
}

void IndexRecJPQPrune::add_sa_codes(
        idx_t n,
        const uint8_t* codes,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(codes != nullptr);
    FAISS_THROW_IF_NOT_MSG(
            !codebooks.empty(),
            "IndexRecJPQPrune::add_sa_codes: no codebooks; use build() to "
            "construct the index first");

    const size_t ntotal_old = (size_t)ntotal;
    item_codes.insert(item_codes.end(), codes, codes + (size_t)n * M);
    if (xids != nullptr) {
        ids.insert(ids.end(), xids, xids + n);
    } else {
        for (idx_t i = 0; i < n; i++) {
            ids.push_back((idx_t)ntotal_old + i);
        }
    }
    recompute_ids_sequential();

    // CSR construction requires all items; rebuild over the whole catalogue
    invlists.build(item_codes.data(), (size_t)ntotal + n, M, B);
    ntotal += n;

    {
        std::lock_guard<std::mutex> lk(id_map_mutex);
        id_to_pos.clear();
        id_map_built = false;
    }
}

size_t IndexRecJPQPrune::sa_code_size() const {
    return M;
}

void IndexRecJPQPrune::sa_decode(
        idx_t n,
        const uint8_t* bytes,
        float* x) const {
    FAISS_THROW_IF_NOT(bytes != nullptr);
    for (idx_t i = 0; i < n; i++) {
        float* r = x + (size_t)i * d;
        const uint8_t* c = bytes + (size_t)i * M;
        for (size_t m = 0; m < M; m++) {
            std::memcpy(
                    r + m * d_split,
                    codebooks.data() + ((size_t)m * B + c[m]) * d_split,
                    d_split * sizeof(float));
        }
    }
}

void IndexRecJPQPrune::compute_sub_item_scores(
        const float* x,
        float* S) const {
    FAISS_THROW_IF_NOT_MSG(
            codebooks.size() == M * B * d_split,
            "IndexRecJPQPrune: codebooks not set; call build() first");
    // S[m][b] = dot(G[m][b], x[m]): x_m (1 x d_split) times the codebook
    // block, whose row-major B x d_split storage is a column-major
    // d_split x B matrix, giving C as 1 x B written contiguously. A and C
    // are single rows, so their leading dimensions are 1.
    FINTEGER m_f = 1;
    FINTEGER n_f = (FINTEGER)B;
    FINTEGER k_f = (FINTEGER)d_split;
    FINTEGER lda = 1;
    FINTEGER ldb = (FINTEGER)d_split;
    FINTEGER ldc = 1;
    float one = 1.0f;
    float zero = 0.0f;
    for (size_t m = 0; m < M; m++) {
        sgemm_("Not transposed",
               "Not transposed",
               &m_f,
               &n_f,
               &k_f,
               &one,
               x + m * d_split,
               &lda,
               codebooks.data() + m * B * d_split,
               &ldb,
               &zero,
               S + m * B,
               &ldc);
    }
}

float IndexRecJPQPrune::score_item(size_t i, const float* S) const {
    FAISS_THROW_IF_NOT(i < (size_t)ntotal);
    float score = 0.f;
    const uint8_t* c = item_codes.data() + i * M;
    for (size_t m = 0; m < M; m++) {
        score += S[m * B + c[m]];
    }
    return score;
}

std::shared_ptr<IndexRecJPQPrune::ThreadState>
IndexRecJPQPrune::get_thread_state() const {
    std::lock_guard<std::mutex> lk(thread_state_mutex);
    const std::thread::id tid = std::this_thread::get_id();
    auto it = thread_states.find(tid);
    if (it != thread_states.end()) {
        return it->second;
    }
    // bound the cache: long-lived thread pools keep their states, stray
    // short-lived threads are evicted (their state is re-created on demand)
    size_t cap = 8;
#ifdef _OPENMP
    cap = std::max<size_t>(8, 2 * (size_t)omp_get_max_threads());
#endif
    while (thread_states.size() >= cap) {
        thread_states.erase(thread_states.begin());
    }
    auto state = std::make_shared<ThreadState>();
    thread_states.emplace(tid, state);
    return state;
}

bool IndexRecJPQPrune::search_one(
        const float* xq,
        idx_t k,
        float* distances,
        idx_t* labels,
        size_t batch_size,
        int64_t max_iterations,
        float ub_inflation,
        bool exact_mode,
        RecJPQPruneStats* stats,
        ThreadState& ts) const {
    // pad the outputs (missing results are -inf / -1)
    std::fill(labels, labels + k, -1);
    std::fill(distances, distances + k, kNegInf);
    if (k <= 0 || ntotal == 0) {
        return true;
    }

    // ---- 1. sub-item score matrix (PQTopK lookup table), once per query
    const size_t nb = M * B;
    // the per-query state is re-created each query (topk entries and their
    // discovery sequences are query-local)
    ts.topk.clear();
    if (ts.S.size() < nb) {
        ts.S.resize(nb);
    }
    compute_sub_item_scores(xq, ts.S.data());

    // ---- 2. per split, sort the sub-item ids by descending score
    // (ties: lower sub-item id first, for determinism)
    if (ts.Q.size() < nb) {
        ts.Q.resize(nb);
    }
    if (ts.P.size() < M) {
        ts.P.resize(M);
    }
    if (ts.order.size() < B) {
        ts.order.resize(B);
    }
    for (size_t m = 0; m < M; m++) {
        std::iota(ts.order.begin(), ts.order.begin() + B, (size_t)0);
        const float* Sm = ts.S.data() + m * B;
        std::sort(
                ts.order.begin(),
                ts.order.begin() + B,
                [Sm](size_t a, size_t b) {
                    return Sm[a] > Sm[b] || (Sm[a] == Sm[b] && a < b);
                });
        for (size_t j = 0; j < B; j++) {
            ts.Q[m * B + j] = (uint8_t)ts.order[j];
        }
        ts.P[m] = 0;
    }

    // ---- 3. initial upper bound: sigma = sum of the best unprocessed
    // sub-item score of every split
    float sigma = 0.f;
    for (size_t m = 0; m < M; m++) {
        sigma += ts.S[m * B + ts.Q[m * B]];
    }
    // threshold: score of the K-th best item so far (-inf while < K scored)
    float theta = kNegInf;

    // ---- 4. deduplication generation for this query
    if (ts.visited.size() < (size_t)ntotal) {
        ts.visited.resize((size_t)ntotal, 0);
    }
    if (ts.generation == std::numeric_limits<uint32_t>::max()) {
        // counter overflow: clear the table and restart at generation 1
        std::fill(ts.visited.begin(), ts.visited.end(), 0);
        ts.generation = 0;
    }
    const uint32_t gen = ++ts.generation;
    uint32_t seq = 0; ///< discovery sequence counter for this query

    // ---- main iteration (Algorithm 1 of Petrov et al., SIGIR 2025)
    int64_t iter = 0;
    size_t items_scored = 0;
    size_t postings_traversed = 0;
    size_t sub_items_processed = 0;
    bool capped = false;

    while (sigma > theta) {
        if (items_scored >= (size_t)ntotal) {
            break; // every item is scored: the top-K is final
        }
        if (max_iterations >= 0 && iter >= max_iterations) {
            capped = true;
            break;
        }

        // (1) split whose next unprocessed sub-item scores highest
        // (ties: lowest split first, for determinism)
        int mstar = -1;
        float best = kNegInf;
        for (size_t m = 0; m < M; m++) {
            if (ts.P[m] >= B) {
                continue; // split exhausted
            }
            const float s = ts.S[m * B + ts.Q[m * B + ts.P[m]]];
            if (s > best) {
                best = s;
                mstar = (int)m;
            }
        }
        if (mstar < 0) {
            break; // all splits exhausted (sigma is -inf then)
        }

        // (2) next batch of sub-item ids from that split, clamped at B
        const size_t j0 = ts.P[mstar];
        const size_t j1 = std::min(B, j0 + batch_size);

        // (3) candidates = union of the posting lists of the batch,
        // deduplicated by internal position
        ts.cand_pos.clear();
        for (size_t j = j0; j < j1; j++) {
            const size_t sub = ts.Q[mstar * B + j];
            const auto range = invlists.get_list_range((size_t)mstar, sub);
            const idx_t* begin = range.first;
            const idx_t* end = range.second;
            for (const idx_t* it = begin; it != end; ++it) {
                const idx_t p = *it;
                if (ts.visited[p] != gen) {
                    ts.visited[p] = gen;
                    ts.cand_pos.push_back(p);
                }
            }
            postings_traversed += (size_t)(end - begin);
        }
        sub_items_processed += j1 - j0;

        // (4) score the new candidates with the S lookup table (PQTopK)
        const size_t nnew = ts.cand_pos.size();
        if (nnew > 0) {
            ts.merge.clear();
            ts.merge.reserve(ts.topk.size() + nnew);
            ts.merge.insert(
                    ts.merge.end(), ts.topk.begin(), ts.topk.end());
            for (size_t t = 0; t < nnew; t++) {
                const uint8_t* c = item_codes.data() + (size_t)ts.cand_pos[t] * M;
                float s = 0.f;
                for (size_t m = 0; m < M; m++) {
                    s += ts.S[m * B + c[m]];
                }
                ts.merge.push_back(ThreadState::Entry{s, ts.cand_pos[t], ++seq});
            }

            // (5) keep the top min(k, |union|) under (score desc, seq asc);
            // the first-seen-wins tie policy falls out of the unique seq
            const size_t keep =
                    std::min<size_t>((size_t)k, ts.merge.size());
            std::partial_sort(
                    ts.merge.begin(),
                    ts.merge.begin() + keep,
                    ts.merge.end(),
                    entry_before);
            ts.topk.assign(ts.merge.begin(), ts.merge.begin() + keep);

            // (6) threshold = score of the K-th best item (if K known)
            theta = (ts.topk.size() == (size_t)k)
                    ? ts.topk.back().score
                    : kNegInf;
            items_scored += nnew;
        }

        // (7) advance the pointer and recompute the upper bound
        ts.P[mstar] = j1;
        sigma = 0.f;
        for (size_t m = 0; m < M; m++) {
            if (ts.P[m] >= B) {
                sigma = kNegInf;
                break;
            }
            sigma += ts.S[m * B + ts.Q[m * B + ts.P[m]]];
        }

        iter++;
    }

    // ---- output, in descending score order (topk is already sorted)
    const size_t out = std::min<size_t>((size_t)k, ts.topk.size());
    for (size_t r = 0; r < out; r++) {
        distances[r] = ts.topk[r].score;
        labels[r] = ids[ts.topk[r].pos];
    }

    if (stats != nullptr) {
        stats->iterations += (size_t)iter;
        stats->items_scored += items_scored;
        stats->postings_traversed += postings_traversed;
        stats->sub_items_processed += sub_items_processed;
        stats->final_sigma = sigma;
        stats->final_theta = theta;
    }

    return !capped;
}

void IndexRecJPQPrune::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k >= 0);
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_INNER_PRODUCT,
            "IndexRecJPQPrune: only METRIC_INNER_PRODUCT is supported");

    const SearchParametersRecJPQPrune* p =
            dynamic_cast<const SearchParametersRecJPQPrune*>(params);
    const size_t bs = p ? p->batch_size : batch_size;
    int64_t max_iter = p ? p->max_iterations : max_iterations;
    const float infl = p ? p->ub_inflation : ub_inflation;
    const bool exact = p ? p->exact_mode : exact_mode;
    RecJPQPruneStats* stats = p ? p->stats : nullptr;

    FAISS_THROW_IF_NOT_FMT(bs >= 1, "batch_size must be >= 1 (got %zu)", bs);
    FAISS_THROW_IF_NOT_FMT(
            infl >= 1.0f,
            "ub_inflation must be >= 1.0 (got %g)",
            (double)infl);
    if (exact) {
        FAISS_THROW_IF_NOT_FMT(
                infl == 1.0f,
                "exact_mode requires ub_inflation == 1.0 (got %g)",
                (double)infl);
        max_iter = -1; // unlimited in exact mode
    }

    auto ts = get_thread_state();

    bool complete = true;
    for (idx_t qi = 0; qi < n; qi++) {
        complete &= search_one(
                x + (size_t)qi * d,
                k,
                distances + (size_t)qi * k,
                labels + (size_t)qi * k,
                bs,
                max_iter,
                infl,
                exact,
                stats,
                *ts);
    }
    last_search_complete.store(complete);
}

void IndexRecJPQPrune::reset() {
    item_codes.clear();
    ids.clear();
    invlists.clear();
    ntotal = 0;
    ids_sequential = true;
    {
        std::lock_guard<std::mutex> lk(id_map_mutex);
        id_to_pos.clear();
        id_map_built = false;
    }
    // the codebooks are kept: they are part of the index configuration
}

idx_t IndexRecJPQPrune::position_of_id(idx_t key) const {
    if (ids_sequential) {
        FAISS_THROW_IF_NOT_FMT(
                key >= 0 && key < ntotal,
                "unknown id %" PRId64,
                key);
        return key;
    }
    std::lock_guard<std::mutex> lk(id_map_mutex);
    if (!id_map_built) {
        id_to_pos.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); i++) {
            id_to_pos[ids[i]] = (idx_t)i;
        }
        id_map_built = true;
    }
    auto it = id_to_pos.find(key);
    FAISS_THROW_IF_NOT_FMT(
            it != id_to_pos.end(), "unknown id %" PRId64, key);
    return it->second;
}

void IndexRecJPQPrune::reconstruct_positions(
        idx_t i0,
        idx_t ni,
        float* recons) const {
    FAISS_THROW_IF_NOT_MSG(
            i0 >= 0 && i0 + ni <= ntotal,
            "IndexRecJPQPrune::reconstruct_positions: range out of bounds");
    for (idx_t i = 0; i < ni; i++) {
        float* r = recons + (size_t)i * d;
        const uint8_t* c = item_codes.data() + ((size_t)i0 + i) * M;
        for (size_t m = 0; m < M; m++) {
            std::memcpy(
                    r + m * d_split,
                    codebooks.data() + ((size_t)m * B + c[m]) * d_split,
                    d_split * sizeof(float));
        }
    }
}

void IndexRecJPQPrune::reconstruct(idx_t key, float* recons) const {
    const idx_t pos = position_of_id(key);
    reconstruct_positions(pos, 1, recons);
}

void IndexRecJPQPrune::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    // Faiss convention: i0 .. i0 + ni - 1 are external ids
    std::vector<idx_t> pos(ni);
    for (idx_t i = 0; i < ni; i++) {
        pos[i] = position_of_id(i0 + i);
    }
    for (idx_t i = 0; i < ni; i++) {
        reconstruct_positions(pos[i], 1, recons + (size_t)i * d);
    }
}

void IndexRecJPQPrune::reconstruct_batch(
        idx_t n,
        const idx_t* keys,
        float* recons) const {
    for (idx_t i = 0; i < n; i++) {
        reconstruct(keys[i], recons + (size_t)i * d);
    }
}

size_t IndexRecJPQPrune::memory_footprint() const {
    return item_codes.size() * sizeof(uint8_t) +
            ids.size() * sizeof(idx_t) + codebooks.size() * sizeof(float) +
            invlists.offsets.size() * sizeof(size_t) +
            invlists.item_ids.size() * sizeof(idx_t);
}

} // namespace faiss
