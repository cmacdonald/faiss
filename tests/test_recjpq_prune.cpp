/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRecJPQPrune.h>
#include <faiss/clone_index.h>
#include <faiss/impl/FaissException.h>
#include <faiss/index_io.h>
#include <faiss/invlists/RecJPQInvertedLists.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

using faiss::idx_t;
using faiss::IndexRecJPQPrune;
using faiss::MetricType;
using faiss::RecJPQPruneStats;
using faiss::RecJPQInvertedLists;
using faiss::SearchParametersRecJPQPrune;

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

/// relative/absolute float tolerance for score comparisons
bool score_eq(float a, float b) {
    const float tol = 1e-4f * std::max({1.0f, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) <= tol;
}

/// synthetic RecJPQ catalogue: random codes and random (Gaussian) codebooks
struct RecJPQData {
    size_t M = 0;
    size_t B = 0;
    size_t d_split = 0;
    size_t N = 0;
    std::vector<uint8_t> codes; ///< N * M
    std::vector<float> codebooks; ///< M * B * d_split
    std::vector<idx_t> ids; ///< N
    std::unordered_map<idx_t, idx_t> id_to_pos; ///< (when ids are unique)

    size_t d() const {
        return M * d_split;
    }
};

RecJPQData make_recjpq(
        size_t M,
        size_t B,
        size_t d_split,
        size_t N,
        uint32_t seed,
        std::function<idx_t(size_t)> id_of = nullptr,
        float codebook_scale = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> subid(0, (int)B - 1);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    RecJPQData data;
    data.M = M;
    data.B = B;
    data.d_split = d_split;
    data.N = N;
    data.codes.resize(N * M);
    for (size_t i = 0; i < N; i++) {
        for (size_t m = 0; m < M; m++) {
            data.codes[i * M + m] = (uint8_t)subid(rng);
        }
    }
    data.codebooks.resize(M * B * d_split);
    for (auto& v : data.codebooks) {
        v = codebook_scale * gauss(rng);
    }
    data.ids.resize(N);
    for (size_t i = 0; i < N; i++) {
        data.ids[i] = id_of ? id_of(i) : (idx_t)i;
    }
    // id -> position map (only meaningful when ids are unique)
    bool unique = true;
    data.id_to_pos.reserve(N * 2);
    for (size_t i = 0; i < N; i++) {
        if (data.id_to_pos.count(data.ids[i]) > 0) {
            unique = false;
            break;
        }
        data.id_to_pos[data.ids[i]] = (idx_t)i;
    }
    if (!unique) {
        data.id_to_pos.clear();
    }
    return data;
}

std::unique_ptr<IndexRecJPQPrune> make_index(
        const RecJPQData& data,
        bool with_ids = true) {
    auto idx = std::make_unique<IndexRecJPQPrune>(
            data.d(), data.M, data.B);
    idx->build(
            data.N,
            data.codes.data(),
            data.codebooks.data(),
            with_ids ? data.ids.data() : nullptr);
    return idx;
}

std::vector<float> random_query(
        const RecJPQData& data,
        uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> q(data.d());
    for (auto& v : q) {
        v = gauss(rng);
    }
    // normalize
    float nrm = 0.f;
    for (float v : q) {
        nrm += v * v;
    }
    nrm = std::sqrt(nrm);
    for (auto& v : q) {
        v /= nrm;
    }
    return q;
}

/// Exhaustive PQTopK baseline: score every item with the sub-item score
/// matrix and select the exact top-K. Ties are broken by ascending
/// internal position (a reference ordering; the index breaks ties by
/// discovery order, which only differs when scores are exactly equal).
struct Exhaustive {
    std::vector<float> all_scores; ///< N
    std::vector<float> sorted_scores; ///< N, descending
    std::vector<idx_t> top_ids; ///< k, -1 padded
    std::vector<float> top_scores; ///< k, -inf padded
};

Exhaustive exhaustive_topk(
        const RecJPQData& data,
        const float* phi,
        idx_t k) {
    const size_t N = data.N;
    std::vector<float> S(data.M * data.B);
    for (size_t m = 0; m < data.M; m++) {
        for (size_t b = 0; b < data.B; b++) {
            float s = 0.f;
            const float* cb =
                    data.codebooks.data() + (m * data.B + b) * data.d_split;
            for (size_t j = 0; j < data.d_split; j++) {
                s += cb[j] * phi[m * data.d_split + j];
            }
            S[m * data.B + b] = s;
        }
    }
    std::vector<float> scores(N, 0.f);
    for (size_t i = 0; i < N; i++) {
        for (size_t m = 0; m < data.M; m++) {
            scores[i] += S[m * data.B + data.codes[i * data.M + m]];
        }
    }
    std::vector<size_t> order(N);
    std::iota(order.begin(), order.end(), (size_t)0);
    std::sort(order.begin(), order.end(), [&scores](size_t a, size_t b) {
        return scores[a] > scores[b] ||
                (scores[a] == scores[b] && a < b);
    });

    Exhaustive exp;
    exp.all_scores = scores;
    exp.sorted_scores.resize(N);
    for (size_t i = 0; i < N; i++) {
        exp.sorted_scores[i] = scores[order[i]];
    }
    const size_t k_eff = std::min<size_t>((size_t)k, N);
    exp.top_ids.resize(k, -1);
    exp.top_scores.resize(k, kNegInf);
    for (size_t r = 0; r < k_eff; r++) {
        exp.top_ids[r] = data.ids[order[r]];
        exp.top_scores[r] = scores[order[r]];
    }
    return exp;
}

/// Check a search result against the exhaustive baseline, tolerating the
/// documented tie-breaking policy: ranks and scores must always match, and
/// the ids must match exactly whenever no tie straddles the K-th boundary.
void check_results(
        const Exhaustive& exp,
        const RecJPQData& data,
        const float* D,
        const idx_t* I,
        idx_t k,
        std::string_view what) {
    const size_t N = data.N;
    for (idx_t r = 0; r < k; r++) {
        if (r >= N) {
            EXPECT_EQ(I[r], -1) << what;
            EXPECT_EQ(D[r], kNegInf) << what;
            continue;
        }
        EXPECT_TRUE(score_eq(D[r], exp.top_scores[r]))
                << what << ": rank " << r << " score " << D[r] << " != "
                << exp.top_scores[r];
        // a negative id is legitimate here (the catalogue may contain
        // negative external ids); membership in the catalogue is the only
        // validity criterion
        if (data.id_to_pos.count(I[r]) == 0) {
            ADD_FAILURE() << what << ": invalid id " << I[r];
            continue;
        }
        const idx_t pos = data.id_to_pos.at(I[r]);
        EXPECT_TRUE(score_eq(D[r], exp.all_scores[pos]))
                << what << ": id " << I[r] << " (pos " << pos
                << ") has score " << D[r] << " but exhaustive says "
                << exp.all_scores[pos];
    }
    // descending order
    for (idx_t r = 1; r < k; r++) {
        EXPECT_GE(D[r - 1], D[r] - 1e-6f) << what << " at rank " << r;
    }
    // exact id match when the K-th boundary has no ties
    if (k <= (idx_t)N && N > 0) {
        const float boundary = exp.top_scores[k - 1];
        size_t n_at_boundary = 0;
        for (size_t i = 0; i < N; i++) {
            if (score_eq(exp.all_scores[i], boundary)) {
                n_at_boundary++;
            }
        }
        size_t boundary_in_topk = 0;
        for (idx_t r = 0; r < k; r++) {
            if (score_eq(exp.top_scores[r], boundary)) {
                boundary_in_topk++;
            }
        }
        if (n_at_boundary == boundary_in_topk) {
            for (idx_t r = 0; r < k; r++) {
                EXPECT_EQ(I[r], exp.top_ids[r])
                        << what << ": rank " << r << " id " << I[r] << " != "
                        << exp.top_ids[r];
            }
        }
    }
}

/// run one search over the given query with the given params
void do_search(
        const IndexRecJPQPrune& idx,
        const float* q,
        idx_t k,
        SearchParametersRecJPQPrune* params,
        std::vector<float>& D,
        std::vector<idx_t>& I) {
    D.assign((size_t)k, 0.f);
    I.assign((size_t)k, -2);
    idx.search(
            1,
            q,
            k,
            D.data(),
            I.data(),
            params ? params : nullptr);
}

} // namespace

// ---------------------------------------------------------------- A:
// inverted index correctness

TEST(RecJPQInvertedLists, BuildCountsAndContents) {
    const size_t M = 4, B = 16, N = 500;
    auto data = make_recjpq(M, B, 8, N, 17);

    RecJPQInvertedLists lists;
    lists.build(data.codes.data(), N, M, B);

    ASSERT_EQ(lists.M, M);
    ASSERT_EQ(lists.B, B);
    ASSERT_EQ(lists.offsets.size(), M * B + 1);
    ASSERT_EQ(lists.item_ids.size(), N * M);
    EXPECT_EQ(lists.total_entries(), N * M);

    // every (split, sub_item_id) list must contain exactly the expected
    // items, in ascending position order
    size_t total = 0;
    for (size_t m = 0; m < M; m++) {
        for (size_t b = 0; b < B; b++) {
            std::vector<idx_t> expected;
            for (size_t i = 0; i < N; i++) {
                if (data.codes[i * M + m] == b) {
                    expected.push_back((idx_t)i);
                }
            }
            const size_t sz = lists.list_size(m, b);
            const auto range = lists.get_list_range(m, b);
            ASSERT_EQ(range.first != nullptr || sz == 0, true);
            EXPECT_EQ(sz, expected.size()) << "split " << m << " sub " << b;
            EXPECT_EQ(range.second - range.first, (ptrdiff_t)sz);
            EXPECT_EQ(lists.get_list(m, b), range.first);
            for (size_t t = 0; t < sz; t++) {
                EXPECT_EQ(range.first[t], expected[t]) << "split " << m
                        << " sub " << b << " slot " << t;
            }
            total += sz;
        }
    }
    EXPECT_EQ(total, N * M);
}

TEST(RecJPQInvertedLists, EmptyListsAndTinyCases) {
    // many empty posting lists
    const size_t M = 4, B = 64, N = 200;
    auto data = make_recjpq(M, B, 8, N, 19);
    RecJPQInvertedLists lists;
    lists.build(data.codes.data(), N, M, B);

    size_t n_empty = 0;
    for (size_t m = 0; m < M; m++) {
        for (size_t b = 0; b < B; b++) {
            if (lists.list_size(m, b) == 0) {
                n_empty++;
                const auto range = lists.get_list_range(m, b);
                EXPECT_EQ(range.first, range.second);
                EXPECT_EQ(lists.get_list(m, b), nullptr);
            }
        }
    }
    // with N*M = 800 postings over 256 lists, some lists must be empty
    EXPECT_GT(n_empty, 0u);

    // single item, single split, single sub-item
    {
        const uint8_t codes[1] = {0};
        RecJPQInvertedLists l1;
        l1.build(codes, 1, 1, 1);
        EXPECT_EQ(l1.list_size(0, 0), 1u);
        auto range = l1.get_list_range(0, 0);
        ASSERT_EQ(range.second - range.first, 1);
        EXPECT_EQ(*range.first, 0);
    }
    // zero items
    {
        const uint8_t codes[1] = {0};
        RecJPQInvertedLists l0;
        l0.build(codes, 0, 2, 3);
        EXPECT_EQ(l0.total_entries(), 0u);
        EXPECT_EQ(l0.list_size(1, 2), 0u);
        l0.clear();
        EXPECT_EQ(l0.offsets.size(), 0u);
    }
    // out-of-range code
    {
        const uint8_t bad[2] = {0, 5};
        RecJPQInvertedLists lb;
        EXPECT_THROW(lb.build(bad, 1, 2, 4), faiss::FaissException);
    }
}

// ---------------------------------------------------------------- B:
// scoring correctness

TEST(IndexRecJPQPrune, ScoringMatchesFullDotProduct) {
    const size_t M = 4, B = 16, ds = 32, N = 2000;
    auto data = make_recjpq(M, B, ds, N, 23);
    auto idx = make_index(data);
    auto q = random_query(data, 7);

    std::vector<float> S(M * B);
    idx->compute_sub_item_scores(q.data(), S.data());

    // for a sample of items, compare the lookup-table score against the
    // direct full-dimensional dot product with the reconstructed embedding
    std::mt19937 rng(99);
    std::uniform_int_distribution<size_t> pick(0, N - 1);
    std::vector<float> w(data.d());
    for (int t = 0; t < 200; t++) {
        const size_t i = pick(rng);
        idx->reconstruct_positions((idx_t)i, 1, w.data());
        float dot = 0.f;
        for (size_t j = 0; j < w.size(); j++) {
            dot += w[j] * q[j];
        }
        const float lut = idx->score_item(i, S.data());
        EXPECT_TRUE(score_eq(lut, dot))
                << "item " << i << ": lut " << lut << " dot " << dot;
    }
}

// ---------------------------------------------------------------- C:
// search correctness against exhaustive PQTopK

TEST(IndexRecJPQPrune, SearchMatchesExhaustive) {
    struct Config {
        size_t M, B, ds, N;
        uint32_t seed;
    };
    const Config configs[] = {
            {1, 4, 8, 64, 101},
            {2, 8, 16, 200, 102},
            {4, 16, 8, 500, 103},
            {4, 64, 8, 200, 104}, // many empty posting lists
            {8, 32, 16, 2000, 105},
            {16, 4, 32, 800, 106},
    };
    const std::vector<idx_t> k_values_base = {1, 7, 50};
    const std::vector<size_t> batch_sizes = {1, 2, 4, 8, 16, 64};

    for (const auto& cfg : configs) {
        auto data = make_recjpq(cfg.M, cfg.B, cfg.ds, cfg.N, cfg.seed);
        auto idx = make_index(data);

        for (int qi = 0; qi < 3; qi++) {
            auto q = random_query(data, cfg.seed + 1000 + qi);
            auto exp = exhaustive_topk(data, q.data(), (idx_t)cfg.N + 3);

            std::vector<idx_t> ks = k_values_base;
            ks.push_back((idx_t)(cfg.N / 2));
            ks.push_back((idx_t)cfg.N);
            ks.push_back((idx_t)cfg.N + 3);
            std::sort(ks.begin(), ks.end());
            ks.erase(std::unique(ks.begin(), ks.end()), ks.end());

            for (idx_t k : ks) {
                for (size_t bs : batch_sizes) {
                    SearchParametersRecJPQPrune params;
                    params.batch_size = bs;
                    std::vector<float> D;
                    std::vector<idx_t> I;
                    do_search(*idx, q.data(), k, &params, D, I);
                    ASSERT_TRUE(idx->last_search_complete.load());
                    check_results(
                            exp,
                            data,
                            D.data(),
                            I.data(),
                            k,
                            "config M=" + std::to_string(cfg.M) +
                                    " B=" + std::to_string(cfg.B) +
                                    " ds=" + std::to_string(cfg.ds) +
                                    " N=" + std::to_string(cfg.N) +
                                    " k=" + std::to_string(k) +
                                    " bs=" + std::to_string(bs));
                }
            }
        }
    }
}

TEST(IndexRecJPQPrune, SearchNegativeScores) {
    // positive codebooks + negative query => mostly negative scores
    const size_t M = 8, B = 32, ds = 16, N = 3000;
    auto data = make_recjpq(
            M, B, ds, N, 111, nullptr, 1.0f);
    for (auto& v : data.codebooks) {
        v = std::fabs(v);
    }
    auto idx = make_index(data);

    auto qpos = random_query(data, 5);
    std::vector<float> qneg(qpos.size());
    for (size_t i = 0; i < qpos.size(); i++) {
        qneg[i] = -qpos[i];
    }

    for (const auto& q : {qpos, qneg}) {
        auto exp = exhaustive_topk(data, q.data(), 20);
        std::vector<float> D;
        std::vector<idx_t> I;
        do_search(*idx, q.data(), 20, nullptr, D, I);
        check_results(exp, data, D.data(), I.data(), 20, "negative scores");
        // all top scores must be negative for the negated query
        if (&q == &qneg) {
            EXPECT_LT(D[0], 0.f);
        }
    }
}

TEST(IndexRecJPQPrune, SearchAllEqualScores) {
    const size_t M = 8, B = 32, ds = 16, N = 500;
    auto data = make_recjpq(M, B, ds, N, 112);
    auto idx = make_index(data);
    std::vector<float> q(data.d(), 0.f); // zero query: every score is 0

    std::vector<float> D;
    std::vector<idx_t> I;
    do_search(*idx, q.data(), 20, nullptr, D, I);
    ASSERT_TRUE(idx->last_search_complete.load());
    for (int r = 0; r < 20; r++) {
        ASSERT_EQ(I[r], I[r]); // no-op to keep indexing
        EXPECT_TRUE(score_eq(D[r], 0.f));
        EXPECT_NE(I[r], -1);
        EXPECT_NE(I[r], -2);
    }
    // all returned ids distinct and valid
    std::vector<idx_t> got(I.begin(), I.end());
    std::sort(got.begin(), got.end());
    EXPECT_EQ(std::adjacent_find(got.begin(), got.end()), got.end());
    for (idx_t id : got) {
        EXPECT_NE(data.id_to_pos.count(id), 0u);
    }
    // deterministic: a second search must return the same list
    std::vector<float> D2;
    std::vector<idx_t> I2;
    do_search(*idx, q.data(), 20, nullptr, D2, I2);
    EXPECT_EQ(I, I2);
}

TEST(IndexRecJPQPrune, SearchSingleSplitAndSingleSubItem) {
    // M = 1
    {
        auto data = make_recjpq(1, 8, 8, 300, 113);
        auto idx = make_index(data);
        auto q = random_query(data, 1);
        auto exp = exhaustive_topk(data, q.data(), 10);
        std::vector<float> D;
        std::vector<idx_t> I;
        do_search(*idx, q.data(), 10, nullptr, D, I);
        check_results(exp, data, D.data(), I.data(), 10, "M=1");
    }
    // B = 1: every item has the same code, hence the same score
    {
        auto data = make_recjpq(4, 1, 8, 200, 114);
        auto idx = make_index(data);
        auto q = random_query(data, 2);
        std::vector<float> D;
        std::vector<idx_t> I;
        do_search(*idx, q.data(), 50, nullptr, D, I);
        ASSERT_TRUE(idx->last_search_complete.load());
        float s0 = D[0];
        for (int r = 0; r < 50; r++) {
            EXPECT_TRUE(score_eq(D[r], s0)) << "rank " << r;
            EXPECT_NE(I[r], -1);
        }
        // the score must match the sum of the four sub-item scores
        std::vector<float> S(4);
        idx->compute_sub_item_scores(q.data(), S.data());
        float expected = 0.f;
        for (size_t m = 0; m < 4; m++) {
            expected += S[m];
        }
        EXPECT_TRUE(score_eq(s0, expected));
    }
}

// ---------------------------------------------------------------- D:
// edge cases

TEST(IndexRecJPQPrune, EmptyIndex) {
    const size_t M = 4, B = 8, ds = 8;
    IndexRecJPQPrune idx(M * ds, M, B);
    std::vector<float> cb(M * B * ds, 1.f);
    idx.build(0, nullptr, cb.data(), nullptr);
    ASSERT_EQ(idx.ntotal, 0);

    std::vector<float> q(M * ds, 0.5f);
    std::vector<float> D;
    std::vector<idx_t> I;
    do_search(idx, q.data(), 5, nullptr, D, I);
    ASSERT_TRUE(idx.last_search_complete.load());
    for (int r = 0; r < 5; r++) {
        EXPECT_EQ(I[r], -1);
        EXPECT_EQ(D[r], kNegInf);
    }
    // k = 0
    D.clear();
    I.clear();
    idx.search(1, q.data(), 0, D.data(), I.data());
    EXPECT_TRUE(D.empty());
}

TEST(IndexRecJPQPrune, KGreaterThanN) {
    const size_t M = 2, B = 4, ds = 4, N = 50;
    auto data = make_recjpq(M, B, ds, N, 115);
    auto idx = make_index(data);
    auto q = random_query(data, 3);
    auto exp = exhaustive_topk(data, q.data(), 100);
    std::vector<float> D;
    std::vector<idx_t> I;
    do_search(*idx, q.data(), 100, nullptr, D, I);
    ASSERT_TRUE(idx->last_search_complete.load());
    check_results(exp, data, D.data(), I.data(), 100, "k > N");
    // the first N results are the full catalogue in score order
    EXPECT_EQ(I[N - 1] < 0, false);
    EXPECT_EQ(I[N], -1);
}

TEST(IndexRecJPQPrune, DuplicateExternalIds) {
    const size_t M = 2, B = 8, ds = 4, N = 40;
    auto data = make_recjpq(M, B, ds, N, 116);
    // make two pairs of items share ids
    data.ids[3] = data.ids[0];
    data.ids[7] = data.ids[1];
    data.id_to_pos.clear(); // not unique anymore
    auto idx = make_index(data);
    auto q = random_query(data, 4);

    std::vector<float> D;
    std::vector<idx_t> I;
    do_search(*idx, q.data(), (idx_t)N, nullptr, D, I);
    ASSERT_TRUE(idx->last_search_complete.load());
    // every internal position must be returned exactly once: the returned
    // ids (as a multiset) must equal the catalogue ids
    std::vector<idx_t> got(I.begin(), I.end());
    std::sort(got.begin(), got.end());
    std::vector<idx_t> catalogue(data.ids.begin(), data.ids.end());
    std::sort(catalogue.begin(), catalogue.end());
    EXPECT_EQ(got, catalogue);
    // the duplicated ids appear twice
    EXPECT_EQ(std::count(got.begin(), got.end(), data.ids[0]), 2);
    EXPECT_EQ(std::count(got.begin(), got.end(), data.ids[1]), 2);
    // each returned score must match the score of some position holding
    // that id (duplicate ids make id -> position ambiguous)
    std::vector<float> S(M * B);
    idx->compute_sub_item_scores(q.data(), S.data());
    for (size_t r = 0; r < N; r++) {
        bool found = false;
        for (size_t p = 0; p < N; p++) {
            if (data.ids[p] == I[r] &&
                score_eq(D[r], idx->score_item(p, S.data()))) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "rank " << r << " id " << I[r];
    }
}

TEST(IndexRecJPQPrune, NonContiguousAndNegativeIds) {
    const size_t M = 4, B = 16, ds = 8, N = 300;
    auto data = make_recjpq(M, B, ds, N, 117, [](size_t i) {
        // arbitrary, non-contiguous, partly negative ids
        return (idx_t)(i * 977 + 12345) * (i % 2 ? 1 : -1);
    });
    auto idx = make_index(data);
    auto q = random_query(data, 5);
    auto exp = exhaustive_topk(data, q.data(), 25);
    std::vector<float> D;
    std::vector<idx_t> I;
    do_search(*idx, q.data(), 25, nullptr, D, I);
    ASSERT_TRUE(idx->last_search_complete.load());
    check_results(exp, data, D.data(), I.data(), 25, "non-contiguous ids");
}

TEST(IndexRecJPQPrune, ResetAndRebuild) {
    const size_t M = 2, B = 8, ds = 4;
    auto data1 = make_recjpq(M, B, ds, 100, 118);
    auto data2 = make_recjpq(M, B, ds, 150, 119);
    auto idx = make_index(data1);
    auto q = random_query(data1, 6);
    auto exp1 = exhaustive_topk(data1, q.data(), 10);
    std::vector<float> D;
    std::vector<idx_t> I;
    do_search(*idx, q.data(), 10, nullptr, D, I);
    check_results(exp1, data1, D.data(), I.data(), 10, "before reset");

    idx->reset();
    ASSERT_EQ(idx->ntotal, 0);
    idx->build(
            data2.N, data2.codes.data(), data2.codebooks.data(),
            data2.ids.data());
    ASSERT_EQ(idx->ntotal, 150);
    auto exp2 = exhaustive_topk(data2, q.data(), 10);
    D.clear();
    I.clear();
    do_search(*idx, q.data(), 10, nullptr, D, I);
    check_results(exp2, data2, D.data(), I.data(), 10, "after rebuild");
    // build on a non-empty index must throw
    EXPECT_THROW(
            idx->build(
                    data2.N,
                    data2.codes.data(),
                    data2.codebooks.data(),
                    nullptr),
            faiss::FaissException);
}

TEST(IndexRecJPQPrune, AddSaCodes) {
    const size_t M = 2, B = 8, ds = 4;
    auto data1 = make_recjpq(M, B, ds, 100, 120);
    auto data2 = make_recjpq(M, B, ds, 50, 121);
    auto idx = make_index(data1);

    std::vector<idx_t> ids2(data2.N);
    for (size_t i = 0; i < data2.N; i++) {
        ids2[i] = 10000 + data2.ids[i];
    }
    idx->add_sa_codes((idx_t)data2.N, data2.codes.data(), ids2.data());
    ASSERT_EQ(idx->ntotal, 150);

    // combined catalogue
    RecJPQData both;
    both.M = M;
    both.B = B;
    both.d_split = ds;
    both.N = 150;
    both.codes = data1.codes;
    both.codebooks = data1.codebooks;
    both.ids = data1.ids;
    for (size_t i = 0; i < data2.N; i++) {
        both.codes.insert(
                both.codes.end(),
                data2.codes.begin(),
                data2.codes.end());
        both.ids.push_back(10000 + data2.ids[i]);
    }
    both.id_to_pos.reserve(300);
    for (size_t i = 0; i < both.N; i++) {
        both.id_to_pos[both.ids[i]] = (idx_t)i;
    }
    auto q = random_query(both, 7);
    auto exp = exhaustive_topk(both, q.data(), 20);
    std::vector<float> D;
    std::vector<idx_t> I;
    do_search(*idx, q.data(), 20, nullptr, D, I);
    ASSERT_TRUE(idx->last_search_complete.load());
    check_results(exp, both, D.data(), I.data(), 20, "add_sa_codes");
}

TEST(IndexRecJPQPrune, Reconstruct) {
    const size_t M = 4, B = 8, ds = 8, N = 200;
    auto data = make_recjpq(M, B, ds, N, 122);
    auto idx = make_index(data, false); // sequential ids
    auto idx_arb = make_index(data, true);

    std::vector<float> r(M * ds);
    // sequential
    idx->reconstruct(42, r.data());
    idx->reconstruct_positions(42, 1, r.data());
    for (size_t m = 0; m < M; m++) {
        const float* cb =
                data.codebooks.data() +
                (m * B + data.codes[42 * M + m]) * ds;
        for (size_t j = 0; j < ds; j++) {
            EXPECT_FLOAT_EQ(r[m * ds + j], cb[j]);
        }
    }
    // batch by keys (arbitrary ids)
    std::vector<idx_t> keys = {0, 42, N - 1};
    std::vector<float> rb(3 * M * ds);
    idx_arb->reconstruct_batch(3, keys.data(), rb.data());
    for (size_t t = 0; t < 3; t++) {
        idx_arb->reconstruct(keys[t], rb.data() + t * M * ds);
        // must match the position-based reconstruction
        const idx_t pos = data.id_to_pos[keys[t]];
        std::vector<float> rp(M * ds);
        idx_arb->reconstruct_positions(pos, 1, rp.data());
        for (size_t j = 0; j < rp.size(); j++) {
            EXPECT_FLOAT_EQ(rb[t * M * ds + j], rp[j]);
        }
    }
    // unknown id throws
    EXPECT_THROW(idx->reconstruct(100000, r.data()), faiss::FaissException);
    // sa_decode matches reconstruct
    std::vector<uint8_t> codes = data.codes;
    std::vector<float> dec(2 * M * ds);
    idx->sa_decode(2, codes.data(), dec.data());
    EXPECT_EQ(idx->sa_code_size(), M);
    std::vector<float> dec2(2 * M * ds);
    idx->reconstruct_positions(0, 2, dec2.data());
    for (size_t j = 0; j < dec.size(); j++) {
        EXPECT_FLOAT_EQ(dec[j], dec2[j]);
    }
}

TEST(IndexRecJPQPrune, AddThrows) {
    const size_t M = 2, B = 4, ds = 4, N = 10;
    auto data = make_recjpq(M, B, ds, N, 123);
    auto idx = make_index(data);
    std::vector<float> x(N * M * ds, 0.1f);
    std::vector<idx_t> ids(N);
    std::iota(ids.begin(), ids.end(), 0);
    EXPECT_THROW(idx->add(N, x.data()), faiss::FaissException);
    EXPECT_THROW(idx->add_with_ids(N, x.data(), ids.data()), faiss::FaissException);
    EXPECT_THROW(idx->sa_encode(N, x.data(), nullptr), faiss::FaissException);
}

TEST(IndexRecJPQPrune, ConstructorValidation) {
    // d not a multiple of M
    EXPECT_THROW(IndexRecJPQPrune(10, 3, 4), faiss::FaissException);
    // B out of range
    EXPECT_THROW(IndexRecJPQPrune(8, 2, 0), faiss::FaissException);
    EXPECT_THROW(IndexRecJPQPrune(8, 2, 257), faiss::FaissException);
    // unsupported metric
    EXPECT_THROW(
            IndexRecJPQPrune(8, 2, 4, MetricType::METRIC_L2),
            faiss::FaissException);
    // good construction
    IndexRecJPQPrune ok(8, 2, 4);
    ASSERT_EQ(ok.metric_type, MetricType::METRIC_INNER_PRODUCT);
    ASSERT_EQ(ok.is_trained, true);
    ASSERT_EQ(ok.d_split, 4u);
}

TEST(IndexRecJPQPrune, ExactModeValidation) {
    const size_t M = 2, B = 4, ds = 4, N = 20;
    auto data = make_recjpq(M, B, ds, N, 124);
    auto idx = make_index(data);
    auto q = random_query(data, 8);
    std::vector<float> D;
    std::vector<idx_t> I;

    // exact mode rejects ub_inflation > 1
    {
        SearchParametersRecJPQPrune params;
        params.ub_inflation = 1.5f;
        EXPECT_THROW(
                do_search(*idx, q.data(), 5, &params, D, I),
                faiss::FaissException);
    }
    // ub_inflation < 1 is rejected in any mode
    {
        SearchParametersRecJPQPrune params;
        params.ub_inflation = 0.5f;
        EXPECT_THROW(
                do_search(*idx, q.data(), 5, &params, D, I),
                faiss::FaissException);
    }
    // batch_size = 0 is rejected
    {
        SearchParametersRecJPQPrune params;
        params.batch_size = 0;
        EXPECT_THROW(
                do_search(*idx, q.data(), 5, &params, D, I),
                faiss::FaissException);
    }
}

TEST(IndexRecJPQPrune, MaxIterationsIncomplete) {
    const size_t M = 8, B = 32, ds = 16, N = 10000;
    auto data = make_recjpq(M, B, ds, N, 125);
    auto idx = make_index(data);
    auto q = random_query(data, 9);
    auto exp = exhaustive_topk(data, q.data(), (idx_t)N);

    // one iteration can score only a fraction of the catalogue
    SearchParametersRecJPQPrune params;
    params.exact_mode = false;
    params.max_iterations = 1;
    params.batch_size = 1;
    std::vector<float> D;
    std::vector<idx_t> I;
    do_search(*idx, q.data(), (idx_t)N, &params, D, I);
    EXPECT_FALSE(idx->last_search_complete.load());

    // with enough iterations and no inflation, the search is complete and
    // exact
    params.max_iterations = -1;
    do_search(*idx, q.data(), 100, &params, D, I);
    EXPECT_TRUE(idx->last_search_complete.load());
    check_results(exp, data, D.data(), I.data(), 100, "capped then full");

    // stats report what happened
    RecJPQPruneStats stats;
    SearchParametersRecJPQPrune p2;
    p2.exact_mode = false;
    p2.max_iterations = 1;
    p2.batch_size = 1;
    p2.stats = &stats;
    do_search(*idx, q.data(), 10, &p2, D, I);
    EXPECT_EQ(stats.iterations, 1u);
    EXPECT_LT(stats.items_scored, (size_t)N);
    EXPECT_EQ(stats.sub_items_processed, 1u);
}

TEST(IndexRecJPQPrune, ConcurrentQueries) {
    const size_t M = 8, B = 32, ds = 16, N = 20000;
    auto data = make_recjpq(M, B, ds, N, 126);
    auto idx = make_index(data);

    const int nthreads = 8;
    const int nq = 25;
    // precompute queries and their exhaustive answers
    std::vector<std::vector<float>> queries(nthreads * nq);
    std::vector<Exhaustive> answers(nthreads * nq);
    for (int i = 0; i < nthreads * nq; i++) {
        queries[i] = random_query(data, 2000 + i);
        answers[i] = exhaustive_topk(data, queries[i].data(), 10);
    }

    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; t++) {
        threads.emplace_back([&, t]() {
            std::vector<float> D(10);
            std::vector<idx_t> I(10);
            // each thread searches in mini-batches of 2 and singletons
            for (int q = 0; q < nq; q++) {
                idx->search(
                        1,
                        queries[t * nq + q].data(),
                        10,
                        D.data(),
                        I.data());
                Exhaustive& exp = answers[t * nq + q];
                for (int r = 0; r < 10; r++) {
                    if (!score_eq(D[r], exp.top_scores[r]) ||
                        I[r] != exp.top_ids[r]) {
                        // random data: no ties expected
                        failures++;
                        return;
                    }
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(failures.load(), 0);
}

// ---------------------------------------------------------------- E:
// serialization and cloning

TEST(IndexRecJPQPrune, IORoundTrip) {
    const size_t M = 4, B = 16, ds = 8, N = 500;
    auto data = make_recjpq(M, B, ds, N, 127, [](size_t i) {
        return (idx_t)(1000000 + i * 31);
    });
    auto idx = make_index(data);
    auto q = random_query(data, 10);
    std::vector<float> D1;
    std::vector<idx_t> I1;
    do_search(*idx, q.data(), 20, nullptr, D1, I1);
    std::vector<float> r1(M * ds);
    idx->reconstruct(data.ids[7], r1.data());

    char tmpl[] = "/tmp/faiss_recjpq_XXXXXX";
    int fd = mkstemp(tmpl);
    ASSERT_GE(fd, 0);
    close(fd);
    unlink(tmpl);
    const std::string path = tmpl;

    faiss::write_index(idx.get(), path.c_str());
    auto idx2 = faiss::read_index(path.c_str());
    unlink(path.c_str());

    ASSERT_NE(idx2, nullptr);
    ASSERT_EQ(idx2->d, idx->d);
    ASSERT_EQ(idx2->ntotal, idx->ntotal);
    ASSERT_EQ(idx2->metric_type, idx->metric_type);
    auto* irj2 = dynamic_cast<IndexRecJPQPrune*>(idx2);
    ASSERT_NE(irj2, nullptr);
    ASSERT_EQ(irj2->M, M);
    ASSERT_EQ(irj2->B, B);

    std::vector<float> D2;
    std::vector<idx_t> I2;
    do_search(*irj2, q.data(), 20, nullptr, D2, I2);
    EXPECT_EQ(D1, D2);
    EXPECT_EQ(I1, I2);

    std::vector<float> r2(M * ds);
    idx2->reconstruct(data.ids[7], r2.data());
    EXPECT_EQ(r1, r2);
    delete idx2;
}

TEST(IndexRecJPQPrune, Clone) {
    const size_t M = 4, B = 16, ds = 8, N = 300;
    auto data = make_recjpq(M, B, ds, N, 128);
    auto idx = make_index(data);
    auto q = random_query(data, 11);
    std::vector<float> D1;
    std::vector<idx_t> I1;
    do_search(*idx, q.data(), 15, nullptr, D1, I1);

    auto clone = faiss::clone_index(idx.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_EQ(clone->ntotal, idx->ntotal);
    auto* irj = dynamic_cast<IndexRecJPQPrune*>(clone);
    ASSERT_NE(irj, nullptr);
    ASSERT_EQ(irj->M, M);
    ASSERT_EQ(irj->B, B);
    ASSERT_EQ(irj->memory_footprint(), idx->memory_footprint());

    std::vector<float> D2;
    std::vector<idx_t> I2;
    do_search(*irj, q.data(), 15, nullptr, D2, I2);
    EXPECT_EQ(D1, D2);
    EXPECT_EQ(I1, I2);
    delete clone;
}

TEST(IndexRecJPQPrune, MemoryFootprint) {
    const size_t M = 8, B = 256, ds = 64, N = 100000;
    auto data = make_recjpq(M, B, ds, N, 129);
    auto idx = make_index(data);
    const size_t expected =
            N * M * sizeof(uint8_t) + N * sizeof(idx_t) +
            M * B * ds * sizeof(float) + (M * B + 1) * sizeof(size_t) +
            N * M * sizeof(idx_t);
    EXPECT_EQ(idx->memory_footprint(), expected);
}
