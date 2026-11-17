/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Benchmark for faiss::IndexRecJPQPrune against an exhaustive PQTopK
// baseline (score every item with the sub-item lookup table, then
// select the top-K).
//
// Usage: bench_recjpq_prune [N] [n_queries] [k] [dataset]
//   N          catalogue size        (default 1000000)
//   n_queries  number of queries     (default 100)
//   k          top-K                 (default 100)
//   dataset    "random" or "structured" (default "random")
//
// The synthetic catalogue uses the paper's typical recommendation
// shape: M = 8 splits, B = 256 sub-item ids per split, d_split = 64
// (d = 512). Codes are uniform, queries are unit-norm Gaussian.
//
//   random:     codebook entries are i.i.d. Gaussian. A hard case for
//               the pruning (no score hierarchy), so the pruning ratios
//               are conservative.
//   structured: codebook entries are unit directions scaled by a
//               log-normal magnitude, emulating the wide spread of
//               sub-item scores found in real recommendation
//               embeddings, where the pruning pays off.
//
// For stable timings run with a single BLAS/OpenMP thread:
//   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./bench_recjpq_prune

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <faiss/IndexRecJPQPrune.h>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

float percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    if (v.empty()) {
        return 0.0;
    }
    const size_t idx =
            std::min<size_t>(v.size() - 1, (size_t)std::llround(p / 100.0 *
                                                                  (v.size() - 1)));
    return (float)v[idx];
}

/// exhaustive PQTopK: score every item with the S lookup table and
/// return the k best (score, position) pairs in descending score order
struct ExhaustiveTopK {
    std::vector<float> scores; ///< k, descending
    std::vector<int64_t> positions; ///< k
};

ExhaustiveTopK exhaustive_search(
        const faiss::IndexRecJPQPrune& idx,
        const std::vector<float>& S,
        int64_t k) {
    const int64_t N = idx.ntotal;
    const size_t M = idx.M;
    const size_t B = idx.B;
    const uint8_t* codes = idx.item_codes.data();

    std::vector<float> all_scores((size_t)N);
    for (int64_t i = 0; i < N; i++) {
        const uint8_t* c = codes + (size_t)i * M;
        float s = 0.f;
        for (size_t m = 0; m < M; m++) {
            s += S[m * B + c[m]];
        }
        all_scores[(size_t)i] = s;
    }
    const size_t keep = std::min<size_t>((size_t)k, (size_t)N);
    std::vector<size_t> order((size_t)N);
    std::iota(order.begin(), order.end(), (size_t)0);
    std::partial_sort(
            order.begin(),
            order.begin() + keep,
            order.end(),
            [&all_scores](size_t a, size_t b) {
                return all_scores[a] > all_scores[b];
            });
    ExhaustiveTopK out;
    out.scores.resize(keep);
    out.positions.resize(keep);
    for (size_t r = 0; r < keep; r++) {
        out.scores[r] = all_scores[order[r]];
        out.positions[r] = (int64_t)order[r];
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int64_t N = argc > 1 ? (int64_t)std::atoll(argv[1]) : 1000000;
    const size_t n_queries = argc > 2 ? (size_t)std::atoll(argv[2]) : 100;
    const int64_t k = argc > 3 ? (int64_t)std::atoll(argv[3]) : 100;
    const std::string dataset =
            argc > 4 ? argv[4] : "random";
    if (dataset != "random" && dataset != "structured") {
        std::fprintf(
                stderr,
                "dataset must be \"random\" or \"structured\", got %s\n",
                argv[4]);
        return 2;
    }

    const size_t M = 8, B = 256, d_split = 64;
    const size_t d = M * d_split;

    std::printf("RecJPQPrune benchmark (dataset: %s)\n", dataset.c_str());
    std::printf(
            "M=%zu B=%zu d_split=%zu d=%zu N=%ld n_queries=%zu k=%ld\n",
            M,
            B,
            d_split,
            d,
            (long)N,
            n_queries,
            (long)k);

    // ---------------- catalogue: uniform codes, Gaussian codebooks
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> subid(0, (int)B - 1);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<uint8_t> item_codes((size_t)N * M);
    for (auto& c : item_codes) {
        c = (uint8_t)subid(rng);
    }
    std::vector<float> codebooks(M * B * d_split);
    if (dataset == "random") {
        for (auto& v : codebooks) {
            v = gauss(rng);
        }
    } else {
        // unit direction x log-normal magnitude, per sub-item
        std::lognormal_distribution<float> mag(0.0f, 1.0f);
        for (size_t e = 0; e < M * B; e++) {
            float* v = codebooks.data() + e * d_split;
            for (size_t j = 0; j < d_split; j++) {
                v[j] = gauss(rng);
            }
            float nrm = 0.f;
            for (size_t j = 0; j < d_split; j++) {
                nrm += v[j] * v[j];
            }
            nrm = std::sqrt(nrm);
            const float scale = mag(rng) / nrm;
            for (size_t j = 0; j < d_split; j++) {
                v[j] *= scale;
            }
        }
    }

    // ---------------- build
    faiss::IndexRecJPQPrune idx(d, M, B);
    auto t0 = Clock::now();
    idx.build((size_t)N, item_codes.data(), codebooks.data());
    const double build_ms = ms_since(t0);
    const size_t footprint = idx.memory_footprint();
    std::printf(
            "build: %.1f ms, memory: %.2f MB (%zu bytes)\n",
            build_ms,
            footprint / (1024.0 * 1024.0),
            footprint);

    // ---------------- queries: unit-norm Gaussian
    std::vector<float> queries(n_queries * d);
    for (size_t i = 0; i < n_queries; i++) {
        float* q = queries.data() + i * d;
        for (size_t j = 0; j < d; j++) {
            q[j] = gauss(rng);
        }
        float nrm = 0.f;
        for (size_t j = 0; j < d; j++) {
            nrm += q[j] * q[j];
        }
        nrm = std::sqrt(nrm);
        for (size_t j = 0; j < d; j++) {
            q[j] /= nrm;
        }
    }

    // ---------------- warm-up (thread scratch allocation, page faults)
    {
        std::vector<float> D((size_t)k);
        std::vector<int64_t> I((size_t)k);
        idx.search(1, queries.data(), k, D.data(), I.data());
    }

    // ---------------- exhaustive baseline, timed
    std::vector<float> S(M * B);
    {
        auto tb = Clock::now();
        for (size_t i = 0; i < n_queries; i++) {
            idx.compute_sub_item_scores(queries.data() + i * d, S.data());
            ExhaustiveTopK top = exhaustive_search(idx, S, k);
            (void)top;
        }
        std::printf(
                "exhaustive PQTopK baseline: %.1f ms/query\n",
                ms_since(tb) / n_queries);
    }

    // ---------------- RecJPQPrune over a sweep of batch sizes
    const size_t batch_sizes[] = {1, 2, 4, 8, 16, 32, 64};
    std::printf(
            "\n%-6s %12s %12s %12s %12s %14s %14s %14s\n",
            "batch",
            "median_ms",
            "p95_ms",
            "items_scored",
            "catalogue_%",
            "iterations",
            "postings_trav",
            "sub_items_proc");
    for (size_t bs : batch_sizes) {
        faiss::SearchParametersRecJPQPrune params;
        params.batch_size = bs;
        params.stats = nullptr;
        faiss::RecJPQPruneStats stats;
        params.stats = &stats;

        std::vector<float> D((size_t)k);
        std::vector<int64_t> I((size_t)k);
        std::vector<double> latencies;
        latencies.reserve(n_queries);
        auto t1 = Clock::now();
        for (size_t i = 0; i < n_queries; i++) {
            auto tq = Clock::now();
            idx.search(
                    1, queries.data() + i * d, k, D.data(), I.data(), &params);
            latencies.push_back(ms_since(tq));
        }
        const double total_ms = ms_since(t1);
        (void)total_ms;
        std::printf(
                "%-6zu %12.2f %12.2f %12zu %12.2f %14zu %14zu %14zu\n",
                bs,
                percentile(latencies, 50) * 1000.0,
                percentile(latencies, 95) * 1000.0,
                stats.items_scored / n_queries,
                100.0 * (double)stats.items_scored /
                        ((double)n_queries * (double)N),
                stats.iterations / n_queries,
                stats.postings_traversed / n_queries,
                stats.sub_items_processed / n_queries);
    }

    // ---------------- exactness spot check against the baseline
    std::printf("\nexactness check (first 8 queries, k=%ld):\n", (long)k);
    size_t max_abs_diff = 0;
    float max_diff = 0.f;
    for (size_t i = 0; i < std::min<size_t>(8, n_queries); i++) {
        const float* q = queries.data() + i * d;
        idx.compute_sub_item_scores(q, S.data());
        ExhaustiveTopK exp = exhaustive_search(idx, S, k);
        std::vector<float> D((size_t)k);
        std::vector<int64_t> I((size_t)k);
        idx.search(1, q, k, D.data(), I.data());
        for (size_t r = 0; r < exp.scores.size(); r++) {
            const float diff = std::fabs(D[r] - exp.scores[r]);
            if (diff > max_diff) {
                max_diff = diff;
                max_abs_diff = i;
            }
        }
    }
    std::printf(
            "  max |score diff| vs exhaustive over top-%ld: %g (query %zu)\n",
            (long)k,
            max_diff,
            max_abs_diff);
    std::printf(max_diff < 1e-4f ? "  PASS: exact\n" : "  FAIL: NOT exact\n");
    return max_diff < 1e-4f ? 0 : 1;
}
