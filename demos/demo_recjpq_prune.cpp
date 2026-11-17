/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Minimal usage example for faiss::IndexRecJPQPrune: build the index
// from pre-computed RecJPQ codes and codebooks (as produced, e.g., by
// an existing IndexPQ), and search with the dynamic pruning algorithm
// of Petrov et al. (SIGIR 2025).

#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include <faiss/IndexRecJPQPrune.h>

int main() {
    // 1. configuration: d = M * d_split, B sub-item ids per split
    const size_t d = 512, M = 8, d_split = 64, B = 256;

    // 2. synthetic catalogue: in a real system the codes come from
    //    pq.compute() on an existing IndexPQ and the codebooks from
    //    pq.centroids
    const size_t N = 100000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> subid(0, (int)B - 1);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<uint8_t> item_codes(N * M);
    for (auto& c : item_codes) {
        c = (uint8_t)subid(rng);
    }
    std::vector<float> codebooks(M * B * d_split);
    for (auto& v : codebooks) {
        v = gauss(rng);
    }

    // 3. build the index (data is copied; the index owns its storage)
    faiss::IndexRecJPQPrune index(d, M, B); // METRIC_INNER_PRODUCT by default
    index.build(N, item_codes.data(), codebooks.data());
    std::printf("built index with %zu items, %.1f KB\n", N,
                index.memory_footprint() / 1024.0);

    // 4. a unit-norm query in R^d
    std::vector<float> q(d);
    for (float& v : q) {
        v = gauss(rng);
    }
    float nrm = 0.f;
    for (float v : q) {
        nrm += v * v;
    }
    for (float& v : q) {
        v /= std::sqrt(nrm);
    }

    // 5. search: exact top-K by default (SearchParametersRecJPQPrune
    //    with exact_mode = true)
    const size_t k = 10;
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);
    index.search(1, q.data(), (faiss::idx_t)k, D.data(), I.data());
    std::printf("top-%zu for the query (descending score):\n", k);
    for (size_t r = 0; r < k; r++) {
        std::printf("  %2zu. id=%lld score=%.6f\n", r + 1,
                    (long long)I[r], D[r]);
    }

    // 6. tuning / diagnostics: batch size, optional statistics
    faiss::SearchParametersRecJPQPrune params;
    params.batch_size = 16;
    faiss::RecJPQPruneStats stats;
    params.stats = &stats;
    index.search(1, q.data(), (faiss::idx_t)k, D.data(), I.data(), &params);
    std::printf(
            "search stats: %zu iterations, %zu items scored "
            "(%.1f%% of the catalogue), %zu postings traversed\n",
            stats.iterations,
            stats.items_scored,
            100.0 * (double)stats.items_scored / (double)N,
            stats.postings_traversed);

    // 7. reconstruct an (approximate) item embedding from its id
    std::vector<float> x(d);
    index.reconstruct(I[0], x.data());
    std::printf("reconstructed embedding of id %lld has norm %.3f\n",
                (long long)I[0],
                std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(),
                                             0.f)));
    return 0;
}
