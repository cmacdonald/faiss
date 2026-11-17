/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/invlists/RecJPQInvertedLists.h>

#include <cinttypes>

#include <faiss/impl/FaissAssert.h>

#include <algorithm>

namespace faiss {

void RecJPQInvertedLists::build(
        const uint8_t* item_codes,
        size_t N,
        size_t M,
        size_t B) {
    FAISS_THROW_IF_NOT_MSG(M >= 1, "RecJPQInvertedLists: M must be >= 1");
    FAISS_THROW_IF_NOT_MSG(B >= 1, "RecJPQInvertedLists: B must be >= 1");
    FAISS_THROW_IF_NOT_MSG(
            B <= 256,
            "RecJPQInvertedLists: B must be <= 256 (item codes are uint8)");
    FAISS_THROW_IF_NOT_MSG(
            N == 0 || item_codes != nullptr,
            "RecJPQInvertedLists::build: item_codes must not be null when "
            "N > 0");

    const size_t nlists = M * B;

    // pass 1: count the items per posting list (validates the codes)
    offsets.assign(nlists + 1, 0);
    for (size_t i = 0; i < N; i++) {
        const uint8_t* c = item_codes + i * M;
        for (size_t m = 0; m < M; m++) {
            FAISS_THROW_IF_NOT_FMT(
                    c[m] < B,
                    "RecJPQInvertedLists: item %" PRId64
                    " has code %u >= B (%zu) in split %zu",
                    (idx_t)i,
                    (unsigned)c[m],
                    B,
                    m);
            offsets[(m * B + c[m]) + 1]++;
        }
    }
    // prefix sum: offsets[l] becomes the start of posting list l
    for (size_t l = 0; l < nlists; l++) {
        offsets[l + 1] += offsets[l];
    }

    // pass 2: fill the contiguous storage, preserving item order
    item_ids.resize(N * M);
    std::vector<size_t> cursor(offsets.begin(), offsets.end() - 1);
    for (size_t i = 0; i < N; i++) {
        const uint8_t* c = item_codes + i * M;
        for (size_t m = 0; m < M; m++) {
            item_ids[cursor[m * B + c[m]]++] = (idx_t)i;
        }
    }

    this->M = M;
    this->B = B;
}

void RecJPQInvertedLists::clear() {
    offsets.clear();
    item_ids.clear();
    M = 0;
    B = 0;
}

size_t RecJPQInvertedLists::list_size(
        size_t split,
        size_t sub_item_id) const {
    FAISS_THROW_IF_NOT(split < M);
    FAISS_THROW_IF_NOT(sub_item_id < B);
    const size_t l = split * B + sub_item_id;
    return offsets[l + 1] - offsets[l];
}

const idx_t* RecJPQInvertedLists::get_list(
        size_t split,
        size_t sub_item_id) const {
    FAISS_THROW_IF_NOT(split < M);
    FAISS_THROW_IF_NOT(sub_item_id < B);
    const size_t l = split * B + sub_item_id;
    if (offsets[l + 1] == offsets[l]) {
        return nullptr;
    }
    return &item_ids[offsets[l]];
}

std::pair<const idx_t*, const idx_t*> RecJPQInvertedLists::get_list_range(
        size_t split,
        size_t sub_item_id) const {
    FAISS_THROW_IF_NOT(split < M);
    FAISS_THROW_IF_NOT(sub_item_id < B);
    if (item_ids.empty()) {
        return {nullptr, nullptr};
    }
    const size_t l = split * B + sub_item_id;
    const idx_t* base = item_ids.data();
    return {base + offsets[l], base + offsets[l + 1]};
}

} // namespace faiss
