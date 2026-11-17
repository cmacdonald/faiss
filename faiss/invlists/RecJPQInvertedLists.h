/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <faiss/Index.h>

namespace faiss {

/** CSR (compressed sparse row) inverted lists for IndexRecJPQPrune.
 *
 * The catalogue of N items is represented by a row-major N x M matrix of
 * item codes; the code at split m of item i is in [0, B).  There are
 * M * B posting lists, identified by
 *
 *     list_id = split * B + sub_item_id
 *
 * The posting list of (split, sub_item_id) contains the internal positions
 * of all items whose code at that split equals sub_item_id.
 *
 * Storage is CSR-style:
 *
 *     offsets has M * B + 1 entries;
 *     the items of list l are item_ids[offsets[l] .. offsets[l + 1]).
 *
 * The storage is dense and unpadded: no slot is reserved for empty posting
 * lists and there is at most one heap allocation for all list contents (and
 * one for the offsets). Within each posting list, item positions appear in
 * ascending order (the order in which items were added), which makes the
 * lists amenable to sequential access.
 */
struct RecJPQInvertedLists {
    size_t M = 0; ///< number of splits
    size_t B = 0; ///< number of sub-item ids per split

    /** offsets of the M * B + 1 posting list boundaries */
    std::vector<size_t> offsets;
    /** concatenation of all posting lists: N * M entries after build */
    std::vector<idx_t> item_ids;

    /** Two-pass construction from a row-major N x M uint8 code matrix.
     *
     * The first pass counts the items per posting list, a prefix sum turns
     * the counts into offsets and the second pass fills the contiguous
     * item storage. Item positions are preserved in ascending order.
     *
     * @param item_codes  N * M codes, the code at split m of item i is at
     *                    item_codes[i * M + m], must be in [0, B)
     * @param N           number of items
     * @param M           number of splits, >= 1
     * @param B           number of sub-item ids per split, in [1, 256]
     */
    void build(const uint8_t* item_codes, size_t N, size_t M, size_t B);

    /** Drop all postings and metadata. */
    void clear();

    /// number of items in the posting list (split, sub_item_id)
    size_t list_size(size_t split, size_t sub_item_id) const;

    /// pointer to the first entry of the posting list (nullptr when the
    /// list is empty or the lists have not been built)
    const idx_t* get_list(size_t split, size_t sub_item_id) const;

    /// half-open range [begin, end) of the posting list contents
    std::pair<const idx_t*, const idx_t*> get_list_range(
            size_t split,
            size_t sub_item_id) const;

    /// total number of postings (= N * M after a build)
    size_t total_entries() const {
        return item_ids.size();
    }
};

} // namespace faiss
