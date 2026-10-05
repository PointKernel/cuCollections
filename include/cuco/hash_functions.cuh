/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/__config>
#include <cuco/detail/hash_functions/identity_hash.cuh>
#include <cuco/detail/hash_functions/murmurhash3.cuh>

#include <cuda/functional>

namespace cuco {

/**
 * @brief An Identity hash function to hash the given argument on host and device
 *
 * @throw A key must not be larger than uint64_t
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
using identity_hash = detail::identity_hash<Key>;

/**
 * @brief The 32-bit integer finalizer function of `MurmurHash3` to hash the given argument on host
 * and device.
 *
 * @throw Key type must be 4 bytes in size
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
using murmurhash3_fmix_32 = detail::MurmurHash3_fmix32<Key>;

/**
 * @brief The 64-bit integer finalizer function of `MurmurHash3` to hash the given argument on host
 * and device.
 *
 * @throw Key type must be 8 bytes in size
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
using murmurhash3_fmix_64 = detail::MurmurHash3_fmix64<Key>;

/**
 * @brief Default 32-bit xxHash function, provided by CCCL.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
using default_hash_function = cuda::hash<Key, cuda::hash_algorithm::xxhash_32>;

}  // namespace cuco
