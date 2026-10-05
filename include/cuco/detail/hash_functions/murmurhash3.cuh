/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuco::detail {

/**
 * @brief The 32-bit integer finalizer function of `MurmurHash3`.
 *
 * This function implements the final mixing step of the `MurmurHash3` algorithm for 32-bit values.
 * It is designed to improve the avalanche behavior of the hash, ensuring that changes in input bits
 * have a more uniform effect on all output bits.
 *
 * @throw Key type must be 4 bytes in size
 *
 * @tparam Key The type of the value to finalize
 *
 * @param key The input value to finalize
 * @param seed Optional seed value
 * @return The finalized 32-bit hash value
 */
template <typename Key>
__host__ __device__ constexpr std::uint32_t fmix32(Key key, std::uint32_t seed = 0) noexcept
{
  static_assert(sizeof(Key) == 4, "Key type must be 4 bytes in size.");

  std::uint32_t h = static_cast<std::uint32_t>(key) ^ seed;
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

/**
 * @brief The 64-bit integer finalizer function of `MurmurHash3`.
 *
 * This function implements the final mixing step of the `MurmurHash3` algorithm for 64-bit values.
 * It is designed to improve the avalanche behavior of the hash, ensuring that changes in input bits
 * have a more uniform effect on all output bits.
 *
 * @throw Key type must be 8 bytes in size
 *
 * @tparam Key The type of the value to finalize
 *
 * @param key The input value to finalize
 * @param seed Optional seed value
 * @return The finalized 64-bit hash value
 */
template <typename Key>
__host__ __device__ constexpr std::uint64_t fmix64(Key key, std::uint64_t seed = 0) noexcept
{
  static_assert(sizeof(Key) == 8, "Key type must be 8 bytes in size.");

  std::uint64_t h = static_cast<std::uint64_t>(key) ^ seed;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h;
}

/**
 * @brief The 32bit integer finalizer hash function of `MurmurHash3`.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_fmix32 {
  using argument_type = Key;            ///< The type of the values taken as argument
  using result_type   = std::uint32_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a MurmurHash3_fmix32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr MurmurHash3_fmix32(std::uint32_t seed = 0) : seed_{seed} {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    return fmix32(key, seed_);
  }

 private:
  std::uint32_t seed_;
};

/**
 * @brief The 64bit integer finalizer hash function of `MurmurHash3`.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_fmix64 {
  using argument_type = Key;            ///< The type of the values taken as argument
  using result_type   = std::uint64_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a MurmurHash3_fmix64 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr MurmurHash3_fmix64(std::uint64_t seed = 0) : seed_{seed} {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    return fmix64(key, seed_);
  }

 private:
  std::uint64_t seed_;
};

}  // namespace cuco::detail
