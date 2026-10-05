/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <test_utils.hpp>

#include <cuco/detail/__config>
#include <cuco/hash_functions.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>

template <typename Hash, typename... HashConstructorArgs>
static __host__ __device__ bool check_hash_result(
  typename Hash::argument_type const& key,
  typename Hash::result_type expected,
  HashConstructorArgs&&... hash_constructor_args) noexcept
{
  Hash h(cuda::std::forward<HashConstructorArgs>(hash_constructor_args)...);
  return (h(key) == expected);
}

template <typename OutputIter>
__global__ void check_identity_hash_result_kernel(OutputIter result)
{
  int i = 0;

  result[i++] = check_hash_result<cuco::identity_hash<signed char>>(0, 0);
  result[i++] = check_hash_result<cuco::identity_hash<signed char>>(
    cuda::std::numeric_limits<signed char>::max(), cuda::std::numeric_limits<signed char>::max());

  result[i++] = check_hash_result<cuco::identity_hash<int32_t>>(0, 0);
  result[i++] = check_hash_result<cuco::identity_hash<int32_t>>(
    cuda::std::numeric_limits<int32_t>::max(), cuda::std::numeric_limits<int32_t>::max());

  result[i++] = check_hash_result<cuco::identity_hash<int64_t>>(0, 0);
  result[i++] = check_hash_result<cuco::identity_hash<int64_t>>(
    cuda::std::numeric_limits<int64_t>::max(), cuda::std::numeric_limits<int64_t>::max());
}

TEST_CASE("utility cuco::identity_hash test", "")
{
  SECTION("Check if host-generated hash values match the identity function.")
  {
    CHECK(check_hash_result<cuco::identity_hash<signed char>>(0, 0));
    CHECK(check_hash_result<cuco::identity_hash<signed char>>(
      cuda::std::numeric_limits<signed char>::max(),
      cuda::std::numeric_limits<signed char>::max()));

    CHECK(check_hash_result<cuco::identity_hash<int32_t>>(0, 0));
    CHECK(check_hash_result<cuco::identity_hash<int32_t>>(
      cuda::std::numeric_limits<int32_t>::max(), cuda::std::numeric_limits<int32_t>::max()));

    CHECK(check_hash_result<cuco::identity_hash<int64_t>>(0, 0));
    CHECK(check_hash_result<cuco::identity_hash<int64_t>>(
      cuda::std::numeric_limits<int64_t>::max(), cuda::std::numeric_limits<int64_t>::max()));
  }
  SECTION("Check if device-generated hash values match the identity function.")
  {
    thrust::device_vector<bool> result(7, true);

    check_identity_hash_result_kernel<<<1, 1>>>(result.begin());

    CHECK(cuco::test::all_of(result.begin(), result.end(), cuda::std::identity{}));
  }
}
