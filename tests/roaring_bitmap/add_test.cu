/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/detail/error.hpp>
#include <cuco/roaring_bitmap.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/stream_ref>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
using key_type = cuda::std::uint32_t;

void require_membership(cuco::experimental::roaring_bitmap<key_type> const& bitmap,
                        std::vector<key_type> const& queries,
                        std::vector<bool> const& expected,
                        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}})
{
  thrust::device_vector<key_type> queries_d(queries);
  thrust::device_vector<bool> results_d(queries.size());
  bitmap.contains(queries_d.begin(), queries_d.end(), results_d.begin(), stream);
  thrust::host_vector<bool> results_h = results_d;
  REQUIRE(results_h.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(results_h[i] == expected[i]);
  }
}
}  // namespace

TEST_CASE("roaring_bitmap constructs an empty portable bitmap", "[roaring_bitmap][add]")
{
  thrust::device_vector<key_type> keys;
  cuco::experimental::roaring_bitmap<key_type> bitmap(keys.begin(), keys.end());

  REQUIRE(bitmap.empty());
  REQUIRE(bitmap.size() == 0);
  REQUIRE(bitmap.size_bytes() == 2 * sizeof(cuda::std::uint32_t));
  require_membership(bitmap, {0, 1, 0xffffffffu}, {false, false, false});
}

TEST_CASE("roaring_bitmap constructs array and bitmap containers", "[roaring_bitmap][add]")
{
  std::vector<key_type> keys;
  keys.reserve(5010);
  for (key_type key = 5000; key != 0; --key) {
    keys.push_back(key);
  }
  keys.insert(keys.end(), {0, 7, 4096, 0x00020002u, 0x0002ffffu, 0x80000001u});

  thrust::device_vector<key_type> keys_d(keys);
  cuco::experimental::roaring_bitmap<key_type> bitmap(keys_d.begin(), keys_d.end());

  REQUIRE(bitmap.size() == 5004);
  REQUIRE_FALSE(bitmap.empty());
  require_membership(
    bitmap,
    {0, 7, 4096, 5000, 5001, 0x00020002u, 0x00020003u, 0x0002ffffu, 0x80000001u, 0xffffffffu},
    {true, true, true, true, false, true, false, true, true, false});
}

TEST_CASE("roaring_bitmap add deduplicates old and new keys", "[roaring_bitmap][add]")
{
  thrust::device_vector<key_type> initial{9, 1, 9, 0x00010001u};
  cuco::experimental::roaring_bitmap<key_type> bitmap(initial.begin(), initial.end());

  thrust::device_vector<key_type> added{2, 9, 2, 0x00010002u, 0xffffffffu};
  bitmap.add(added.begin(), added.end());

  REQUIRE(bitmap.size() == 6);
  require_membership(bitmap,
                     {0, 1, 2, 9, 10, 0x00010001u, 0x00010002u, 0xffffffffu},
                     {false, true, true, true, false, true, true, true});
}

TEST_CASE("roaring_bitmap add_async orders same-stream queries", "[roaring_bitmap][add]")
{
  cudaStream_t raw_stream{};
  CUCO_CUDA_TRY(cudaStreamCreate(&raw_stream));
  cuda::stream_ref stream{raw_stream};

  {
    thrust::device_vector<key_type> initial{1, 3};
    cuco::experimental::roaring_bitmap<key_type> bitmap(initial.begin(), initial.end(), {}, stream);
    thrust::device_vector<key_type> added{2, 4};
    thrust::device_vector<key_type> added_again{4, 5, 6};
    thrust::device_vector<key_type> queries{1, 2, 3, 4, 5, 6, 7};
    thrust::device_vector<bool> results(queries.size());

    bitmap.add_async(added.begin(), added.end(), stream);
    bitmap.add_async(added_again.begin(), added_again.end(), stream);
    bitmap.contains_async(queries.begin(), queries.end(), results.begin(), stream);
    stream.sync();

    thrust::host_vector<bool> results_h = results;
    REQUIRE(results_h == thrust::host_vector<bool>{true, true, true, true, true, true, false});
    REQUIRE(bitmap.size() == 6);
  }
  CUCO_CUDA_TRY(cudaStreamDestroy(raw_stream));
}

TEST_CASE("roaring_bitmap GPU serialization round trips", "[roaring_bitmap][add]")
{
  thrust::device_vector<key_type> keys{0, 1, 7, 0x10000u, 0xffffffffu};
  cuco::experimental::roaring_bitmap<key_type> built(keys.begin(), keys.end());

  std::vector<cuda::std::byte> serialized(built.size_bytes());
  CUCO_CUDA_TRY(
    cudaMemcpy(serialized.data(), built.data(), serialized.size(), cudaMemcpyDeviceToHost));

  cuco::experimental::roaring_bitmap<key_type> loaded{
    cuda::std::span<cuda::std::byte const>{serialized.data(), serialized.size()}};
  REQUIRE(loaded.size() == 5);
  require_membership(
    loaded, {0, 1, 2, 7, 0x10000u, 0xffffffffu}, {true, true, false, true, true, true});
}

TEST_CASE("roaring_bitmap adds to serialized array and bitmap containers", "[roaring_bitmap][add]")
{
  std::vector<key_type> keys;
  for (key_type key = 0; key <= 5000; ++key) {
    keys.push_back(key);
  }
  keys.push_back(0x00020002u);
  thrust::device_vector<key_type> keys_d(keys);
  cuco::experimental::roaring_bitmap<key_type> built(keys_d.begin(), keys_d.end());

  std::vector<cuda::std::byte> serialized(built.size_bytes());
  CUCO_CUDA_TRY(
    cudaMemcpy(serialized.data(), built.data(), serialized.size(), cudaMemcpyDeviceToHost));
  cuco::experimental::roaring_bitmap<key_type> loaded{
    cuda::std::span<cuda::std::byte const>{serialized.data(), serialized.size()}};

  thrust::device_vector<key_type> added{5001, 0x00020003u};
  loaded.add(added.begin(), added.end());

  REQUIRE(loaded.size() == 5004);
  require_membership(
    loaded, {0, 5000, 5001, 5002, 0x00020002u, 0x00020003u}, {true, true, true, false, true, true});
}

#ifdef CUCO_ROARING_DATA_DIR
TEST_CASE("roaring_bitmap adds to serialized run containers", "[roaring_bitmap][add]")
{
  std::string const path = std::string{CUCO_ROARING_DATA_DIR} + "/bitmapwithruns.bin";
  if (not std::ifstream(path).good()) { SKIP("Missing RoaringFormatSpec run-container data"); }

  auto const file_size = std::filesystem::file_size(path);
  std::vector<cuda::std::byte> serialized(file_size);
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(serialized.data()), file_size);

  cuco::experimental::roaring_bitmap<key_type> bitmap{
    cuda::std::span<cuda::std::byte const>{serialized.data(), serialized.size()}};
  auto const old_size = bitmap.size();
  thrust::device_vector<key_type> added{0xfffffffeu, 0xffffffffu};

  bitmap.add(added.begin(), added.end());

  REQUIRE(bitmap.size() == old_size + 2);
  require_membership(bitmap,
                     {0, 300000, 700000, 799999, 0xfffffffeu, 0xffffffffu},
                     {true, true, true, true, true, true});
}
#endif
