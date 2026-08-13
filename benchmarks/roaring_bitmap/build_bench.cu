/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark_defaults.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/cstdint>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <optional>

using namespace cuco::utility;

template <bool Dense>
void roaring_bitmap_build(nvbench::state& state)
{
  using key_type    = cuda::std::uint32_t;
  using bitmap_type = cuco::experimental::roaring_bitmap<key_type>;

  auto const num_keys = state.get_int64("NumInputs");
  thrust::device_vector<key_type> keys(num_keys);
  if constexpr (Dense) {
    thrust::sequence(thrust::device, keys.begin(), keys.end(), key_type{0});
  } else {
    key_generator gen{};
    gen.generate(distribution::unique{}, keys.begin(), keys.end());
  }

  state.add_element_count(num_keys);
  state.add_global_memory_reads<key_type>(num_keys, "InputSize");

  std::optional<bitmap_type> bitmap;
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuda::stream_ref stream{launch.get_stream()};
               timer.start();
               bitmap.emplace(keys.begin(), keys.end(), bitmap_type::allocator_type{}, stream);
               timer.stop();
               bitmap.reset();
             });
}

void roaring_bitmap_build_dense(nvbench::state& state) { roaring_bitmap_build<true>(state); }

void roaring_bitmap_add(nvbench::state& state)
{
  using key_type    = cuda::std::uint32_t;
  using bitmap_type = cuco::experimental::roaring_bitmap<key_type>;

  auto const num_base   = state.get_int64("BaseInputs");
  auto const num_update = state.get_int64("UpdateInputs");
  thrust::device_vector<key_type> base(num_base);
  thrust::device_vector<key_type> update(num_update);
  thrust::sequence(thrust::device, base.begin(), base.end(), key_type{0});
  thrust::sequence(thrust::device, update.begin(), update.end(), static_cast<key_type>(num_base));

  state.add_element_count(num_update);
  state.add_global_memory_reads<key_type>(num_update, "UpdateSize");

  std::optional<bitmap_type> bitmap;
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuda::stream_ref stream{launch.get_stream()};
               bitmap.emplace(base.begin(), base.end(), bitmap_type::allocator_type{}, stream);
               timer.start();
               bitmap->add(update.begin(), update.end(), stream);
               timer.stop();
               bitmap.reset();
             });
}
void roaring_bitmap_build_sparse(nvbench::state& state) { roaring_bitmap_build<false>(state); }

NVBENCH_BENCH(roaring_bitmap_build_dense)
  .set_name("roaring_bitmap_build_dense")
  .add_int64_power_of_two_axis("NumInputs", {20, 24});

NVBENCH_BENCH(roaring_bitmap_build_sparse)
  .set_name("roaring_bitmap_build_sparse")
  .add_int64_power_of_two_axis("NumInputs", {20, 24});

NVBENCH_BENCH(roaring_bitmap_add)
  .set_name("roaring_bitmap_add")
  .add_int64_power_of_two_axis("BaseInputs", {20, 24})
  .add_int64_power_of_two_axis("UpdateInputs", {10, 20});
