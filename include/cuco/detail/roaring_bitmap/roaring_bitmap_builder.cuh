/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/detail/roaring_bitmap/util.cuh>
#include <cuco/detail/storage/storage_base.cuh>
#include <cuco/detail/utility/memcpy_async.hpp>

#include <cub/detail/temporary_storage.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/memory>
#include <cuda/stream_ref>

#include <memory>
#include <new>
#include <type_traits>

namespace cuco::experimental::detail {

template <class T, class Allocator>
class roaring_bitmap_builder;

namespace roaring_bitmap_builder_ns {

using index_type = cuda::std::uint32_t;

template <class T, class Allocator>
using rebind_allocator_t = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;

template <class T, class Allocator>
using device_deleter =
  cuco::detail::custom_deleter<cuda::std::size_t, rebind_allocator_t<T, Allocator>>;

template <class T, class Allocator>
using device_unique_ptr = std::unique_ptr<T, device_deleter<T, Allocator>>;

template <class T, class Allocator>
device_unique_ptr<T, Allocator> make_device_buffer(Allocator const& alloc,
                                                   cuda::std::size_t size,
                                                   cuda::stream_ref stream)
{
  rebind_allocator_t<T, Allocator> rebound{alloc};
  auto const allocation_size = cuda::std::max<cuda::std::size_t>(size, 1);
  return device_unique_ptr<T, Allocator>{
    rebound.allocate(allocation_size, stream),
    cuco::detail::custom_deleter<cuda::std::size_t, rebind_allocator_t<T, Allocator>>{
      allocation_size, rebound, stream}};
}

template <class T>
struct pinned_deleter {
  void operator()(T* ptr) const noexcept
  {
    if (ptr != nullptr) { cudaFreeHost(ptr); }
  }
};

template <class T>
std::unique_ptr<T, pinned_deleter<T>> make_pinned()
{
  T* ptr{};
  CUCO_CUDA_TRY(cudaMallocHost(&ptr, sizeof(T)));
  new (ptr) T{};
  return std::unique_ptr<T, pinned_deleter<T>>{ptr};
}

template <class T>
struct to_value {
  template <class U>
  __device__ T operator()(U value) const
  {
    return static_cast<T>(value);
  }
};

struct container_cardinality {
  cuda::std::byte const* bitmap;
  roaring_bitmap_metadata<cuda::std::uint32_t> metadata;

  __device__ index_type operator()(index_type i) const
  {
    auto const card_ptr = bitmap + metadata.key_cards + (2 * i + 1) * sizeof(cuda::std::uint16_t);
    return 1u + misaligned_load<cuda::std::uint16_t>(card_ptr);
  }
};

__device__ inline index_type container_offset(
  cuda::std::byte const* bitmap,
  roaring_bitmap_metadata<cuda::std::uint32_t> const& metadata,
  index_type i)
{
  if (metadata.offsets_in_serialized_data) {
    return misaligned_load<index_type>(bitmap + metadata.container_offsets +
                                       i * sizeof(index_type));
  }
  return metadata.computed_offsets[i];
}

static __global__ void decode_containers(cuda::std::byte const* bitmap,
                                         roaring_bitmap_metadata<cuda::std::uint32_t> metadata,
                                         index_type const* container_starts,
                                         cuda::std::uint32_t* keys)
{
  for (auto i = static_cast<index_type>(blockIdx.x);
       i < static_cast<index_type>(metadata.num_containers);
       i += static_cast<index_type>(gridDim.x)) {
    auto const card         = container_cardinality{bitmap, metadata}(i);
    auto const key          = misaligned_load<cuda::std::uint16_t>(bitmap + metadata.key_cards +
                                                          2 * i * sizeof(cuda::std::uint16_t));
    auto const upper        = static_cast<cuda::std::uint32_t>(key) << 16;
    auto const* container   = bitmap + container_offset(bitmap, metadata, i);
    auto const output_start = container_starts[i];

    if (metadata.has_run and check_bit(bitmap + metadata.run_container_bitmap, i)) {
      __shared__ index_type write_index;
      if (threadIdx.x == 0) { write_index = output_start; }
      __syncthreads();

      auto const num_runs = misaligned_load<cuda::std::uint16_t>(container);
      for (auto run = static_cast<index_type>(threadIdx.x); run < num_runs; run += blockDim.x) {
        auto const* run_ptr =
          container + sizeof(cuda::std::uint16_t) + 2 * run * sizeof(cuda::std::uint16_t);
        auto const start = misaligned_load<cuda::std::uint16_t>(run_ptr);
        auto const length =
          1u + misaligned_load<cuda::std::uint16_t>(run_ptr + sizeof(cuda::std::uint16_t));
        auto const output = atomicAdd(&write_index, length);
        for (index_type j = 0; j < length; ++j) {
          keys[output + j] = upper | static_cast<cuda::std::uint16_t>(start + j);
        }
      }
    } else if (card <= roaring_bitmap_metadata<cuda::std::uint32_t>::max_array_container_card) {
      for (auto j = static_cast<index_type>(threadIdx.x); j < card; j += blockDim.x) {
        auto const lower =
          misaligned_load<cuda::std::uint16_t>(container + j * sizeof(cuda::std::uint16_t));
        keys[output_start + j] = upper | lower;
      }
    } else {
      __shared__ index_type write_index;
      if (threadIdx.x == 0) { write_index = output_start; }
      __syncthreads();

      for (index_type lower = static_cast<index_type>(threadIdx.x); lower < (1u << 16);
           lower += blockDim.x) {
        if (check_bit(container, lower)) { keys[atomicAdd(&write_index, 1u)] = upper | lower; }
      }
    }
    __syncthreads();
  }
}

inline constexpr index_type container_key_space = index_type{1} << 16;

struct container_present {
  index_type const* starts_by_key;

  __device__ index_type operator()(index_type key) const
  {
    return static_cast<index_type>(starts_by_key[key] !=
                                   cuda::std::numeric_limits<index_type>::max());
  }
};

template <class T>
__global__ void mark_container_starts(T const* unique_keys,
                                      index_type max_keys,
                                      index_type const* num_unique,
                                      index_type* starts_by_key)
{
  auto const count  = *num_unique;
  auto const tid    = static_cast<index_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const stride = static_cast<index_type>(blockDim.x * gridDim.x);
  for (auto i = tid; i < max_keys; i += stride) {
    if (i < count) {
      auto const key = static_cast<index_type>(unique_keys[i] >> 16);
      if (i == 0 or key != static_cast<index_type>(unique_keys[i - 1] >> 16)) {
        starts_by_key[key] = i;
      }
    }
  }
}

static __global__ void compact_container_index(index_type const* starts_by_key,
                                               index_type const* container_indices,
                                               cuda::std::uint16_t* container_keys,
                                               index_type* container_starts,
                                               index_type const* num_unique,
                                               index_type* num_containers)
{
  auto const tid    = static_cast<index_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const stride = static_cast<index_type>(blockDim.x * gridDim.x);
  for (auto key = tid; key < container_key_space; key += stride) {
    auto const start   = starts_by_key[key];
    auto const present = start != cuda::std::numeric_limits<index_type>::max();
    if (present) {
      auto const container        = container_indices[key];
      container_keys[container]   = static_cast<cuda::std::uint16_t>(key);
      container_starts[container] = start;
    }
    if (key == container_key_space - 1) {
      auto const count        = container_indices[key] + static_cast<index_type>(present);
      *num_containers         = count;
      container_starts[count] = *num_unique;
    }
  }
}
static __global__ void compute_container_sizes(index_type max_containers,
                                               index_type const* num_containers,
                                               index_type const* container_starts,
                                               index_type* container_sizes)
{
  auto const count  = *num_containers;
  auto const tid    = static_cast<index_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const stride = static_cast<index_type>(blockDim.x * gridDim.x);
  for (auto i = tid; i < max_containers; i += stride) {
    if (i < count) {
      auto const cardinality = container_starts[i + 1] - container_starts[i];
      container_sizes[i] =
        cardinality <= roaring_bitmap_metadata<cuda::std::uint32_t>::max_array_container_card
          ? cardinality * sizeof(cuda::std::uint16_t)
          : roaring_bitmap_metadata<cuda::std::uint32_t>::bitset_container_bytes;
    } else {
      container_sizes[i] = 0;
    }
  }
}

template <class T>
__device__ index_type
lower_bound(T const* keys, index_type first, index_type last, cuda::std::uint16_t value)
{
  while (first < last) {
    auto const mid = first + (last - first) / 2;
    if (static_cast<cuda::std::uint16_t>(keys[mid]) < value) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  return first;
}

template <class T>
__global__ void build_containers(T const* unique_keys,
                                 cuda::std::uint16_t const* container_keys,
                                 index_type const* container_starts,
                                 index_type const* container_offsets,
                                 index_type const* num_containers,
                                 cuda::std::byte* output)
{
  auto const count = *num_containers;
  for (auto container_index = static_cast<index_type>(blockIdx.x); container_index < count;
       container_index += static_cast<index_type>(gridDim.x)) {
    auto const start       = container_starts[container_index];
    auto const end         = container_starts[container_index + 1];
    auto const cardinality = end - start;
    auto const header_size =
      2 * sizeof(cuda::std::uint32_t) + 2 * sizeof(cuda::std::uint32_t) * count;
    auto* container = output + header_size + container_offsets[container_index];

    if (threadIdx.x == 0) {
      auto const key_card_offset =
        2 * sizeof(cuda::std::uint32_t) + 2 * sizeof(cuda::std::uint16_t) * container_index;
      auto const offset_offset = 2 * sizeof(cuda::std::uint32_t) +
                                 2 * sizeof(cuda::std::uint16_t) * count +
                                 sizeof(cuda::std::uint32_t) * container_index;
      auto const card_minus_one    = static_cast<cuda::std::uint16_t>(cardinality - 1);
      auto const serialized_offset = header_size + container_offsets[container_index];
      cuda::std::memcpy(
        output + key_card_offset, container_keys + container_index, sizeof(cuda::std::uint16_t));
      cuda::std::memcpy(output + key_card_offset + sizeof(cuda::std::uint16_t),
                        &card_minus_one,
                        sizeof(cuda::std::uint16_t));
      cuda::std::memcpy(output + offset_offset, &serialized_offset, sizeof(cuda::std::uint32_t));
    }

    if (cardinality <= roaring_bitmap_metadata<cuda::std::uint32_t>::max_array_container_card) {
      for (auto i = static_cast<index_type>(threadIdx.x); i < cardinality; i += blockDim.x) {
        auto const lower = static_cast<cuda::std::uint16_t>(unique_keys[start + i]);
        cuda::std::memcpy(
          container + i * sizeof(cuda::std::uint16_t), &lower, sizeof(cuda::std::uint16_t));
      }
    } else {
      constexpr index_type words =
        roaring_bitmap_metadata<cuda::std::uint32_t>::bitset_container_bytes /
        sizeof(cuda::std::uint64_t);
      for (auto word_index = static_cast<index_type>(threadIdx.x); word_index < words;
           word_index += blockDim.x) {
        auto const word_begin = static_cast<cuda::std::uint16_t>(word_index * 64);
        auto const word_end   = static_cast<cuda::std::uint32_t>(word_begin) + 64;
        auto item             = lower_bound(unique_keys, start, end, word_begin);
        cuda::std::uint64_t word{};
        while (item < end) {
          auto const lower = static_cast<cuda::std::uint16_t>(unique_keys[item]);
          if (static_cast<cuda::std::uint32_t>(lower) >= word_end) { break; }
          word |= cuda::std::uint64_t{1} << (lower & 63);
          ++item;
        }
        cuda::std::memcpy(
          container + word_index * sizeof(cuda::std::uint64_t), &word, sizeof(cuda::std::uint64_t));
      }
    }
    __syncthreads();
  }
}

static __global__ void finalize_bitmap(index_type const* num_unique,
                                       index_type const* num_containers,
                                       index_type const* container_sizes,
                                       index_type const* container_offsets,
                                       cuda::std::byte* output,
                                       roaring_bitmap_metadata<cuda::std::uint32_t>* metadata)
{
  if (blockIdx.x != 0 or threadIdx.x != 0) { return; }

  constexpr cuda::std::uint32_t serial_cookie_no_runcontainer = 12346;
  auto const count   = num_containers == nullptr ? 0 : *num_containers;
  auto const header  = 2 * sizeof(cuda::std::uint32_t) + 2 * sizeof(cuda::std::uint32_t) * count;
  auto const payload = count == 0 ? 0 : container_offsets[count - 1] + container_sizes[count - 1];
  auto const size_bytes = header + payload;

  cuda::std::memcpy(output, &serial_cookie_no_runcontainer, sizeof(cuda::std::uint32_t));
  cuda::std::memcpy(output + sizeof(cuda::std::uint32_t), &count, sizeof(cuda::std::uint32_t));

  metadata->size_bytes           = size_bytes;
  metadata->num_keys             = num_unique == nullptr ? 0 : *num_unique;
  metadata->run_container_bitmap = 0;
  metadata->key_cards            = 2 * sizeof(cuda::std::uint32_t);
  metadata->container_offsets =
    2 * sizeof(cuda::std::uint32_t) + 2 * sizeof(cuda::std::uint16_t) * count;
  metadata->num_containers             = static_cast<cuda::std::int32_t>(count);
  metadata->has_run                    = false;
  metadata->valid                      = true;
  metadata->offsets_in_serialized_data = true;
}

inline cuda::std::size_t max_containers(cuda::std::size_t num_inputs)
{
  return cuda::std::min<cuda::std::size_t>(num_inputs, cuda::std::size_t{1} << 16);
}

inline cuda::std::size_t max_serialized_bytes(cuda::std::size_t num_inputs)
{
  auto const containers = max_containers(num_inputs);
  CUCO_EXPECTS(
    num_inputs <= (cuda::std::numeric_limits<cuda::std::uint32_t>::max() - 8 - 8 * containers) / 2,
    "Serialized Roaring bitmap exceeds the portable format's 32-bit offset range");
  return 8 + 8 * containers + 2 * num_inputs;
}

}  // namespace roaring_bitmap_builder_ns

template <class Allocator>
class roaring_bitmap_builder<cuda::std::uint32_t, Allocator> {
  using namespace_type = roaring_bitmap_builder_ns::index_type;

 public:
  using value_type    = cuda::std::uint32_t;
  using metadata_type = roaring_bitmap_metadata<value_type>;
  using allocator_type =
    typename std::allocator_traits<Allocator>::template rebind_alloc<cuda::std::byte>;
  using data_pointer_type =
    roaring_bitmap_builder_ns::device_unique_ptr<cuda::std::byte, Allocator>;
  using key_pointer_type = roaring_bitmap_builder_ns::device_unique_ptr<value_type, Allocator>;
  using metadata_pointer_type =
    roaring_bitmap_builder_ns::device_unique_ptr<metadata_type, Allocator>;
  using host_metadata_pointer_type =
    std::unique_ptr<metadata_type, roaring_bitmap_builder_ns::pinned_deleter<metadata_type>>;

  template <class InputIt>
  roaring_bitmap_builder(value_type const* old_keys,
                         cuda::std::size_t old_count,
                         InputIt first,
                         InputIt last,
                         allocator_type const& alloc,
                         cuda::stream_ref stream)
    : allocator_{alloc},
      staged_count_{checked_count(old_count, first, last)},
      data_{roaring_bitmap_builder_ns::make_device_buffer<cuda::std::byte>(
        alloc, roaring_bitmap_builder_ns::max_serialized_bytes(staged_count_), stream)},
      staged_keys_{
        roaring_bitmap_builder_ns::make_device_buffer<value_type>(alloc, staged_count_, stream)},
      metadata_device_{
        roaring_bitmap_builder_ns::make_device_buffer<metadata_type>(alloc, 1, stream)},
      metadata_host_{roaring_bitmap_builder_ns::make_pinned<metadata_type>()}
  {
    build(nullptr, metadata_type{}, old_keys, old_count, first, alloc, stream);
  }

  template <class InputIt>
  roaring_bitmap_builder(cuda::std::byte const* old_bitmap,
                         metadata_type const& old_metadata,
                         InputIt first,
                         InputIt last,
                         allocator_type const& alloc,
                         cuda::stream_ref stream)
    : allocator_{alloc},
      staged_count_{checked_count(old_metadata.num_keys, first, last)},
      data_{roaring_bitmap_builder_ns::make_device_buffer<cuda::std::byte>(
        alloc, roaring_bitmap_builder_ns::max_serialized_bytes(staged_count_), stream)},
      staged_keys_{
        roaring_bitmap_builder_ns::make_device_buffer<value_type>(alloc, staged_count_, stream)},
      metadata_device_{
        roaring_bitmap_builder_ns::make_device_buffer<metadata_type>(alloc, 1, stream)},
      metadata_host_{roaring_bitmap_builder_ns::make_pinned<metadata_type>()}
  {
    CUCO_EXPECTS(old_bitmap != nullptr or old_metadata.num_keys == 0,
                 "Serialized Roaring bitmap data is null");
    build(old_bitmap, old_metadata, nullptr, old_metadata.num_keys, first, alloc, stream);
  }

  roaring_bitmap_builder(roaring_bitmap_builder&&)            = default;
  roaring_bitmap_builder& operator=(roaring_bitmap_builder&&) = default;

  [[nodiscard]] cuda::std::byte* data() const noexcept { return data_.get(); }
  [[nodiscard]] value_type const* staged_keys() const noexcept { return staged_keys_.get(); }
  [[nodiscard]] cuda::std::size_t staged_count() const noexcept { return staged_count_; }
  [[nodiscard]] metadata_type const& metadata() const noexcept { return *metadata_host_; }
  [[nodiscard]] metadata_type const* dynamic_metadata() const noexcept
  {
    return metadata_device_.get();
  }

  [[nodiscard]] allocator_type allocator() const noexcept { return allocator_; }

  [[nodiscard]] data_pointer_type take_data() noexcept { return cuda::std::move(data_); }
  [[nodiscard]] key_pointer_type take_staged_keys() noexcept
  {
    return cuda::std::move(staged_keys_);
  }
  [[nodiscard]] metadata_pointer_type take_dynamic_metadata() noexcept
  {
    return cuda::std::move(metadata_device_);
  }
  [[nodiscard]] host_metadata_pointer_type take_host_metadata() noexcept
  {
    return cuda::std::move(metadata_host_);
  }

 private:
  template <class InputIt>
  static cuda::std::size_t checked_count(cuda::std::size_t old_count, InputIt first, InputIt last)
  {
    auto const distance = cuda::std::distance(first, last);
    CUCO_EXPECTS(distance >= 0, "Roaring bitmap input range is invalid");
    auto const new_count = static_cast<cuda::std::size_t>(distance);
    CUCO_EXPECTS(new_count <= cuda::std::numeric_limits<namespace_type>::max() - old_count,
                 "Roaring bitmap construction supports at most 2^32 - 1 inputs");
    return old_count + new_count;
  }

  template <class InputIt>
  void build(cuda::std::byte const* old_bitmap,
             metadata_type const& old_metadata,
             value_type const* old_keys,
             cuda::std::size_t old_count,
             InputIt first,
             allocator_type const& alloc,
             cuda::stream_ref stream)
  {
    using namespace roaring_bitmap_builder_ns;
    auto const new_count = staged_count_ - old_count;
    auto const n         = static_cast<index_type>(staged_count_);

    auto raw_keys = make_device_buffer<value_type>(alloc, staged_count_, stream);
    if (old_count != 0 and old_keys == nullptr) {
      auto const num_old_containers = static_cast<index_type>(old_metadata.num_containers);
      CUCO_EXPECTS(old_bitmap != nullptr and num_old_containers != 0,
                   "Serialized Roaring bitmap metadata is inconsistent");

      auto old_container_starts = make_device_buffer<index_type>(alloc, num_old_containers, stream);
      auto container_index      = cuda::counting_iterator<index_type>{0};
      auto cardinalities        = cuda::make_transform_iterator(
        container_index, container_cardinality{old_bitmap, old_metadata});
      cuda::std::size_t scan_bytes{};
      CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                                  scan_bytes,
                                                  cardinalities,
                                                  old_container_starts.get(),
                                                  num_old_containers,
                                                  stream.get()));
      auto scan_temp = make_device_buffer<cuda::std::byte>(alloc, scan_bytes, stream);
      CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(scan_temp.get(),
                                                  scan_bytes,
                                                  cardinalities,
                                                  old_container_starts.get(),
                                                  num_old_containers,
                                                  stream.get()));

      constexpr int block_size = 256;
      auto const grid_size = static_cast<int>(cuda::std::min<index_type>(num_old_containers, 1024));
      decode_containers<<<grid_size, block_size, 0, stream.get()>>>(
        old_bitmap, old_metadata, old_container_starts.get(), raw_keys.get());
    }
    if (new_count != 0) {
      auto const new_offset = old_keys == nullptr ? old_count : 0;
      CUCO_CUDA_TRY(cub::DeviceTransform::Transform(
        first, raw_keys.get() + new_offset, new_count, to_value<value_type>{}, stream.get()));
    }

    if (staged_count_ == 0) {
      finalize_bitmap<<<1, 1, 0, stream.get()>>>(
        nullptr, nullptr, nullptr, nullptr, data_.get(), metadata_device_.get());
      CUCO_CUDA_TRY(cuco::detail::memcpy_async(metadata_host_.get(),
                                               metadata_device_.get(),
                                               sizeof(metadata_type),
                                               cudaMemcpyDeviceToHost,
                                               stream));
      return;
    }

    auto const container_capacity = max_containers(staged_count_);
    auto index_begin              = cuda::counting_iterator<index_type>{0};
    auto query_container_flags =
      cuda::make_transform_iterator(index_begin, container_present{nullptr});

    auto const incremental    = old_keys != nullptr and old_count != 0;
    auto const sort_count     = static_cast<index_type>(incremental ? new_count : staged_count_);
    auto* const sorted_output = staged_keys_.get() + (incremental ? old_count : 0);

    cuda::std::size_t sort_bytes{}, merge_bytes{}, unique_bytes{}, container_scan_bytes{},
      size_scan_bytes{};
    CUCO_CUDA_TRY(cub::DeviceRadixSort::SortKeys(nullptr,
                                                 sort_bytes,
                                                 raw_keys.get(),
                                                 sorted_output,
                                                 sort_count,
                                                 0,
                                                 sizeof(value_type) * 8,
                                                 stream.get()));
    if (incremental) {
      CUCO_CUDA_TRY(cub::DeviceMerge::MergeKeys(nullptr,
                                                merge_bytes,
                                                old_keys,
                                                old_count,
                                                sorted_output,
                                                new_count,
                                                raw_keys.get(),
                                                cuda::std::less<>{},
                                                stream.get()));
    }
    CUCO_CUDA_TRY(cub::DeviceSelect::Unique(nullptr,
                                            unique_bytes,
                                            staged_keys_.get(),
                                            raw_keys.get(),
                                            static_cast<index_type*>(nullptr),
                                            n,
                                            stream.get()));
    CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                                container_scan_bytes,
                                                query_container_flags,
                                                static_cast<index_type*>(nullptr),
                                                container_key_space,
                                                stream.get()));
    CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                                size_scan_bytes,
                                                static_cast<index_type*>(nullptr),
                                                static_cast<index_type*>(nullptr),
                                                container_capacity,
                                                stream.get()));

    auto const temp_bytes =
      cuda::std::max(cuda::std::max(cuda::std::max(sort_bytes, merge_bytes), unique_bytes),
                     cuda::std::max(container_scan_bytes, size_scan_bytes));
    cub::detail::temporary_storage::layout<8> temporary_storage;
    auto starts_by_key =
      temporary_storage.get_slot(0)->create_alias<index_type>(container_key_space);
    auto container_indices =
      temporary_storage.get_slot(1)->create_alias<index_type>(container_key_space);
    auto container_keys =
      temporary_storage.get_slot(2)->create_alias<cuda::std::uint16_t>(container_capacity);
    auto container_starts =
      temporary_storage.get_slot(3)->create_alias<index_type>(container_capacity + 1);
    auto container_sizes =
      temporary_storage.get_slot(4)->create_alias<index_type>(container_capacity);
    auto container_offsets =
      temporary_storage.get_slot(5)->create_alias<index_type>(container_capacity);
    auto counters = temporary_storage.get_slot(6)->create_alias<index_type>(2);
    auto temp     = temporary_storage.get_slot(7)->create_alias<cuda::std::byte>(temp_bytes);

    auto const arena_bytes = temporary_storage.get_size();
    auto arena             = make_device_buffer<cuda::std::byte>(alloc, arena_bytes, stream);
    CUCO_CUDA_TRY(temporary_storage.map_to_buffer(arena.get(), arena_bytes));
    auto* num_unique     = counters.get();
    auto* num_containers = counters.get() + 1;
    CUCO_CUDA_TRY(cudaMemsetAsync(
      starts_by_key.get(), 0xff, container_key_space * sizeof(index_type), stream.get()));

    CUCO_CUDA_TRY(cub::DeviceRadixSort::SortKeys(temp.get(),
                                                 sort_bytes,
                                                 raw_keys.get(),
                                                 sorted_output,
                                                 sort_count,
                                                 0,
                                                 sizeof(value_type) * 8,
                                                 stream.get()));
    if (incremental) {
      CUCO_CUDA_TRY(cub::DeviceMerge::MergeKeys(temp.get(),
                                                merge_bytes,
                                                old_keys,
                                                old_count,
                                                sorted_output,
                                                new_count,
                                                raw_keys.get(),
                                                cuda::std::less<>{},
                                                stream.get()));
      CUCO_CUDA_TRY(cuco::detail::memcpy_async(staged_keys_.get(),
                                               raw_keys.get(),
                                               staged_count_ * sizeof(value_type),
                                               cudaMemcpyDeviceToDevice,
                                               stream));
    }
    CUCO_CUDA_TRY(cub::DeviceSelect::Unique(
      temp.get(), unique_bytes, staged_keys_.get(), raw_keys.get(), num_unique, n, stream.get()));

    constexpr int block_size = 256;
    auto const grid_size     = static_cast<int>(
      cuda::std::min<cuda::std::size_t>((staged_count_ + block_size - 1) / block_size, 4096));
    mark_container_starts<<<grid_size, block_size, 0, stream.get()>>>(
      raw_keys.get(), n, num_unique, starts_by_key.get());

    auto container_flags =
      cuda::make_transform_iterator(index_begin, container_present{starts_by_key.get()});
    CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(temp.get(),
                                                container_scan_bytes,
                                                container_flags,
                                                container_indices.get(),
                                                container_key_space,
                                                stream.get()));
    compact_container_index<<<256, block_size, 0, stream.get()>>>(starts_by_key.get(),
                                                                  container_indices.get(),
                                                                  container_keys.get(),
                                                                  container_starts.get(),
                                                                  num_unique,
                                                                  num_containers);

    compute_container_sizes<<<256, block_size, 0, stream.get()>>>(
      container_capacity, num_containers, container_starts.get(), container_sizes.get());
    CUCO_CUDA_TRY(cub::DeviceScan::ExclusiveSum(temp.get(),
                                                size_scan_bytes,
                                                container_sizes.get(),
                                                container_offsets.get(),
                                                container_capacity,
                                                stream.get()));

    auto const container_grid =
      static_cast<int>(cuda::std::min<cuda::std::size_t>(container_capacity, 1024));
    build_containers<<<container_grid, block_size, 0, stream.get()>>>(raw_keys.get(),
                                                                      container_keys.get(),
                                                                      container_starts.get(),
                                                                      container_offsets.get(),
                                                                      num_containers,
                                                                      data_.get());
    finalize_bitmap<<<1, 1, 0, stream.get()>>>(num_unique,
                                               num_containers,
                                               container_sizes.get(),
                                               container_offsets.get(),
                                               data_.get(),
                                               metadata_device_.get());
    CUCO_CUDA_TRY(cuco::detail::memcpy_async(metadata_host_.get(),
                                             metadata_device_.get(),
                                             sizeof(metadata_type),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  }

  allocator_type allocator_;
  cuda::std::size_t staged_count_;
  data_pointer_type data_;
  key_pointer_type staged_keys_;
  metadata_pointer_type metadata_device_;
  host_metadata_pointer_type metadata_host_;
};

}  // namespace cuco::experimental::detail
