/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream_ref>

namespace cuco::experimental {

template <class T, class Allocator>
roaring_bitmap<T, Allocator>::roaring_bitmap(cuda::std::span<cuda::std::byte const> bitmap,
                                             Allocator const& alloc,
                                             cuda::stream_ref stream)
  : storage_{bitmap, alloc, stream}
{
}

template <class T, class Allocator>
roaring_bitmap<T, Allocator>::roaring_bitmap(cuda::std::byte const* bitmap,
                                             Allocator const& alloc,
                                             cuda::stream_ref stream)
  : storage_{bitmap, alloc, stream}
{
}

template <class T, class Allocator>
template <class InputIt,
          class U,
          cuda::std::enable_if_t<cuda::std::is_same_v<U, cuda::std::uint32_t>, int>>
roaring_bitmap<T, Allocator>::roaring_bitmap(InputIt first,
                                             InputIt last,
                                             Allocator const& alloc,
                                             cuda::stream_ref stream)
  : storage_{detail::roaring_bitmap_builder<T, Allocator>{
      nullptr, 0, first, last, allocator_type{alloc}, stream}}
{
  stream.sync();
}

template <class T, class Allocator>
template <class InputIt,
          class U,
          cuda::std::enable_if_t<cuda::std::is_same_v<U, cuda::std::uint32_t>, int>>
void roaring_bitmap<T, Allocator>::add(InputIt first, InputIt last, cuda::stream_ref stream)
{
  this->add_async(first, last, stream);
  stream.sync();
}

template <class T, class Allocator>
template <class InputIt,
          class U,
          cuda::std::enable_if_t<cuda::std::is_same_v<U, cuda::std::uint32_t>, int>>
void roaring_bitmap<T, Allocator>::add_async(InputIt first, InputIt last, cuda::stream_ref stream)
{
  if (first == last) { return; }
  auto const* staged_keys = storage_.staged_keys();
  auto const old_ref      = storage_.ref();
  auto builder =
    staged_keys != nullptr
      ? detail::roaring_bitmap_builder<T, Allocator>{staged_keys,
                                                     storage_.staged_count(),
                                                     first,
                                                     last,
                                                     storage_.allocator(),
                                                     stream}
      : detail::roaring_bitmap_builder<T, Allocator>{
          old_ref.data(), old_ref.metadata(), first, last, storage_.allocator(), stream};
  storage_.release_on(stream);
  storage_ = storage_type{cuda::std::move(builder)};
}

template <class T, class Allocator>
template <class InputIt, class OutputIt>
void roaring_bitmap<T, Allocator>::contains(InputIt first,
                                            InputIt last,
                                            OutputIt output,
                                            cuda::stream_ref stream) const
{
  ref_type{storage_.ref()}.contains(first, last, output, stream);
}

template <class T, class Allocator>
template <class InputIt, class OutputIt>
void roaring_bitmap<T, Allocator>::contains_async(InputIt first,
                                                  InputIt last,
                                                  OutputIt output,
                                                  cuda::stream_ref stream) const noexcept
{
  ref_type{storage_.ref()}.contains_async(first, last, output, stream);
}

template <class T, class Allocator>
cuda::std::size_t roaring_bitmap<T, Allocator>::size() const noexcept
{
  return ref_type{storage_.ref()}.size();
}

template <class T, class Allocator>
bool roaring_bitmap<T, Allocator>::empty() const noexcept
{
  return ref_type{storage_.ref()}.empty();
}

template <class T, class Allocator>
cuda::std::byte const* roaring_bitmap<T, Allocator>::data() const noexcept
{
  return ref_type{storage_.ref()}.data();
}

template <class T, class Allocator>
cuda::std::size_t roaring_bitmap<T, Allocator>::size_bytes() const noexcept
{
  return ref_type{storage_.ref()}.size_bytes();
}

template <class T, class Allocator>
typename roaring_bitmap<T, Allocator>::allocator_type roaring_bitmap<T, Allocator>::allocator()
  const noexcept
{
  return storage_.allocator();
}

template <class T, class Allocator>
typename roaring_bitmap<T, Allocator>::ref_type roaring_bitmap<T, Allocator>::ref() const noexcept
{
  return ref_type{storage_.ref()};
}
}  // namespace cuco::experimental