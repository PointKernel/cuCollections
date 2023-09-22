/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 */

#pragma once

#include <thrust/device_reference.h>
#include <thrust/tuple.h>

#include <cuda/std/type_traits>

#include <tuple>

namespace cuco::detail {

template <typename T, typename = void>
struct is_std_pair_like : cuda::std::false_type {
};

template <typename T>
struct is_std_pair_like<T,
                        cuda::std::void_t<decltype(std::get<0>(cuda::std::declval<T>())),
                                          decltype(std::get<1>(cuda::std::declval<T>()))>>
  : cuda::std::
      conditional_t<std::tuple_size<T>::value == 2, cuda::std::true_type, cuda::std::false_type> {
};

template <typename T, typename = void>
struct is_thrust_pair_like_impl : cuda::std::false_type {
};

template <typename T>
struct is_thrust_pair_like_impl<
  T,
  cuda::std::void_t<decltype(thrust::get<0>(cuda::std::declval<T>())),
                    decltype(thrust::get<1>(cuda::std::declval<T>()))>>
  : cuda::std::conditional_t<thrust::tuple_size<T>::value == 2,
                             cuda::std::true_type,
                             cuda::std::false_type> {
};

template <typename T>
struct is_thrust_pair_like
  : is_thrust_pair_like_impl<cuda::std::remove_reference_t<decltype(thrust::raw_reference_cast(
      cuda::std::declval<T>()))>> {
};

template <typename T, typename = void>
struct is_extent : cuda::std::false_type {
};

template <typename T>
struct is_extent<
  T,
  cuda::std::void_t<
    typename T::value_type,
    cuda::std::enable_if_t<cuda::std::is_convertible_v<typename T::value_type, std::size_t> and
                           cuda::std::is_convertible_v<T, std::size_t> and
                           cuda::std::is_constructible_v<T, std::size_t>>>> : cuda::std::true_type {
};

template <typename T>
inline constexpr bool is_extent_v = is_extent<T>::value;

}  // namespace cuco::detail
