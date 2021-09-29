/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
 * limitations under the License.
 */

#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#include <catch2/catch.hpp>
#include <cuco/dynamic_map.cuh>

#include <util.hpp>

TEMPLATE_TEST_CASE_SIG("Unique sequence of keys",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int32_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t num_keys{50'000'000};
  cuco::dynamic_map<Key, Value> map{30'000'000, -1, -1};

  thrust::device_vector<Key> d_keys(num_keys);
  thrust::device_vector<Value> d_values(num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(num_keys);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  thrust::sequence(thrust::device, d_values.begin(), d_values.end());
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_keys),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair_type<Key, Value>{i, i};
                    });

  thrust::device_vector<Value> d_results(num_keys);
  thrust::device_vector<bool> d_contained(num_keys);

  // bulk function test cases
  SECTION("All inserted keys-value pairs should be correctly recovered during find")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_results.begin(), d_values.begin()));

    REQUIRE(all_of(zip, zip + num_keys, [] __device__(auto const& p) {
      return thrust::get<0>(p) == thrust::get<1>(p);
    }));
  }

  SECTION("All non-inserted keys-value pairs should have the empty sentinel value recovered")
  {
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());

    REQUIRE(
      all_of(d_results.begin(), d_results.end(), [] __device__(auto const& p) { return p == -1; }));
  }

  SECTION("All inserted keys-value pairs should be contained")
  {
    map.insert(d_pairs.begin(), d_pairs.end());
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(
      all_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }

  SECTION("Non-inserted keys-value pairs should not be contained")
  {
    map.contains(d_keys.begin(), d_keys.end(), d_contained.begin());

    REQUIRE(
      none_of(d_contained.begin(), d_contained.end(), [] __device__(bool const& b) { return b; }));
  }
}
