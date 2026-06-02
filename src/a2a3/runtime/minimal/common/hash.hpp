#pragma once

/**
 * @file hash.hpp
 * @brief Contains common function related to hashing
 */

#include "third_party/metrohash/metrohash64.h"
#include <stdio.h>
#include <string>

namespace simpler
{

namespace hash
{

/**
 * Definition for the standard 64-bit hash value in Jaffar.
 */
typedef uint64_t hash_t;

/**
 * Calculates the 64 bit Metrohash of a given buffer
 *
 * @param[in] data The input buffer to hash
 * @param[in] size The size of the buffer to hash
 * @return The calculated 64-bit metro hash
 */
inline hash_t calculateMetroHash(const void* data, size_t size)
{
  MetroHash64 hash;
  hash.Update((const uint8_t*)data, size);
  hash_t result;
  hash.Finalize(reinterpret_cast<uint8_t*>(&result));
  return result;
}

/**
 * Produces an output string given a 64-bit hash
 *
 * @param[in] hash The hash value to stringify
 * @return The string containing the stringified hash, formatted as upper case hexadecimal
 */
inline std::string hashToString(const hash_t hash)
{
  // Creating hash string
  char hashStringBuffer[256];
  sprintf(hashStringBuffer, "0x%016lX", hash);
  return std::string(hashStringBuffer);
}

/**
 * Calculates the 64-bit metro hash of a given string
 *
 * @param[in] string The string to calculate the hash for
 * @return The hash value of the provided string
 */
inline hash_t hashString(const std::string& string) { return calculateMetroHash(string.data(), string.size()); }

} // namespace hash

} // namespace simpler