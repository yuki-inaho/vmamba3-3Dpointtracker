#pragma once
// Minimal, strict NPY v1/v2 float32 I/O. No pickle, Python, CUDA or custom ops.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace refiner {
struct Array {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

inline size_t elements(const std::vector<int64_t>& shape) {
  require(shape.size() <= 6, "Unsupported tensor rank");
  size_t count = 1;
  constexpr size_t limit = size_t{1} << 29;  // A hard 2 GiB bound per input.
  for (int64_t dimension : shape) {
    require(dimension > 0 && static_cast<uint64_t>(dimension) <= limit,
            "Invalid or oversized tensor dimension");
    require(count <= limit / static_cast<size_t>(dimension), "Oversized tensor");
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

inline void read_exact(std::istream& stream, char* output, size_t count) {
  stream.read(output, static_cast<std::streamsize>(count));
  require(static_cast<size_t>(stream.gcount()) == count, "Truncated NPY file");
}

inline Array load_npy(const std::filesystem::path& path) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "IEEE754 float32 is required");
  const uint16_t endian = 1;
  require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "Little-endian host required");
  std::ifstream stream(path, std::ios::binary);
  require(stream.good(), "Cannot read input: " + path.filename().string());
  char prefix[8];
  read_exact(stream, prefix, sizeof(prefix));
  require(std::string(prefix, 6) == std::string("\x93NUMPY", 6), "Not an NPY file");
  const auto version = static_cast<uint8_t>(prefix[6]);
  require((version == 1 || version == 2) && prefix[7] == 0, "Only NPY v1/v2 supported");
  const size_t length_bytes = version == 1 ? 2 : 4;
  char bytes[4]{};
  read_exact(stream, bytes, length_bytes);
  uint32_t header_size = 0;
  for (size_t i = 0; i < length_bytes; ++i)
    header_size |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[i])) << (8 * i);
  require(header_size > 0 && header_size <= 65536, "Invalid NPY header size");
  std::string header(header_size, '\0');
  read_exact(stream, header.data(), header.size());
  std::smatch match;
  require(std::regex_search(header, match, std::regex("['\"]descr['\"]\\s*:\\s*['\"](<f4|=f4)['\"]")),
          "Only little-endian float32 NPY tensors are supported");
  require(std::regex_search(header, std::regex("['\"]fortran_order['\"]\\s*:\\s*False")),
          "Fortran-order arrays are not supported");
  require(std::regex_search(header, match, std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)")),
          "Missing NPY shape");
  std::string shape_text = match[1].str();
  require(std::regex_match(shape_text, std::regex("\\s*([0-9]+\\s*(,\\s*[0-9]+\\s*)*,?\\s*)?")),
          "Malformed NPY shape");
  Array value;
  const std::regex integer("[0-9]+");
  for (auto it = std::sregex_iterator(shape_text.begin(), shape_text.end(), integer);
       it != std::sregex_iterator(); ++it)
    value.shape.push_back(std::stoll(it->str()));
  const size_t count = elements(value.shape);
  const auto position = static_cast<uintmax_t>(stream.tellg());
  require(std::filesystem::file_size(path) == position + count * sizeof(float),
          "NPY payload length does not match shape");
  value.data.resize(count);
  read_exact(stream, reinterpret_cast<char*>(value.data.data()), count * sizeof(float));
  require(std::all_of(value.data.begin(), value.data.end(), [](float x) { return std::isfinite(x); }),
          "Nonfinite input: " + path.filename().string());
  return value;
}

inline void save_npy(const std::filesystem::path& path, const Array& value) {
  require(value.data.size() == elements(value.shape), "Output shape/storage mismatch");
  std::string shape = "(";
  for (auto dim : value.shape) shape += std::to_string(dim) + ", ";
  shape += ")";
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }";
  const size_t padding = (64 - (10 + header.size() + 1) % 64) % 64;
  header += std::string(padding, ' ') + "\n";
  require(header.size() < 65536, "Output header too large");
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  require(stream.good(), "Cannot open output: " + path.string());
  stream.write("\x93NUMPY\x01\x00", 8);
  const char size[2] = {static_cast<char>(header.size() & 255),
                        static_cast<char>((header.size() >> 8) & 255)};
  stream.write(size, 2);
  stream.write(header.data(), static_cast<std::streamsize>(header.size()));
  stream.write(reinterpret_cast<const char*>(value.data.data()),
               static_cast<std::streamsize>(value.data.size() * sizeof(float)));
  stream.close();
  require(!stream.fail(), "Failed to write NPY output");
}
}  // namespace refiner
