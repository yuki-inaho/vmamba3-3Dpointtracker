#include "npy.hpp"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <chrono>
#include <iostream>
#include <memory>

namespace {
using refiner::Array;
using refiner::require;
constexpr std::array<const char*, 8> kInputs = {
    "ray", "z_raw", "visibility", "uv", "depth_map", "dino_features", "intrinsics", "z_ref"};
constexpr std::array<const char*, 4> kOutputs = {"xyz", "uv_refined", "vis_logits", "delta_uv"};
constexpr std::array<int64_t, 4> kChannels = {3, 2, 1, 2};

std::array<int64_t, 3> validate(const std::array<Array, 8>& inputs) {
  const auto& ray = inputs[0];
  require(ray.shape.size() == 4 && ray.shape[3] == 2, "ray must be B,F,N,2");
  const int64_t b = ray.shape[0], f = ray.shape[1], n = ray.shape[2];
  const std::array<std::vector<int64_t>, 8> shapes = {
      ray.shape, std::vector<int64_t>{b, f, n}, {b, f, n}, {b, f, n, 2},
      inputs[4].shape, {b, f, 384, 28, 28}, {b, 3, 3}, {}};
  for (size_t i = 0; i < inputs.size(); ++i)
    require(inputs[i].shape == shapes[i], std::string("Invalid shape: ") + kInputs[i]);
  const auto& depth = inputs[4];
  require(depth.shape.size() == 4 && depth.shape[0] == b && depth.shape[1] == f,
          "depth_map must be B,F,Hd,Wd");
  for (const size_t i : {size_t{1}, size_t{4}})
    require(std::all_of(inputs[i].data.begin(), inputs[i].data.end(), [](float x) { return x >= 0; }),
            "Depth must be nonnegative");
  require(std::all_of(inputs[2].data.begin(), inputs[2].data.end(),
                      [](float x) { return x == 0 || x == 1; }), "visibility must be binary");
  const auto& k = inputs[6].data;
  for (int64_t i = 0; i < b; ++i) {
    const auto p = static_cast<size_t>(i * 9);
    require(k[p] > 0 && k[p+4] > 0 && k[p+1] == 0 && k[p+3] == 0 &&
                k[p+6] == 0 && k[p+7] == 0 && k[p+8] == 1,
            "Only positive-focal, unskewed pinhole intrinsics supported");
  }
  // Compute ONCE from the full point set, before any track splitting.
  auto z = inputs[1].data;
  const size_t middle = (z.size() - 1) / 2;
  std::nth_element(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(middle), z.end());
  const float expected = z[middle] + 1e-6f;
  require(inputs[7].data[0] > 0 && inputs[7].data[0] == expected,
          "z_ref must equal full-input lower median plus 1e-6");
  return {b, f, n};
}

Array slice_tracks(const Array& source, int64_t begin, int64_t count) {
  Array output;
  output.shape = source.shape;
  const int64_t total = source.shape[2];
  output.shape[2] = count;
  const int64_t channels = source.shape.size() == 4 ? source.shape[3] : 1;
  output.data.resize(refiner::elements(output.shape));
  for (int64_t bf = 0; bf < source.shape[0] * source.shape[1]; ++bf)
    std::copy_n(source.data.begin() + (bf * total + begin) * channels, count * channels,
                output.data.begin() + bf * count * channels);
  return output;
}

void check_schema(const Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  require(session.GetInputCount() == kInputs.size() && session.GetOutputCount() == kOutputs.size(),
          "Unexpected ONNX schema");
  for (size_t i = 0; i < kInputs.size(); ++i) {
    auto name = session.GetInputNameAllocated(i, allocator);
    require(std::string(name.get()) == kInputs[i], "Unexpected ONNX input name");
    require(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType() ==
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "ONNX inputs must be float32");
  }
  for (size_t i = 0; i < kOutputs.size(); ++i) {
    auto name = session.GetOutputNameAllocated(i, allocator);
    require(std::string(name.get()) == kOutputs[i], "Unexpected ONNX output name");
    require(session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType() ==
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "ONNX outputs must be float32");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc >= 4 && argc <= 7,
            "Usage: refiner_runner MODEL.onnx INPUT_DIRECTORY OUTPUT_DIRECTORY [TRACK_CHUNK=32] [THREADS=4] [WORKSPACE_MIB=1024]");
    const auto parse_positive = [](const char* text) {
      size_t used = 0;
      const int value = std::stoi(text, &used);
      require(used == std::string(text).size() && value > 0, "Arguments must be positive integers");
      return value;
    };
    const int chunk = argc > 4 ? parse_positive(argv[4]) : 32;
    const int threads = argc > 5 ? parse_positive(argv[5]) : 4;
    const int budget = argc > 6 ? parse_positive(argv[6]) : 1024;
    const std::filesystem::path input_dir(argv[2]), output_dir(argv[3]);
    require(std::filesystem::absolute(input_dir).lexically_normal() !=
                std::filesystem::absolute(output_dir).lexically_normal(),
            "Input and output directories must differ");
    std::filesystem::create_directories(output_dir);
    std::filesystem::remove(output_dir / "SUCCESS.json");
    std::array<Array, 8> inputs;
    for (size_t i = 0; i < kInputs.size(); ++i)
      inputs[i] = refiner::load_npy(input_dir / (std::string(kInputs[i]) + ".npy"));
    const auto [b, f, n] = validate(inputs);
    const long double estimate = static_cast<long double>(b) * std::min<int64_t>(n, chunk) * 24 * f * f * 4 * 8;
    require(estimate <= static_cast<long double>(budget) * 1024 * 1024,
            "Attention workspace exceeds budget; reduce track chunk, not frames");
    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "mamba3-refiner-cpu");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetInterOpNumThreads(1);
    // No custom operators and no additional provider: built-in CPU EP only.
    Ort::Session session(environment, argv[1], options);
    check_schema(session);
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<Array, 4> result;
    for (size_t i = 0; i < result.size(); ++i) {
      result[i].shape = {b, f, n};
      if (i != 2) result[i].shape.push_back(kChannels[i]);
      result[i].data.resize(refiner::elements(result[i].shape));
    }
    const auto started = std::chrono::steady_clock::now();
    for (int64_t start = 0; start < n; start += chunk) {
      const int64_t count = std::min<int64_t>(chunk, n - start);
      std::array<Array, 4> sliced;
      std::vector<Ort::Value> values;
      values.reserve(kInputs.size());
      for (size_t i = 0; i < kInputs.size(); ++i) {
        Array* value = &inputs[i];
        if (i < 4) {
          sliced[i] = slice_tracks(inputs[i], start, count);
          value = &sliced[i];
        }
        values.push_back(Ort::Value::CreateTensor<float>(memory, value->data.data(), value->data.size(),
                                                        value->shape.data(), value->shape.size()));
      }
      auto outputs = session.Run(Ort::RunOptions{nullptr}, kInputs.data(), values.data(), values.size(),
                                 kOutputs.data(), kOutputs.size());
      for (size_t i = 0; i < result.size(); ++i) {
        require(outputs[i].IsTensor(), "Output is not a tensor");
        const auto info = outputs[i].GetTensorTypeAndShapeInfo();
        auto expected = result[i].shape;
        expected[2] = count;
        require(info.GetShape() == expected && info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Invalid output shape/type");
        const float* data = outputs[i].GetTensorData<float>();
        const size_t size = info.GetElementCount();
        require(std::all_of(data, data + size, [](float x) { return std::isfinite(x); }), "Nonfinite ONNX output");
        const int64_t c = kChannels[i];
        for (int64_t bf = 0; bf < b * f; ++bf)
          std::copy_n(data + bf * count * c, count * c,
                      result[i].data.begin() + (bf * n + start) * c);
      }
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    for (size_t i = 0; i < result.size(); ++i)
      refiner::save_npy(output_dir / (std::string(kOutputs[i]) + ".npy"), result[i]);
    const std::string report = "{\"success\":true,\"provider\":\"CPUExecutionProvider\",\"runtime\":\"" +
        std::string(OrtGetApiBase()->GetVersionString()) + "\",\"batch\":" + std::to_string(b) +
        ",\"frames\":" + std::to_string(f) + ",\"tracks\":" + std::to_string(n) +
        ",\"track_chunk\":" + std::to_string(chunk) + ",\"inference_seconds\":" + std::to_string(elapsed) + "}";
    std::ofstream marker(output_dir / "SUCCESS.json");
    marker << report << '\n';
    marker.close();
    require(!marker.fail(), "Failed to write success marker");
    std::cout << report << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "refiner_runner: " << error.what() << '\n';
    return 1;
  }
}
