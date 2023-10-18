#pragma once

#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "fstream"

#define CHECK(call)                                                   \
  do {                                                                \
    const cudaError_t error_code = call;                              \
    if (error_code != cudaSuccess) {                                  \
      printf("CUDA Error:\n");                                        \
      printf("    File:       %s\n", __FILE__);                       \
      printf("    Line:       %d\n", __LINE__);                       \
      printf("    Error code: %d\n", error_code);                     \
      printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

class Logger : public nvinfer1::ILogger {
 public:
  nvinfer1::ILogger::Severity reportableSeverity;

  explicit Logger(
      nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
      : reportableSeverity(severity) {}

  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override {
    if (severity > reportableSeverity) {
      return;
    }
    switch (severity) {
      case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case nvinfer1::ILogger::Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case nvinfer1::ILogger::Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case nvinfer1::ILogger::Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "VERBOSE: ";
        break;
    }
    std::cerr << msg << std::endl;
  }
};

struct EngineBinding {
  int index;

  std::string name;

  nvinfer1::Dims dims;

  nvinfer1::DataType type;

  int volume; /*  note: calculated based on dims  */
};

struct Detection {
  int id;

  std::string className;

  cv::Rect boudingBox;

  float score;
};

class Processor {
 public:
  Processor(std::string model_path, std::string class_file);

  cv::Mat process(const cv::Mat& src);

  std::vector<Detection> detect(const cv::Mat& src);

  ~Processor();

 private:
  nvinfer1::IRuntime* _runtime{nullptr};

  nvinfer1::ICudaEngine* _engine{nullptr};

  nvinfer1::IExecutionContext* _context{nullptr};

  cudaStream_t _stream{nullptr};

  std::vector<std::string> _classes;

  bool load_classes(std::string class_file);

  int32_t dimsVolume(const nvinfer1::Dims& dims) noexcept;

  std::string dimsToString(const nvinfer1::Dims& dims);

  cv::Mat pre_process(const cv::Mat& img);

  void infer(cv::Mat& m);

  std::vector<Detection> post_process(const cv::Mat& src);

  void draw_detection(cv::Mat& src, const std::vector<Detection>&);

  EngineBinding _inputBinding;
  EngineBinding _outputBinding;

  std::vector<void*> _device_memory;

  void* _host_output_memory;

  void* _host_input_memory;

  Logger _logger;

  float _scoreThreshold = 0.8;

  float _nmsThreshold = 0.4;

  std::vector<cv::Scalar> _colors;
};
