#include "Processor.h"

#include <stdexcept>

Processor::Processor(std::string engine_file_path, std::string class_file) {
  cudaSetDevice(0);
  FILE* file = fopen(engine_file_path.c_str(), "rb");
  if (!file) {
    throw std::runtime_error("the engine file not good");
  }
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);

  char* trtModelStream = new char[size];
  if (!trtModelStream) {
    throw std::runtime_error("failed to alloc the memory");
  }
  if (fread(trtModelStream, 1, size, file)!= size) {
    throw std::runtime_error("fread error");
  }
  fclose(file);
  if (!load_classes(class_file)) {
    throw std::runtime_error("failed to load the class");
  }
  _colors.resize(_classes.size());

  cv::RNG rng;
  for (unsigned int i = 0; i < _classes.size(); ++i) {
    _colors[i] = cv::Scalar(rng.uniform(25, 225), rng.uniform(25, 225),
                            rng.uniform(25, 225));
  }

  _runtime = nvinfer1::createInferRuntime(_logger);
  if (!_runtime) {
    throw std::runtime_error("failed to create infer runtime");
  }

  _engine = _runtime->deserializeCudaEngine(trtModelStream, size);
  if (!_engine) {
    throw std::runtime_error("failed to deserialize cuda engine");
  }
  delete[] trtModelStream;

  _context = _engine->createExecutionContext();
  if (!_context) {
    throw std::runtime_error("failed to create executionContext");
  }
  auto num_bindings = _engine->getNbBindings();
  _device_memory.resize(num_bindings);
  for (int i = 0; i < num_bindings; ++i) {
    std::string name = _engine->getBindingName(i);
    nvinfer1::Dims dims = _engine->getBindingDimensions(i);
    auto type = _engine->getTensorDataType(name.c_str());
    const int volume = dimsVolume(dims);
    bool IsInput = _engine->bindingIsInput(i);
    if (IsInput) {
      _inputBinding.name = name;
      _inputBinding.index = i;
      _inputBinding.dims = dims;
      _inputBinding.type = type;
      _inputBinding.volume = dimsVolume(dims);
      void** ptr = &(_device_memory[i]);
      if (_inputBinding.type == nvinfer1::DataType::kHALF) {
        CHECK(cudaMalloc(ptr, volume * sizeof(int16_t)));
      } else {
        CHECK(cudaMalloc(ptr, volume * sizeof(float)));
      }
    } else {
      _outputBinding.name = name;
      _outputBinding.index = i;
      _outputBinding.dims = dims;
      _outputBinding.volume = dimsVolume(dims);
      _outputBinding.type = type;
      void** ptr = &(_device_memory[i]);
      if (_inputBinding.type == nvinfer1::DataType::kHALF) {
        CHECK(cudaMalloc(ptr, volume * sizeof(short)));
      } else {
        CHECK(cudaMalloc(ptr, volume * sizeof(float)));
      }
      if (static_cast<int>(_classes.size()) != _outputBinding.dims.d[2] - 5)
        throw std::runtime_error(
            "the classes number not equal to the outputsize");
    }
  }
  CHECK(cudaStreamCreate(&_stream));
}

cv::Mat Processor::process(const cv::Mat& src) {
  cv::Mat input = pre_process(src);
  infer(input);
  auto result = post_process(src);
  auto dst = src.clone();
  draw_detection(dst, result);
  return dst;
}

std::vector<Detection> Processor::detect(const cv::Mat& src) {
  cv::Mat input = pre_process(src);
  infer(input);
  auto result = post_process(src);
  return result;
}

Processor::~Processor() {
  if (_context) _context->destroy();
  if (_engine) _engine->destroy();
  if (_runtime) _runtime->destroy();
  if (_stream) cudaStreamDestroy(_stream);
  for (auto& ptr : _device_memory) {
    CHECK(cudaFree(ptr));
  }
}

bool Processor::load_classes(std::string class_file) {
  _classes.clear();

  std::ifstream fs(class_file);

  if (!fs.is_open()) return false;

  std::string line;
  while (std::getline(fs, line)) {
    _classes.push_back(line);
  }

  return true;
}

int32_t Processor::dimsVolume(const nvinfer1::Dims& dims) noexcept {
  int32_t r = 0;
  if (dims.nbDims > 0) {
    r = 1;
  }

  for (int32_t i = 0; i < dims.nbDims; ++i) {
    r = r * dims.d[i];
  }
  return r;
}

std::string Processor::dimsToString(const nvinfer1::Dims& dims) {
  std::string ret;

  ret = "(";
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    ret += std::to_string(dims.d[i]);
    if (i < dims.nbDims - 1) {
      ret += ",";
    }
  }
  ret += ")";
  return ret;
}

cv::Mat Processor::pre_process(const cv::Mat& input) {
  const int networkRows = _inputBinding.dims.d[2];

  const int networkCols = _inputBinding.dims.d[3];

  const double f = MIN((double)networkRows / (double)input.rows,
                       (double)networkCols / (double)input.cols);
  const cv::Size boxSize = cv::Size(input.cols * f, input.rows * f);

  const int dr = networkRows - boxSize.height;
  const int dc = networkCols - boxSize.width;
  const int topHeight = std::floor(dr / 2.0);
  const int bottomHeight = std::ceil(dr / 2.0);
  const int leftWidth = std::floor(dc / 2.0);
  const int rightWidth = std::ceil(dc / 2.0);

  cv::Mat out;
  cv::resize(input, out, boxSize, 0, 0, cv::INTER_LINEAR);
  cv::copyMakeBorder(out, out, topHeight, bottomHeight, leftWidth, rightWidth,
                     cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  out.convertTo(out, CV_32FC3, 1 / 255.0f);
  return out;
}

void Processor::infer(cv::Mat& m) {
  assert(_inputBinding.volume * sizeof(float) ==
         m.size().area() * m.channels() * sizeof(float));

  if (_inputBinding.type == nvinfer1::DataType::kHALF) {
    CHECK(cudaMallocHost(&_host_input_memory,
                         sizeof(short) * _inputBinding.volume));

    cv::convertFp16(m, m);

    cv::Size size = cv::Size(m.cols, m.rows);
    int step = size.area();

    std::vector<cv::Mat> channels;

    channels.push_back(
        cv::Mat(size, CV_16S, (short*)_host_input_memory + 2 * step));
    channels.push_back(
        cv::Mat(size, CV_16S, (short*)_host_input_memory + 1 * step));
    channels.push_back(cv::Mat(size, CV_16S, (short*)_host_input_memory));

    cv::split(m, channels);

    CHECK(cudaMemcpyAsync(_device_memory[0], _host_input_memory,
                          _inputBinding.volume * sizeof(short),
                          cudaMemcpyHostToDevice, _stream));
  } else {
    CHECK(cudaMallocHost(&_host_input_memory,
                         sizeof(float) * _inputBinding.volume));

    cv::Size size = cv::Size(m.cols, m.rows);
    int step = size.area();

    std::vector<cv::Mat> channels;

    channels.push_back(
        cv::Mat(size, CV_32FC1, (float*)_host_input_memory + 2 * step));
    channels.push_back(
        cv::Mat(size, CV_32FC1, (float*)_host_input_memory + 1 * step));
    channels.push_back(cv::Mat(size, CV_32FC1, (float*)_host_input_memory));

    cv::split(m, channels);
    CHECK(cudaMemcpyAsync(_device_memory[0], _host_input_memory,
                          _inputBinding.volume * sizeof(float),
                          cudaMemcpyHostToDevice, _stream));
  }

  CHECK(cudaStreamSynchronize(_stream));

  CHECK(cudaFreeHost(_host_input_memory));

  if (!_context->enqueueV2(_device_memory.data(), _stream, nullptr)) {
    throw std::runtime_error("failure: could not enqueue data for inference");
  }

  if (_outputBinding.type == nvinfer1::DataType::kFLOAT) {
    CHECK(cudaMallocHost(&_host_output_memory,
                         sizeof(float) * _outputBinding.volume));
  } else {
    CHECK(cudaMallocHost(&_host_output_memory,
                         sizeof(unsigned short) * _outputBinding.volume));
  }
  if (_outputBinding.type == nvinfer1::DataType::kHALF) {
    CHECK(cudaMemcpyAsync(_host_output_memory, _device_memory[1],
                          _outputBinding.volume * sizeof(unsigned short),
                          cudaMemcpyDeviceToHost, _stream));
  } else {
    CHECK(cudaMemcpyAsync(_host_output_memory, _device_memory[1],
                          _outputBinding.volume * sizeof(float),
                          cudaMemcpyDeviceToHost, _stream));
  }

  CHECK(cudaStreamSynchronize(_stream));
}

std::vector<Detection> Processor::post_process(const cv::Mat& src) {
  cv::Mat dst = src.clone();

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> classes;

  const int nrClasses = _classes.size();

  const int networkRows = _inputBinding.dims.d[2];

  const int networkCols = _inputBinding.dims.d[3];

  const int numGridBoxes = _outputBinding.dims.d[1];

  const int rowSize = _outputBinding.dims.d[2];

  if (_outputBinding.type == nvinfer1::DataType::kFLOAT) {
    float* begin = (float*)_host_output_memory;
    for (int i = 0; i < numGridBoxes; ++i) {
      float* ptr = begin + i * rowSize;

      const float objectness = ptr[4];
      if (objectness < _scoreThreshold) {
        continue;
      }

      /*  Get the class with the highest score attached to it */
      double maxClassScore = 0.0;
      int maxScoreIndex = 0;
      for (int i = 0; i < nrClasses; ++i) {
        const float& v = ptr[5 + i];
        if (v > maxClassScore) {
          maxClassScore = v;
          maxScoreIndex = i;
        }
      }
      const double score = objectness * maxClassScore;
      if (score < _scoreThreshold) {
        continue;
      }

      const float w = ptr[2];
      const float h = ptr[3];
      const float x = ptr[0] - w / 2.0;
      const float y = ptr[1] - h / 2.0;
      boxes.push_back(cv::Rect(x, y, w, h));
      scores.push_back(score);
      classes.push_back(maxScoreIndex);
    }
  } else {
    unsigned short* begin = (unsigned short*)_host_output_memory;
    for (int i = 0; i < numGridBoxes; ++i) {
      unsigned short* ptr = begin + i * rowSize;
      const float objectness = cv::float16_t::fromBits(ptr[4]);
      if (objectness < _scoreThreshold) {
        continue;
      }

      /*  Get the class with the highest score attached to it */
      double maxClassScore = 0.0;
      int maxScoreIndex = 0;
      for (int i = 0; i < nrClasses; ++i) {
        const float& v = cv::float16_t::fromBits(ptr[5 + i]);
        if (v > maxClassScore) {
          maxClassScore = v;
          maxScoreIndex = i;
        }
      }
      const double score = objectness * maxClassScore;
      if (score < _scoreThreshold) {
        continue;
      }

      const float w = cv::float16_t::fromBits(ptr[2]);
      const float h = cv::float16_t::fromBits(ptr[3]);
      const float x = cv::float16_t::fromBits(ptr[0]) - w / 2.0;
      const float y = cv::float16_t::fromBits(ptr[1]) - h / 2.0;
      boxes.push_back(cv::Rect(x, y, w, h));
      scores.push_back(score);
      classes.push_back(maxScoreIndex);
    }
  }

  CHECK(cudaFreeHost(_host_output_memory));

  /*  Apply non-max-suppression   */
  std::vector<int> indices;

  cv::dnn::NMSBoxes(boxes, scores, _scoreThreshold, _nmsThreshold, indices);
  /*  Convert to Detection objects    */

  const double f = MIN((double)networkRows / (double)src.rows,
                       (double)networkCols / (double)src.cols);
  const cv::Size boxSize = cv::Size(src.cols * f, src.rows * f);

  const int dr = networkRows - boxSize.height;

  const int dc = networkCols - boxSize.width;

  const int topHeight = std::floor(dr / 2.0);

  const int leftWidth = std::floor(dc / 2.0);

  std::vector<Detection> result;

  for (unsigned int i = 0; i < indices.size(); ++i) {
    const int& j = indices[i];

    cv::Rect input = boxes[j];

    cv::Rect r;

    r.x = (input.x - leftWidth) / f;
    r.x = MAX(0, MIN(r.x, src.rows - 1));

    r.y = (input.y - topHeight) / f;
    r.y = MAX(0, MIN(r.y, src.cols - 1));

    r.width = input.width / f;
    if (r.x + r.width > src.cols) {
      r.width = src.cols - r.x;
    }
    r.height = input.height / f;
    if (r.y + r.height > src.rows) {
      r.height = src.rows - r.y;
    }
    const int id = classes[j];
    const std::string className = _classes[id];

    const float score = MAX(0.0, MIN(1.0, scores[j]));
    result.push_back({id, className, r, score});
  }
  return result;
}

void Processor::draw_detection(cv::Mat& src,
                               const std::vector<Detection>& detections) {
  for (const auto& detection : detections) {
    const int bboxThickness = 2;
    const double fontScale = 1.0;
    const cv::Rect& bbox = detection.boudingBox;
    const int id = detection.id;

    cv::rectangle(src, bbox, _colors[id], bboxThickness);

    char text[64];
    sprintf(text, "%s:%.2f", _classes[id].c_str(), detection.score);

    const int textThickness = 1;

    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(
        text, cv::FONT_HERSHEY_COMPLEX, fontScale, textThickness, &baseline);
    const cv::Point tl(bbox.x - bboxThickness / 2.0, bbox.y - textSize.height);
    const cv::Rect labelRect(tl, textSize);
    cv::rectangle(src, labelRect, _colors[id], -1); /*  filled rectangle */

    /*  white text on top of the previously drawn rectangle */
    const cv::Point bl(tl.x, bbox.y - bboxThickness / 2.0);
    cv::putText(src, text, bl, cv::FONT_HERSHEY_COMPLEX, fontScale,
                cv::Scalar(255, 255, 255), textThickness);
  }
}
