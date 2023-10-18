#include "Decoder.h"

Decoder::Decoder() {
  _fmtCtx = avformat_alloc_context();
  if (!_fmtCtx) {
    throw std::runtime_error("failed to alloc avformat context");
  }
  _queue = new CircularQueue<cv::Mat>(50);
}

Decoder::~Decoder() {
  _running = false;

  _decode_thread->join();

  avcodec_close(_codecCtx);

  avcodec_free_context(&_codecCtx);

  avformat_free_context(_fmtCtx);

  sws_freeContext(_swsCtx);

  av_buffer_unref(&_deviceCtx);
}

bool Decoder::open(const char* url) {

  int ret = avformat_open_input(&_fmtCtx, url, NULL, NULL);
  if (ret != 0) {
    fprintf(stderr, "failed to call avformat_open_input");
    return false;
  }

  ret = avformat_find_stream_info(_fmtCtx, NULL);

  if (ret != 0) {
    fprintf(stderr, "failed to call avformat_find_stream_info");
    return false;
  }

  ret = av_find_best_stream(_fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);

  if (ret < 0) {
    fprintf(stderr, "failed to find video stream");
    return false;
  }

  _video_index = ret;

  _codecParameters = _fmtCtx->streams[_video_index]->codecpar;

  _width = _codecParameters->width;

  _height = _codecParameters->height;

  _fps = av_q2d(_fmtCtx->streams[_video_index]->avg_frame_rate);

  _bit_rate = _fmtCtx->bit_rate;

  const AVCodec* codec = avcodec_find_decoder(_codecParameters->codec_id);
  if (!codec) {
    fprintf(stderr, "failed to find the decoder:%s",
            avcodec_get_name(_codecParameters->codec_id));
    return false;
  }

  for (int i = 0;; i++) {
    const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);

    if (!config) {
      fprintf(stderr, "Decoder %s does not support device type %s.\n",
              codec->name, av_hwdevice_get_type_name(_device_type));
      return false;
    }
    if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
        config->device_type == _device_type) {
        _device_pix_fmt = config->pix_fmt;
        break;
    }
  }

  _codecCtx = avcodec_alloc_context3(codec);
  if (!_codecCtx) {
    fprintf(stderr, "failed to call avcodec_alloc_context");
    return false;
  }
  ret = avcodec_parameters_to_context(_codecCtx, _codecParameters);
  if (ret < 0) {
    fprintf(stderr, "failed to call avcodec_parameters_to_context");
    return false;
  }

  ret = av_hwdevice_ctx_create(&_deviceCtx, _device_type, NULL, NULL, 0);
  if (ret < 0) {
    fprintf(stderr, "Failed to create specified HW device.\n");
    return false;
  }

  _codecCtx->hw_device_ctx = av_buffer_ref(_deviceCtx);

  ret = avcodec_open2(_codecCtx, codec, nullptr);

  if (ret < 0) {
    fprintf(stderr, "failed to call avcodec_open2");
    return false;
  }

  return true;
}

int Decoder::Decode(AVPacket* packet) {
  AVFrame *gpu_frame = NULL, *rgb_frame = NULL;

  AVFrame* cpu_frame = NULL;

  int ret = avcodec_send_packet(_codecCtx, packet);
  if (ret < 0) {
    char buf[64];
    av_make_error_string(buf,64,AVERROR(ret));
    fprintf(stderr, "Error during decoding:%s\n",buf);
    return ret;
  }

  while (1) {
    if (!(cpu_frame = av_frame_alloc()) || !(gpu_frame = av_frame_alloc()) ||
        !(rgb_frame = av_frame_alloc())) {
      fprintf(stderr, "Can not alloc frame\n");
      ret = AVERROR(ENOMEM);
      return ret;
    }

    ret = avcodec_receive_frame(_codecCtx, gpu_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_frame_free(&gpu_frame);
      av_frame_free(&rgb_frame);
      av_frame_free(&cpu_frame);
      return 0;
    } else if (ret < 0) {
      fprintf(stderr, "Error while decoding\n");
      av_frame_free(&gpu_frame);
      av_frame_free(&rgb_frame);
      av_frame_free(&cpu_frame);
      return -1;
    }
    if (gpu_frame->format == _device_pix_fmt) {
      /* retrieve data from GPU to CPU */
      if ((ret = av_hwframe_transfer_data(cpu_frame, gpu_frame, 0)) < 0) {
        fprintf(stderr, "Error transferring the data to system memory\n");
        av_frame_free(&gpu_frame);
        av_frame_free(&rgb_frame);
        av_frame_free(&cpu_frame);
        return -1;
      } 
      if (nullptr == _swsCtx) {
        _swsCtx = sws_getContext(
            _width, _height, AVPixelFormat(cpu_frame->format), _width, _height,
            AV_PIX_FMT_BGR24, 4, nullptr, nullptr, nullptr);
      }
      cv::Mat tmpImage;
      tmpImage.create(cv::Size(_width, _height), CV_8UC3);

      av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize,
                           (uint8_t*)tmpImage.data,
                           (AVPixelFormat)AV_PIX_FMT_BGR24, _width, _height, 1);

      ret = sws_scale(_swsCtx, (uint8_t const* const*)cpu_frame->data,
                      cpu_frame->linesize, 0, cpu_frame->height,
                      rgb_frame->data, rgb_frame->linesize);

      _queue->push(tmpImage.clone());
      av_frame_free(&gpu_frame);
      av_frame_free(&rgb_frame);
      av_frame_free(&cpu_frame);
    }
  }
}

void Decoder::start() {
  _running = true;
  _decode_thread = new std::thread([=]() {
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
      fprintf(stderr, "Failed to allocate AVPacket\n");
      return;
    }

    int ret = 0;
    while (true) {
      if ((ret = av_read_frame(_fmtCtx, packet)) < 0) {
        _running = false;
        break;
      }
      if (_video_index == packet->stream_index) {
          ret=Decode(packet);
          if(ret<0){
            Decode(NULL);
          }
      }
      av_packet_unref(packet);
    }
  });
}

cv::Mat Decoder::pop()
{
    return _queue->pop();
}
