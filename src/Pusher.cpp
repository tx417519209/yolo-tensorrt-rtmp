
#include "Pusher.h"

Pusher::Pusher()
{
  _queue = new CircularQueue<cv::Mat>(50);
}
Pusher::~Pusher()
{
  _running = false;
  _encode_thread->join();
  while (!_queue->empty())
  {
    Encode(_queue->pop());
  }
  avformat_flush(_fmtCtx);
  av_write_trailer(_fmtCtx);
  if(_fmtCtx->pb)
    avio_close(_fmtCtx->pb);

  avformat_free_context(_fmtCtx);

  sws_freeContext(_swsCtx);

  av_buffer_unref(&_deviceCtx);

  delete _queue;
}

bool Pusher::open(const char* url)
{


  const AVCodec* _codec = avcodec_find_encoder_by_name("h264_nvenc");
  if (!_codec) {
    fprintf(stderr, "failed to call avcodec_find_encoder");
    return false;
  }
  _codecCtx = avcodec_alloc_context3(_codec);

  if(!_codecCtx) {
      fprintf(stderr, "failed to call avcodec_alloc_context3");
      return false;
  }

  const AVRational dst_fps ={(int) _fps,1 };
  _codecCtx->height = _height;
  _codecCtx->codec_tag = 0;
  _codecCtx->width = _width;
  _codecCtx->max_b_frames = 0;
  _codecCtx->bit_rate = _bit_rate;
  _codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  _codecCtx->profile = FF_PROFILE_H264_HIGH;
  _codecCtx->codec_id = _codec->id;
  _codecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
  _codecCtx->framerate = dst_fps;
  _codecCtx->time_base = av_inv_q(dst_fps);
  _codecCtx->pix_fmt=_codec->pix_fmts[0];
  _device_pix_fmt=_codecCtx->pix_fmt;

  int ret = avcodec_open2(_codecCtx, _codec, NULL);
  if (ret < 0) {
    char buf[64];
    av_make_error_string(buf,64,ret);
    fprintf(stderr, "failed to call avcodec_open2:%s\n",buf);
    return false;
  }

  ret = avformat_alloc_output_context2(&_fmtCtx, NULL, "flv", url);
  if (ret < 0) {
    fprintf(stderr, "failed to call avformat_alloc_output_context2");
    return false;
  }

  _stream = avformat_new_stream(_fmtCtx, _codec);
  if (!_stream) {
    fprintf(stderr, "Could not allocate stream\n");
    return false;
  }

   
  _video_index= _stream->index;
  
  _stream->avg_frame_rate = dst_fps;
 
  _stream->time_base=dst_fps;

  ret = avcodec_parameters_from_context(_stream->codecpar, _codecCtx);
  if (ret < 0) {
    fprintf(stderr, "failed to call avcodec_parameters_from_context");
    return false;
  }

  av_dump_format(_fmtCtx, 0, url, 1);

  if (!(_fmtCtx->flags & AVFMT_NOFILE)) {
    ret = avio_open(&_fmtCtx->pb, url, AVIO_FLAG_WRITE);
    if (ret < 0) {
      fprintf(stderr, "failed to call avio_open2");
      return false;
    }
  }

  if (avformat_write_header(_fmtCtx, nullptr) < 0) {
    fprintf(stderr, "Could not write header!\n");
    return false;
  }

  return true;
  
}

bool Pusher::setCodecParameters(int width,int height,double fps,int64_t bit_rate)
{
  _height = height;
  _width = width;
  _fps = fps;
  _bit_rate = bit_rate;
  return true;
}

void Pusher::start()
{
 
  _running = true;

  _encode_thread = new std::thread([=]() {
 
    while (_running) {
      cv::Mat m = _queue->pop();
      Encode(m);
    }
    return ;
   });
}

void Pusher::push(const cv::Mat& m)
{
  _queue->push(m);
}

int Pusher::Encode(const cv::Mat& m)
{
  AVFrame *cpu_frame = NULL, *gpu_frame = NULL;

  AVPacket* pkt=av_packet_alloc();

  if (nullptr == _swsCtx) {
    _swsCtx = sws_getContext(_width, _height, AV_PIX_FMT_BGR24,
      _width, _height, _device_pix_fmt,
      4, nullptr, nullptr, nullptr);
  }


  cpu_frame=av_frame_alloc();
  cpu_frame->height = _height;
  cpu_frame->width = _width;
  cpu_frame->format = AV_PIX_FMT_BGR24;
  av_frame_get_buffer(cpu_frame, 0);

  gpu_frame=av_frame_alloc();
  gpu_frame->height = _height;
  gpu_frame->width = _width;
  gpu_frame->format = AV_PIX_FMT_YUV420P;
  av_frame_get_buffer(gpu_frame, 0);

  av_image_fill_arrays(cpu_frame->data, cpu_frame->linesize, m.data, AV_PIX_FMT_BGR24, _width, _height, 1);

  int ret=sws_scale(_swsCtx,
    (uint8_t const* const*)cpu_frame->data, cpu_frame->linesize, 0, cpu_frame->height,
    gpu_frame->data, gpu_frame->linesize);
  if (ret < 0) {
    return ret;
  }
  gpu_frame->pts = _pts++;
  ret = avcodec_send_frame(_codecCtx, gpu_frame);
  if (ret < 0)
  {
    fprintf(stderr, "Error sending frame to codec context!\n");
    return ret;
  }
  while (ret >= 0) {
    ret = avcodec_receive_packet(_codecCtx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      //fprintf(stderr, "error eagain or eof\n");
      break;
    }
    else if (ret < 0) {
      fprintf(stderr, "Error during encoding\n");
      av_packet_free(&pkt);
      av_frame_unref(gpu_frame);
      av_frame_unref(cpu_frame);
      return -1;
    }
    
    pkt->stream_index = _video_index;
    av_packet_rescale_ts(pkt, _codecCtx->time_base, _stream->time_base);
   
    ret = av_interleaved_write_frame(_fmtCtx, pkt);
    if(ret!=0){
      fprintf(stderr, "av_interleaved_write_frame error\n");
      break;
    }
  }
  av_packet_free(&pkt);
  av_frame_unref(gpu_frame);
  av_frame_unref(cpu_frame);
  return 0;
}

