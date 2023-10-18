#pragma once
extern "C" {
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
}
#include <opencv2/opencv.hpp>
#include "circleQueue.hpp"
#include<atomic>
#include<thread>

class Decoder {

public:

	Decoder();

	~Decoder();

	bool open(const char* url);

	int witdh()const { return _width; }

	int height()const { return _height; }

	double fps()const { return _fps; }

	int64_t bit_rate()const { return _bit_rate; }

	bool isrunning()const { return _running.load();}

	void start();

	cv::Mat pop();

private:
		
	int Decode(AVPacket* packet);

	AVFormatContext* _fmtCtx{ nullptr };

	AVCodecContext* _codecCtx{ nullptr };

	AVCodec* _codec{ nullptr };

	AVBufferRef *_deviceCtx{nullptr};

	AVHWDeviceType _device_type{AV_HWDEVICE_TYPE_CUDA};
  	
	AVPixelFormat _device_pix_fmt{AV_PIX_FMT_NONE};

	SwsContext* _swsCtx{ nullptr };

	int _width, _height;

	double _fps;

	int64_t _bit_rate;

	int _video_index{ -1 };

	AVCodecParameters* _codecParameters{ nullptr };

	

	CircularQueue<cv::Mat>* _queue;

	std::atomic<bool> _running{ false };

	std::thread* _decode_thread{nullptr};



};