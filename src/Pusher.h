#pragma once

extern "C" {
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
}
#include <opencv2/opencv.hpp>
#include "circleQueue.hpp"
#include <atomic>
#include<thread>
class Pusher {
public:

	Pusher();

	~Pusher();

	bool open(const char* url);

	bool setCodecParameters(int width, int height, double fps,int64_t bit_rate);

	void start();

	void push(const cv::Mat&);


private:

		int Encode(const cv::Mat& m);

		AVFormatContext* _fmtCtx{ nullptr };

		AVCodecContext* _codecCtx{ nullptr };

		AVCodec* _codec{ nullptr };

		SwsContext* _swsCtx{ nullptr };

		AVBufferRef *_deviceCtx{nullptr};

		AVHWDeviceType _device_type{AV_HWDEVICE_TYPE_CUDA};
  	
		AVPixelFormat _device_pix_fmt{AV_PIX_FMT_NONE};

		AVStream* _stream{ nullptr };

		int _width, _height;

		double _fps;

		int64_t _bit_rate;

		int64_t _pts{ 1 };

		int _video_index{ -1 };

		AVCodecParameters* _codecParameters{ nullptr };

		CircularQueue<cv::Mat>* _queue;

		std::atomic<bool> _running{ false };

		std::thread* _encode_thread{ nullptr };

};
