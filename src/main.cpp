#include "Decoder.h"
#include "Pusher.h"
#include "Processor.h"
#include<opencv2/videoio.hpp>
int main(int argc,char** argv) {
		

	std::string model_file = "/home/xing/yolov5-tensorrt/build/yolov5s.engine";

	std::string classes_file = R"(/home/xing/learnffmpeg/labels.txt)";

	std::string video_file=R"(/home/xing/learnffmpeg/test.mp4)";


	Processor processor(model_file,classes_file);



	

	Decoder decoder;

	if (!decoder.open(video_file.c_str())) {
		fprintf(stderr, "failed to open ");
	 	return 1;
	}

	decoder.start();
	
	Pusher pusher;
	
	pusher.setCodecParameters(decoder.witdh(), decoder.height(), decoder.fps(),decoder.bit_rate());

	if (!pusher.open(R"(rtmp:119.3.60.202:11935/live/test)")) {
		fprintf(stderr, "failed to open ");
		return 1;
	}

	pusher.start();

	while(true) {
		auto mat = decoder.pop();
		if(mat.empty())
			break;
		auto start = cv::getTickCount();	
		auto dst = processor.process(mat);
		auto end = cv::getTickCount();
		auto used_time = (end - start) / (cv::getTickFrequency());
		fprintf(stderr, "used time:%.5f\n", used_time);
		pusher.push(dst);
	}
}