# rtmp-yolo-tensorrt
using ffmpeg to decoder the video then use tenssorrt to infer and then push to the rtmp server

there are three part of this work:
decoder: using the ffmpeg to decode the video to cv::Mat,using the cuda to accelerate it provided by ffmpeg
processor: using  tenssorrt with the yolov5 to detect,
pusher: encode the cv::mat to h264stream and push the h264stream to the rtmpserver

