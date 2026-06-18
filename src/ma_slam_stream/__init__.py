"""ma_slam_stream — real-time RGBD streaming front-end for ma_slam.

Server side (needs the repo + GPU + MapAnything/DA3):
    ma_slam_stream.run_stream   CLI: --source {local, zmq, folder}
    ma_slam_stream.sources      LocalRealSenseSource / ZmqFrameSource / FolderFrameSource
Laptop side (camera host; needs only pyrealsense2 + opencv + numpy [+ pyzmq]):
    ma_slam_stream.client       capture + blur-select + send (zmq / folder)
    ma_slam_stream.realsense    RealSenseCapture (shared; official API + post-processing)

See README.md.
"""
