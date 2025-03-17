import logging
import time

from labgrid import Environment
from labgrid.logging import basicConfig, StepLogger
import os
import cv2

# Get the absolute path of the script file

parent_folder = os.path.abspath(os.path.dirname(__file__))

# enable info logging
basicConfig(level=logging.INFO)

# show labgrid steps on the console
StepLogger.start()

a = cv2.IStreamReader()
e = Environment(os.path.join(parent_folder, "import-video.yaml"))
t = e.get_target()

p = t.get_driver("USBVideoDriver")
print(p.video.path)
p.start_stream(caps_hint="mid")
#cap = cv2.VideoCapture(p, apiPreference=cv2.CAP_GSTREAMER, params=[])

# if gstreamer opencv support
#gst_pipeline = (
#    "shmsrc socket-path=/tmp/gst-shm ! "
#    "video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! "
#    "videoconvert ! appsink sync=false max-buffers=1 drop=true"
#)
#print(gst_pipeline)
#cap = cv2.VideoCapture(gst_pipeline, apiPreference=cv2.CAP_GSTREAMER)
#if not cap.isOpened():
#    print("Error: Could not open GStreamer pipeline.")
#    exit()

while p.is_stream_open():
    ret, frame = p.read()
    print(frame.shape)

    if ret:
        cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
p.stop_stream()
cv2.destroyAllWindows()

print(p)

