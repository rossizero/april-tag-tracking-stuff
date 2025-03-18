import logging

from labgrid import Environment
from labgrid.logging import basicConfig, StepLogger
import os
import cv2

parent_folder = os.path.abspath(os.path.dirname(__file__))

basicConfig(level=logging.INFO)
StepLogger.start()

e = Environment(os.path.join(parent_folder, "import-video.yaml"))
t = e.get_target()

p = t.get_driver("USBVideoDriver")
print(p.video.path)

p.start_stream(caps_hint="mid")

while p.is_stream_open():
    ret, frame = p.read()

    if ret:
        cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

p.stop_stream()
cv2.destroyAllWindows()

