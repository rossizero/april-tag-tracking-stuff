import logging
import time

from labgrid import Environment
from labgrid.logging import basicConfig, StepLogger

# enable info logging
basicConfig(level=logging.INFO)

# show labgrid steps on the console
StepLogger.start()

e = Environment("/home/rossi/repos/april-tag-tracking-stuff/labgrid/import-video.yaml")
t = e.get_target()

p = t.get_driver("USBVideoDriver")
print(p.video.path)
p.get_opencv_cap(caps_hint="mid")
print(p)

