import subprocess
import numpy as np
import cv2

width=1280
height=720

#width = 640        
#height = 480
width = 320       
height = 240

capture_cmd = ["gst-launch-1.0", "-q",  "v4l2src", "device=/dev/video0",
     "!", "image/jpeg,width={width},height={height},framerate=30/1".format(width=width, height=height),
     "!", "matroskamux", "streamable=true",
     "!", "fdsink",
    ]

#gst-launch-1.0 -v v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! autovideosink
#gst-launch-1.0 -v v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! autovideosink
proc = subprocess.Popen(
    capture_cmd,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    bufsize=10**5
)

# https://gstreamer.freedesktop.org/documentation/videoconvertscale/videoconvert.html?gi-language=c
decode_cmd = [
            "gst-launch-1.0", "-q",
            "fdsrc", "fd=0",
            "!", "matroskademux",
            "!", "jpegdec",
            "!", "videoconvert",
            "!", "video/x-raw(ANY),format=BGRA",
            "!", "fdsink", "fd=1"
        ]
#gst-launch-1.0 fdsrc fd=0 ! decodebin ! fdsink fd=1
decode = subprocess.Popen(
            decode_cmd,
            stdin=proc.stdout,
            stdout=subprocess.PIPE,
            bufsize=10**5
        )

def bytes_to_np_array(raw_bytes, width, height):
    """Convert raw bytes organized in separate R, G, B channels to an RGB NumPy array."""
    total_pixels = width * height
    r = np.frombuffer(raw_bytes[:total_pixels], dtype=np.uint8)
    b = np.frombuffer(raw_bytes[total_pixels:2*total_pixels], dtype=np.uint8)
    g = np.frombuffer(raw_bytes[2*total_pixels:], dtype=np.uint8)
    img = np.stack((r, g, b), axis=-1).reshape((height, width, 3))
    return img

print(width*height*3)
try:
    channels = 4
    size = width * height * channels
    while True:
        raw_data = decode.stdout.read(size)

        if len(raw_data) != size:
            continue
        arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, channels))
        img = np.transpose(arr, (0, 1, 2))
        cv2.imshow('img', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    proc.terminate()
    decode.terminate()
    cv2.destroyAllWindows()