import cv2
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


def get_reference_frame():
    # Variables for stabilization check
    stability_threshold = 0.02  # magic number
    diffs = np.arange(10)  # some numbers to force some iterations
    previous_frame = None
    stabilized = False

    while not stabilized:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_frame is not None:
            diff = cv2.absdiff(previous_frame, gray)
            mean_diff = np.mean(diff)
            diffs = np.append(diffs, mean_diff)
            diffs = diffs[1:]
            print(f"Frame change: {mean_diff:.2f}", np.std(diffs))  # TODO remove

            if np.std(diffs) < stability_threshold:
                stabilized = True
                previous_frame = frame
                break
        previous_frame = gray

    return previous_frame


# Convert reference image to grayscale
reference_gray = cv2.cvtColor(get_reference_frame(), cv2.COLOR_BGR2GRAY)
reference_gray = cv2.GaussianBlur(reference_gray, (9, 9), 0)
#cv2.imshow("LED reference_frame", reference_gray)

blink_times = []
start_time = time.time()

led_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # gray and blur to minimize noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    diff = cv2.absdiff(reference_gray, gray)

    # Apply thresholding to highlight changed areas (LEDs turning on)
    # input: image, threshold, max value, what to do: in this case everything < thresh = 0 else 255
    _, thresholded = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)  # TODO make threshold configurable

    # Find contours of changed regions
    # input: "binary" image, how to draw contours: only external, how "many": in this case less is better
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    old_led_detected = led_detected
    
    led_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 20:  # Adjust sensitivity if needed
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.9 < aspect_ratio < 1.1: # we more or less only need quadratic rects, because led = point light source
                led_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Detect first LED blink based on new contours appearing
    if led_detected and not blink_times:
        blink_times.append(time.time() - start_time)
        print(f"First LED activation detected at {blink_times[-1]:.2f} seconds")
    
    if led_detected != old_led_detected:
        print("LED is", "on" if led_detected else "off")

    # Display results
    cv2.imshow("LED Tracking", frame)
    cv2.imshow("diff", diff)
    cv2.imshow("Difference", thresholded)
    #cv2.imshow("gray", gray)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print detected blink times
print("Detected LED activation time:", blink_times)
