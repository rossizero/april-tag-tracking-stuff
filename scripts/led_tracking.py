import cv2
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


def get_reference_frame():
    # Capture reference image (before LEDs turn on)
    print("Capturing reference image... Please ensure LEDs are OFF.")

    # Variables for stabilization check
    stability_threshold = 0.02  # Adjust this for sensitivity
    diffs = np.arange(10)
    previous_frame = None
    stabilized = False

    print("Waiting for camera to stabilize...")

    while not stabilized:
        ret, frame = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compare with previous frame if it exists
        if previous_frame is not None:
            diff = cv2.absdiff(previous_frame, gray)
            mean_diff = np.mean(diff)
            diffs = np.append(diffs, mean_diff)
            diffs = diffs[1:]
            print(f"Frame change: {mean_diff:.2f}", np.std(diffs))  # Debugging info

            if np.std(diffs) < stability_threshold:
                stabilized = True
                previous_frame = frame
                break

        # Update previous frame for next comparison
        previous_frame = gray

    # Use reference_frame as the baseline for LED tracking
    print("Reference frame captured.")
    return previous_frame




# Convert reference image to grayscale
reference_gray = cv2.cvtColor(get_reference_frame(), cv2.COLOR_BGR2GRAY)
reference_gray = cv2.GaussianBlur(reference_gray, (9, 9), 0)
#cv2.imshow("LED reference_frame", reference_gray)

blink_times = []
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Compute absolute difference between reference and current frame
    diff = cv2.absdiff(reference_gray, gray)

    # Apply thresholding to highlight changed areas (LEDs turning on)
    _, thresholded = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY)

    # Find contours of changed regions
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    led_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 20:  # Adjust sensitivity if needed
            led_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Detect first LED blink based on new contours appearing
    if led_detected and not blink_times:
        blink_times.append(time.time() - start_time)
        print(f"First LED activation detected at {blink_times[-1]:.2f} seconds")

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
