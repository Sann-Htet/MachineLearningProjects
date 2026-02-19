# # Import All the Requried libraries
# from ultralytics import YOLO

# # Load the YOLO11 Model
# model = YOLO("yolo11n.pt")

# # Tracking with default tracker bot-sort
# # results = model.track(source = "video1.mp4", show = True, save = True)
# # Tracking with Byte-track
# results = model.track(source = "video1.mp4", show = True, save = True, tracker = "bytetrack.yaml", conf = 0.20, iou = 0.3)

#---------------------------------#
# Python Script using OpenCV-Python (cv2) and YOLO11 to run Obect Tracking on Video Frames and on Live Webcam Feed

# Import All the Requried libraries
import cv2
from ultralytics import YOLO

# Load the YOLO11 Model
model = YOLO("yolo11n.pt")

# Create a Video Capture Obect
cap = cv2.VideoCapture("video1.mp4")

# Loop through Video Frames
while True:
    ret, frame = cap.read()
    if ret:
        # Run YOLO11 Tracking on the Video Frames
        result = model.track(frame, persist=True)
        # Visualize the results on the frame
        annotated_frame = result[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLO11 Object Tracking", annotated_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()