import cv2
from ultralytics import YOLO  

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(f"Mouse moved to: {point}")

# Load YOLO model
model = YOLO(r"C:\Users\nevil\Desktop\custom-obj-track-count-yolo11-main\custom-obj-track-count-yolo11-main\best.pt")

# Open video file
cap = cv2.VideoCapture(r"C:\Users\nevil\Desktop\custom-obj-track-count-yolo11-main\custom-obj-track-count-yolo11-main\Dataset_3.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a full-screen window
cv2.namedWindow("Traffic Surveillance", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Traffic Surveillance", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.setMouseCallback("Traffic Surveillance", RGB)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # Resize frame to fit the screen (optional)
    screen_width = 1920  # Adjust according to your screen
    screen_height = 1080
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Object tracking with YOLO
    results = model.track(frame, persist=True)  
    print(results)  

    # Annotate frame with detected objects
    frame = results[0].plot()

    # Show frame in full-screen mode
    cv2.imshow("Traffic Surveillance", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
