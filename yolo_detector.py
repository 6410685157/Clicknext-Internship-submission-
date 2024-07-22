from ultralytics import YOLO
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors

import cv2 as cv

# Load YOLO model
model = YOLO("yolov8n.pt")
track_history = defaultdict(lambda: [])

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""
    track_ids = [0]
    # Create annotator object
    annotator = Annotator(frame)
    for box, track_id in zip(boxes, track_ids):
        # print(box)
        x, y, w, h = box.xywh[0]
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

    # Draw bounding box
    annotator.box_label(
        box=coordinator, label='Cat', color=(250, 50, 0)
    )

    # Draw tracking line
    annotator.draw_centroid_and_tracks(
        track, color=(255, 0, 255), track_thickness=2
    )

    return annotator.result()

def detect_object(frame):
    """Detect object from image frame"""
    # Detect object from image frame
    results = model.track(frame, classes=15, tracker="bytetrack.yaml", persist=True)

    for result in results:
      if len(result.boxes) > 0:
        frame = draw_boxes(frame, result.boxes)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    # video_writer = cv.VideoWriter(
    #     video_path + "_demo.avi", cv.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)
    # )

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if ret:
            # Detect motorcycle from image frame
            frame_result = detect_object(frame)

            # Write result to video
            cv.putText(
                frame,
                text = "Thanakorn-Praimanee-Clicknext-Internship-2024",
                org = (450, 50),
                fontFace = cv.FONT_HERSHEY_SIMPLEX,
                fontScale = 1,
                color = (0, 0, 250),
                thickness = 2,
            )
            # video_writer.write(frame_result)

            # Show result
            cv.namedWindow("Video", cv.WINDOW_NORMAL)
            cv.imshow("Video", frame_result)
            cv.waitKey(30)

        else:
            break

    # Release the VideoCapture object and close the window
    # video_writer.release()
    cap.release()
    cv.destroyAllWindows()
