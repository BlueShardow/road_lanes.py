import numpy as np
import cv2 as cv
import json
import os
import time
from ultralytics import YOLO
import torch

print("NumPy version:", np.__version__)
print("Torch version:", torch.__version__)

def load_apollo_data(data_path):
    image_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.jpg', '.png'))])
    json_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.json')])
    
    return image_files, json_files

def parse_apollo_json(json_file):
    with open(json_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}")
            return {}

    print(f"JSON structure of {json_file}: {data.keys()}")  # Debugging line
    
    if 'lane_data' in data:
        return data['lane_data']
    
    print(f"Warning: Unexpected JSON format in {json_file}")
    
    return data  # Return full data for debugging

def resize_frame(frame):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_height = 480
    new_width = int(new_height * aspect_ratio)
    return cv.resize(frame, (new_width, new_height)), new_height, new_width

def enhance_contrast(frame):
    return cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX)

def process_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.medianBlur(frame, 5)
    frame = cv.bilateralFilter(frame, 9, 75, 75)
    return frame

def draw_arrow(frame, start_point, end_point, color, thickness):
    cv.arrowedLine(frame, start_point, end_point, color, thickness)
    return frame

def get_perspective_transform(frame, roi_points, width, height):
    src_pts = np.float32(roi_points)
    dst_pts = np.float32([
        [0, height],
        [width, height],
        [width, 0],
        [0, 0]
    ])
    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
    warped = cv.warpPerspective(frame, H, (width, height))
    return warped

def main():
    # Load the Apollo pre-trained YOLO model
    model = YOLO("apollo_yolo.pt")
    model.info()

    apollo_path = "/Users/pl1001515/apollo"  # Change this to your dataset path
    image_files, json_files = load_apollo_data(apollo_path)
    
    vid_path = "/Users/pl1001515/Downloads/Sunday Drive Along Country Roads During Spring, USA ｜ Driving Sounds for Sleep and Study.mp4"
    cap = cv.VideoCapture(vid_path)
    frame_skip = 5

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    json_index = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame, height, width = resize_frame(frame)
        contrast_frame = enhance_contrast(frame)
        preprocessed_frame = process_frame(contrast_frame)

        if json_index < len(json_files):
            annotation = parse_apollo_json(json_files[json_index])
            print(f"Processing frame {frame_count} with annotation: {annotation}")
            json_index += 1

        roi_points = [
            (0, height - 40),  # bottom left
            (width - 200, height - 10),  # bottom right
            (width - 325, 300),  # top right
            (215, 300)  # top left
        ]

        mask = np.zeros_like(preprocessed_frame)
        roi_corners = np.array(roi_points, dtype=np.int32)
        cv.fillPoly(mask, [roi_corners], 255)
        roi_frame = cv.bitwise_and(preprocessed_frame, mask)

        roi_points_np = np.array(roi_points, np.int32).reshape((-1, 1, 2))
        cv.polylines(frame, [roi_points_np], True, (0, 255, 0), 2)

        warped_frame = get_perspective_transform(preprocessed_frame, roi_points, width, height)
        frame = draw_arrow(frame, (25, 100), (25, 5), (0, 255, 0), 2)

        # Run inference using the Apollo YOLO model
        results = model(frame)
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = box.cls
                confidence = box.conf
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow("Frame", frame)
        cv.imshow("Preprocessed Frame", preprocessed_frame)
        cv.imshow("ROI Frame", roi_frame)
        cv.imshow("Warped Frame", warped_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
