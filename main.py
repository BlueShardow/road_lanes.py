import numpy as np
import cv2 as cv
#import math
#import os
import time
#import sys
#import torch
#from ultralytics import YOLO

def resize_frame(frame):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_height = 480
    new_width = int(new_height * aspect_ratio)

    return cv.resize(frame, (new_width, new_height)), new_height, new_width

def enhance_contrast(frame):
    return cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX)

def preprocess_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame = cv.GaussianBlur(frame, (5, 5), 0)
    frame = cv.medianBlur(frame, 5)
    frame = cv.bilateralFilter(frame, 9, 75, 75)

    return frame

def draw_arrow(frame, start_point, end_point, color, thickness):
    cv.arrowedLine(frame, start_point, end_point, color, thickness)

    return frame

def get_perspective_transform(frame, roi_points, width, height):
    src_pts = np.float32([roi_points])

    dst_pts = np.float32([
        [0, height],       # bottom left -> (0, height)
        [width, height],   # bottom right -> (width, height)
        [width, 0],        # top right -> (width, 0)
        [0, 0],            # top left -> (0, 0)
        [50, height // 2]  # mid left -> arbitrary
    ])

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
    warped = cv.warpPerspective(frame, H, (width, height))

    return warped

"""
def get_perspective_transform(roi_points, width, height):
    src_pts = np.float32(roi_points)
    dst_pts = np.float32([
        [0, height],  # bottom left
        [width, height],  # bottom right
        [width, 0],  # top right
        [0, 0]  # top left
    ])
    
    matrix = cv.getPerspectiveTransform(src_pts, dst_pts)

    return matrix
"""

def main():
    #model = YOLO("yolov8n.pt")
    #model.info()

    vid_path = "/Users/pl1001515/Downloads/Sunday Drive Along Country Roads During Spring, USA ｜ Driving Sounds for Sleep and Study.mp4"
    cap = cv.VideoCapture(vid_path)
    frame_skip = 5

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_count = 0

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
        preprocessed_frame = preprocess_frame(contrast_frame)

        roi_point0 = (0, height - 10) #bottom left
        roi_point1 = (width - 200, height - 10) # bottom right
        roi_point2 = (width - 375, 285) # top right
        roi_point3 = (325, 285) # top left
        roi_point4 = (125, 355) # mid left

        mask = np.zeros_like(preprocessed_frame)
        roi_corners = np.array([roi_point0, roi_point1, roi_point2, roi_point3, roi_point4], dtype=np.int32)
        cv.fillPoly(mask, [roi_corners], 255)

        roi_frame = cv.bitwise_and(preprocessed_frame, mask)

        roi_points = np.array([[roi_point0], [roi_point1], [roi_point2], [roi_point3], [roi_point4]], np.int32).reshape((-1, 1, 2))
        cv.polylines(frame, [roi_points], True, (0, 255, 0), 2)

        warped_frame = get_perspective_transform(preprocess_frame, [roi_point0, roi_point1, roi_point2, roi_point3, roi_point4], width, height)

        frame = draw_arrow(frame, (25, 100), (25, 5), (0, 255, 0), 2)

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
