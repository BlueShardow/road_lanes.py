import numpy as np
import cv2 as cv
import math
import os
import time
import sys
from ultralytics import YOLO

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
    frame = cv.GaussianBlur(frame, (5, 5), 0)
    frame = cv.medianBlur(frame, 5)
    frame = cv.bilateralFilter(frame, 9, 75, 75)

def main():
    model = YOLO('yolov8n.pt')
    model.info()

    vid_path = "tobeaddedlater"
    cap = cv.VideoCapture(vid_path)
    frame_skip = 10

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


        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
