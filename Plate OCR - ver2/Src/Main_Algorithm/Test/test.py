import sys
sys.path.insert(0, "../")

import os
os.chdir('../../../')

import argparse
from ocrPlate.Src.Main_Algorithm.Codes.main import OCRModel
import cv2
import numpy as np
import time

def detect_and_print(ocr_model, img_paths):

    results = {}
    for img_path in img_paths:
        img = cv2.imread(img_path)
        # Main method!
        detection_list, median_conf, detected_car = ocr_model.detect_character(img)
        results[img_path] = [detection_list, median_conf, detected_car]

    return results


def time_evaluation(ocr_model, img_paths, runs_num):

    ####################################################### Warmup #######################################################
    img_path = img_paths[0]
    img = cv2.imread(img_path)
    # Main method!
    detection_list, median_conf, detected_car = ocr_model.detect_character(img)
    ####################################################### Warmup #######################################################
    

    times = []
    for i in range(runs_num):
        for img_path in img_paths:
            img = cv2.imread(img_path)
            start_time = time.time()
            # Main method!
            detection_list, median_conf, detected_car = ocr_model.detect_character(img)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)

    execution_time = np.mean(times)
    return execution_time

def main():
    parser = argparse.ArgumentParser(description="OCR Module Evaluating")
    parser.add_argument("--device", type=int, help="{0: gpu, 1: cpu}")
    parser.add_argument("--runs_num", type=int, help="{Repeat the detection to obtain a valid runtime}")


    args = parser.parse_args()
    device = args.device
    runs_num = args.runs_num
    
    img_paths = ["Datasets/IR_LPR/test_samples/day_00474.jpg",
                 "Datasets/IR_LPR/test_samples/day_00019.jpg",
                 "Datasets/IR_LPR/test_samples/night (2270).jpg",
                 "Datasets/IR_LPR/test_samples/day_00069.jpg",
                 "Datasets/IR_LPR/test_samples/day_16278.jpg",
                 "Datasets/IR_LPR/test_samples/day_01483.jpg",
                 "Datasets/IR_LPR/test_samples/day_02103.jpg",
                 "Datasets/IR_LPR/test_samples/day_02414.jpg",
                 "Datasets/IR_LPR/test_samples/day_02958.jpg",
                 "Datasets/IR_LPR/test_samples/day_11520.jpg",
                 "Datasets/IR_LPR/test_samples/net.png"
                 ]

    # device=0 for cuda and device='cpu' for cpu

    if device == 0:
        d = 0
        ocr_model = OCRModel(model_path="Models/OCR_0/best.pt",
                             plate_conf=0.6,
                             char_conf=0.5,
                             plate_iou=0.7,
                             char_iou=0.7,
                             plate_imgsz=(640, 640),
                             char_imgsz=(320, 320),
                             device=d)

        elapsed_time = time_evaluation(ocr_model, img_paths, runs_num)
        print(f"Average elapsed time: {round(elapsed_time, 2)} seconds for gpu\n")
        print("---------------------------------------------------------------------------------------------------------")
        results = detect_and_print(ocr_model, img_paths)
        for key in results:
            print(f"OCR result: {key}    :    {results[key]}")

    elif device == 1:
        d = 'cpu'
        ocr_model = OCRModel(model_path="Models/OCR_0/best.pt",
                             plate_conf=0.6,
                             char_conf=0.5,
                             plate_iou=0.7,
                             char_iou=0.7,
                             plate_imgsz=(640, 640),
                             char_imgsz=(320, 320),
                             device=d)

        elapsed_time = time_evaluation(ocr_model, img_paths, runs_num)
        print(f"Average elapsed time: {round(elapsed_time, 2)} seconds for cpu\n")
        print("---------------------------------------------------------------------------------------------------------") 
        results = detect_and_print(ocr_model, img_paths)
        for key in results:
            print(f"OCR result: {key}    :    {results[key]}")


if __name__ == "__main__":
    main()