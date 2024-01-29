import os
import cv2
import time
import glob
import pickle
import argparse
import numpy as np
from torch.cuda import is_available as Cuda_Available

from OCR.main_model import OCRModel


class OCROperations:
    def __init__(self, model_params, mapping_files, output_dir):
        self.load_mappings(mapping_files)
        self.ocr_model = OCRModel(**model_params, id_to_name=self.id_to_name, eng_to_persian=self.eng_to_persian)
        self.output_dir = output_dir

    def load_mappings(self, mapping_files):
        with open(mapping_files['id_to_name'], 'rb') as file:
            self.id_to_name = pickle.load(file)
        with open(mapping_files['eng_to_persian'], 'rb') as file:
            self.eng_to_persian = pickle.load(file)

    def detect_and_print(self, img_paths):
        results = {}
        for img_path in img_paths:
            img = cv2.imread(img_path)
            detection_list = self.ocr_model.detect_character(img)
            results[img_path] = detection_list
            self.save_result(img_path, detection_list)
        return results

    def save_result(self, img_path, detection_list):
        base_name = os.path.basename(img_path)
        result_path = os.path.join(self.output_dir, f"{base_name}_result.txt")
        with open(result_path, 'w') as file:
            file.write(f"{detection_list}\n")

    def time_evaluation(self, img_paths, runs_num):
        img = cv2.imread(img_paths[0])
        self.ocr_model.detect_character(img)

        times = []
        for _ in range(runs_num):
            for img_path in img_paths:
                img = cv2.imread(img_path)
                start_time = time.time()
                self.ocr_model.detect_character(img)
                end_time = time.time()
                times.append(end_time - start_time)

        return np.mean(times)

def main():
    parser = argparse.ArgumentParser(description="OCR Module Evaluating")
    parser.add_argument("--runs_num", type=int, help="Repeat the detection to obtain a valid runtime")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing test images")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save results")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model", default="../../../Models/OCR_0/best.pt")
    parser.add_argument("--plate_conf", type=float, help="Confidence threshold for plate detection", default=0.83)
    parser.add_argument("--char_conf", type=float, help="Confidence threshold for character detection", default=0.5)
    parser.add_argument("--plate_iou", type=float, help="IOU threshold for plate detection", default=0.7)
    parser.add_argument("--char_iou", type=float, help="IOU threshold for character detection", default=0.7)
    parser.add_argument("--plate_imgsz", type=int, nargs=2, help="Image size for plate detection", default=(640, 640))
    parser.add_argument("--char_imgsz", type=int, nargs=2, help="Image size for character detection", default=(320, 320))

    args = parser.parse_args()

    device = 'cuda' if Cuda_Available() else 'cpu'
    runs_num = args.runs_num
    input_dir = args.input_dir
    output_dir = args.output_dir

    model_params = {
        "model_path": args.model_path,
        "plate_conf": args.plate_conf,
        "char_conf": args.char_conf,
        "plate_iou": args.plate_iou,
        "char_iou": args.char_iou,
        "plate_imgsz": tuple(args.plate_imgsz),
        "char_imgsz": tuple(args.char_imgsz),
        "device": device
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_paths = glob.glob(os.path.join(input_dir, '*.jpg'))

    mapping_files = {
        'id_to_name': './Assets/character_id_mapping.pkl',
        'eng_to_persian': './Assets/persian_alphabet_translation.pkl'
          }

    ocr_operations = OCROperations(model_params, mapping_files, output_dir)
    elapsed_time = ocr_operations.time_evaluation(img_paths, runs_num)
    
    print(f"Using device: {device}")
    print(f"Average elapsed time: {round(elapsed_time, 2)} seconds\n")
    print("---------------------------------------------------------------------------------------------------------")

    results = ocr_operations.detect_and_print(img_paths)
    for key, value in results.items():
        print(f"OCR result for {key}: {value}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
