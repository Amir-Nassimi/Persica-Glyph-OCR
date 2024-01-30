import sys
from ultralytics import YOLO
from torch import argmax as torch_argmax
from singleton_decorator import singleton

from post_proc import working_with_results 


@singleton
class OCRModel:
    def __init__(self, model_path, plate_conf, char_conf, plate_iou, char_iou, plate_imgsz, char_imgsz, device, id_to_name, eng_to_persian):
        self.model_path = model_path
        self.plate_conf = plate_conf
        self.char_conf = char_conf
        self.plate_iou = plate_iou
        self.char_iou = char_iou
        self.plate_imgsz = plate_imgsz
        self.char_imgsz = char_imgsz
        self.device = device
        self.id_to_name = id_to_name
        self.eng_to_persian = eng_to_persian
        self.ocr_model = self.load_model()

    def load_model(self):
        try:
            return YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading the model: {e}")
            sys.exit(1)

    def detect_plate(self, img):
        try:
            results = self.ocr_model.predict(source=img, conf=self.plate_conf, iou=self.plate_iou, imgsz=self.plate_imgsz, device=self.device, classes=self.plate_classes, verbose=False)
            return self.process_plate_results(results, img)
        except Exception as e:
            print(f"Error detecting plate: {e}")
            return None

    @staticmethod
    def process_plate_results(results, img):
        for r in results:
            confs = r.boxes.conf
            if len(confs) >= 1:
                coordination = r.boxes.xyxy[torch_argmax(confs) if len(confs) > 1 else 0]
                x1, y1, x2, y2 = map(int, coordination)
                return img[y1:y2, x1:x2]
        return None

    def detect_character(self, img):
        try:
            plate = self.detect_plate(img)
            if plate is not None:
                results = self.ocr_model.predict(source=plate, conf=self.char_conf, iou=self.char_iou, imgsz=self.char_imgsz, device=self.device, classes=self.char_classes, verbose=False)
                return working_with_results(results, self.id_to_name)
            else:
                print("Plate is not detected!")
                return [None, None, None, "-", None]
        except Exception as e:
            print(f"Error in detect_character: {e}")
            return None
