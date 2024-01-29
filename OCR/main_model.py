import sys
import pickle
from ultralytics import YOLO
from torch import argmax as torch_argmax

from postprocessing import working_with_results


class OCRModel:
    def __init__(self, model_path: str, plate_conf: float = 0.83, char_conf: float = 0.5, plate_iou: float = 0.7, char_iou: float = 0.7, plate_imgsz: tuple = (640, 640), char_imgsz: tuple = (320, 320), device='cpu', mapping_files: dict = None):
        
        """
        Constructor method that initializes an instance of the PlateOCR class.

        Args:
            model_path (str): Path to the YOLO model used for object detection.
            plate_conf (float): Confidence threshold for plate detection (default is 0.83).
            char_conf (float): Confidence threshold for character detection (default is 0.5).
            plate_iou (float): IOU threshold for plate detection (default is 0.7).
            char_iou (float): IOU threshold for character detection (default is 0.7).
            plate_imgsz (tuple): Image size for plate detection (default is (640, 640)).
            char_imgsz (tuple): Image size for character detection (default is (320, 320)).
            device (int or str): device to run on, i.e. cuda device=0/1/2/3 or device='cpu'. int type for gpu and str type for cpu (default is 0).

        Initializes various parameters and loads the YOLO model.
        """
        
        self.validate_parameters(model_path, plate_conf, char_conf, plate_iou, char_iou, plate_imgsz, char_imgsz, device, mapping_files)
        self.initialize_attributes(model_path, plate_conf, char_conf, plate_iou, char_iou, plate_imgsz, char_imgsz, device, mapping_files)
        self.ocr_model = self.load_model()

    @staticmethod
    def validate_parameters(model_path, plate_conf, char_conf, plate_iou, char_iou, plate_imgsz, char_imgsz, device, mapping_files):
        param_types = {
            "model_path": (model_path, str),
            "plate_conf": (plate_conf, float), "char_conf": (char_conf, float),
            "plate_iou": (plate_iou, float), "char_iou": (char_iou, float),
            "plate_imgsz": (plate_imgsz, tuple), "char_imgsz": (char_imgsz, tuple),
            "device": (device, (int, str)), "mapping_files": (mapping_files, dict)
        }
        for param, types in param_types.items():
            if not isinstance(types[0], types[1]):
                raise TypeError(f"{param} should be of type {types[1]}")

    def initialize_attributes(self, model_path, plate_conf, char_conf, plate_iou, char_iou, plate_imgsz, char_imgsz, device, mapping_files):
        self.model_path = model_path
        self.plate_conf = plate_conf
        self.char_conf = char_conf
        self.plate_iou = plate_iou
        self.char_iou = char_iou
        self.plate_imgsz = plate_imgsz
        self.char_imgsz = char_imgsz
        self.device = device
        self.char_classes = list(range(36))  # Assuming 36 character classes
        self.plate_classes = [36]
        self.id_to_name, self.eng_to_persian = self.load_mappings(mapping_files)

    @staticmethod
    def load_mappings(mapping_files):
        try:
            with open(mapping_files['id_to_name'], 'rb') as file:
                id_to_name = pickle.load(file)
            with open(mapping_files['eng_to_persian'], 'rb') as file:
                eng_to_persian = pickle.load(file)
            return id_to_name, eng_to_persian
        except Exception as e:
            print(f"Error loading mappings: {e}")
            sys.exit(1)

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
