from torch import argmax as torch_argmax
from ultralytics import YOLO
import numpy as np
from ocrPlate.Src.Utils.postprocessing import working_with_results



class OCRModel:
    def __init__(self,
                 model_path: str,
                 plate_conf: float = 0.6,
                 char_conf: float = 0.5,
                 plate_iou: float = 0.7,
                 char_iou: float = 0.7,
                 plate_imgsz: tuple = (640, 640),
                 char_imgsz: tuple = (320, 320),
                 device='cpu'):

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

        try:
            # Check the type of model_path
            if not isinstance(model_path, str):
                raise TypeError("model_path should be a string")

            # Check the types of confidence and IOU parameters
            if not all(isinstance(item, float) for item in [plate_conf, char_conf, plate_iou, char_iou]):
                raise TypeError("Confidence and IOU parameters should be floats")

            # Check the type of imgsz parameters
            if not all(isinstance(item, tuple) for item in [plate_imgsz, char_imgsz]):
                raise TypeError("imgsz should be an integer")

            # Check the type of device
            if (not isinstance(device, int)) and (not isinstance(device, str)):
                raise TypeError(
                    "device should be an integer for cuda or a string for cpu. device=0 for cuda and device='cpu' for cpu")

            self.model_path = model_path
            self.plate_conf = plate_conf
            self.char_conf = char_conf
            self.plate_iou = plate_iou
            self.char_iou = char_iou
            self.plate_imgsz = plate_imgsz
            self.char_imgsz = char_imgsz
            self.device = device

        except TypeError as e:
            # Handle the exception by printing an error message or taking appropriate action
            print(f"Initialization error: {e}")

        self.char_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                             34, 35]
        self.plate_classes = [36]

        self.id_to_name = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
            10: "b", 11: "j", 12: "dal", 13: "sin", 14: "sad", 15: "ta", 16: "gh", 17: "l",
            18: "m", 19: "v", 20: "h", 21: "n", 22: "y", 23: "a", 24: "p", 25: "t", 26: "se",
            27: "z", 28: "zh", 29: "sh", 30: "ein", 31: "f", 32: "k", 33: "g", 34: "D", 35: "S",
            36: "plate_area"
        }

        self.id_to_persian_name = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
            10: "ب", 11: "ج", 12: "د", 13: "س", 14: "ص", 15: "ط", 16: "ق", 17: "ل",
            18: "م", 19: "و", 20: "ه", 21: "ن", 22: "ی", 23: "الف", 24: "پ", 25: "ت", 26: "ث",
            27: "ز", 28: "ژ (معلولین و جانبازان)", 29: "ش", 30: "ع", 31: "ف", 32: "ک", 33: "گ", 34: "D", 35: "S",
            36: "plate_area"
        }

        self.eng_to_presian = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
                               'b': 'ب', 'j': 'ج', 'dal': 'د', 'sin': 'س', 'sad': 'ص', 'ta': 'ط',
                               'gh': 'ق', 'l': 'ل', 'm': 'م', 'v': 'و', 'h': 'ه', 'n': 'ن', 'y': 'ی',
                               'a': 'الف', 'p': 'پ', 't': 'ت', 'se': 'ث', 'z': 'ز', 'zh': 'ژ (معلولین و جانبازان)',
                               'sh': 'ش', 'ein': 'ع', 'f': 'ف', 'k': 'ک', 'g': 'گ', 'D': 'D', 'S': 'S',
                               '*': '*', '-': '-'}

        self.ocr_model = self.load_model()

    def load_model(self):
        """
        Method to load the YOLO model specified by the model_path.

        Returns:
            YOLO: The loaded YOLO model.
        """

        try:
            if self.device == 0:
                d = 'cuda'
            else:
                d = 'cpu'

            model = YOLO(self.model_path).to(d)
            return model
        except Exception as e:
            # Handle the exception here (e.g., print an error message or take appropriate action)
            print(f"Error loading the model: {str(e)}")

    def detect_plate(self, img):
        """
        Method to detect the license plate in an input image.

        Args:
            img (numpy.ndarray): Input image for license plate detection.

        Updates the self.plate attribute with the detected license plate image (numpy.ndarray).
        """

        try:
            results = self.ocr_model.predict(source=img,
                                             conf=self.plate_conf,
                                             iou=self.plate_iou,
                                             imgsz=self.plate_imgsz,
                                             device=self.device,
                                             classes=self.plate_classes,
                                             verbose=False)

            for r in results:
                confs = r.boxes.conf
                if len(confs) == 1:
                    coordination = r.boxes.xyxy
                    for i, c in enumerate(coordination):
                        x1 = int(c[0])
                        y1 = int(c[1])
                        x2 = int(c[2])
                        y2 = int(c[3])

                    self.plate = img[y1:y2, x1:x2]

                elif len(confs) > 1:
                    argmax_conf = torch_argmax(confs)
                    coordination = r.boxes.xyxy[argmax_conf].unsqueeze(0)
                    for i, c in enumerate(coordination):
                        x1 = int(c[0])
                        y1 = int(c[1])
                        x2 = int(c[2])
                        y2 = int(c[3])

                    self.plate = img[y1:y2, x1:x2]

                elif len(confs) == 0:
                    self.plate = np.array([None])


        except Exception as e:
            # Handle the exception here (e.g., print an error message or take appropriate action)
            print(f"Error detecting plate: {str(e)}")

    def enhance_plate(self):
        """
        Placeholder method for enhancing the detected license plate image.
        This method is currently not implemented and marked as pass.
        """

        # self.improved_plate = ...
        pass


    def detect_character(self, img):
        """
        Method to detect individual characters on the license plate.

        Args:
            img (numpy.ndarray): Input image containing the license plate.

        Uses the YOLO model to detect characters and arranges them in the correct order.
        Handles cases where characters may be missing and inserts missing characters using handle_missed_character.

        Returns:
            list: a list of detected characters on the license plate.
        """
        try:
            detected_car = False
            self.detect_plate(img)
            # self.plate = np.array([1])

            if (self.plate != None).all():

                detected_car = True

                # self.enhance_plate()

                results = self.ocr_model.predict(source=self.plate,
                                                 conf=self.char_conf,
                                                 iou=self.char_iou,
                                                 imgsz=self.char_imgsz,
                                                 device=self.device,
                                                 classes=self.char_classes,
                                                 verbose=False)

                detection_list, median_conf = working_with_results(results, self.id_to_persian_name)
                return detection_list, median_conf, detected_car

            else:
                detection_list = [None, None, None, "-", None]
                median_conf = None
                print("Plate is not detected!")
                return detection_list, median_conf, detected_car

        except Exception as e:
            # Handle the exception here (e.g., print an error message or take appropriate action)
            print(f"Error in detect_character: {str(e)}")