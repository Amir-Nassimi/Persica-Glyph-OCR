import cv2
import numpy as np
from singleton_decorator import singleton


# Image Processing Strategies
class ImageProcessor:
    def __init__(self, strategy):
        self.strategy = strategy

    def process(self, image_path):
        return self.strategy.process(image_path)

@singleton
class EnhanceImageStrategy:
    @staticmethod
    def process(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        merged = cv2.merge([y, u, v])
        result = cv2.cvtColor(merged, cv2.COLOR_YUV2BGR)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        result = cv2.filter2D(result, -1, sharpen_kernel)
        return result

@singleton
class MakeNumbersBolderStrategy:
    @staticmethod
    def process(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 0), 3)
        return image