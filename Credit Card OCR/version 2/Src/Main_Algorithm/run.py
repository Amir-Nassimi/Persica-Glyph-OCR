import cv2
import argparse 

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[0]))

from ocr import OCRReader, OCRDataProcessor
from utils.pre_proc import ImageProcessor, MakeNumbersBolderStrategy

def main(img_path):
    # Create an instance of ImageProcessor with the desired strategy
    image_processor = ImageProcessor(MakeNumbersBolderStrategy())

    # Process an image
    try :
        img = image_processor.process(img_path)  # Use the provided image path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    except FileNotFoundError:
        print("Error: The provided image file does not exist.")
        return -1
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1
    
    # Read text using OCRReader (Singleton)
    ocr_reader = OCRReader()
    results = ocr_reader.read_text(img)

    # Process OCR data
    ocr_processor = OCRDataProcessor()
    groups = ocr_processor.group_and_sort_ocr_data(results)
    extracted_card_info = ocr_processor.extract_card_info(groups)

    print(extracted_card_info)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image for credit card information extraction.')
    
    # Add an argument to specify the image file path
    parser.add_argument('-pth', '--image_path', type=str, help='Path to the image file to process',required=True)
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the main function with the provided image path
    main(args.image_path)