import cv2
import argparse 

from OCR.main_model import OCRReader, OCRDataProcessor
from OCR.pre_proc import ImageProcessor, EnhanceImageStrategy, MakeNumbersBolderStrategy


def main(img_path, strategy_name):
    # Map strategy names to strategy classes
    strategies = {
        'enhance': EnhanceImageStrategy,
        'bold': MakeNumbersBolderStrategy
    }
    
    # Select the image processing strategy based on user input
    strategy = strategies.get(strategy_name, MakeNumbersBolderStrategy)()
    
    # Create an instance of ImageProcessor with the selected strategy
    image_processor = ImageProcessor(strategy)

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
    parser = argparse.ArgumentParser(description='Credit Card OCR System')

    parser.add_argument('--image_path', type=str, help='Path to the image file to process', required=True)
    parser.add_argument('--strategy', type=str, choices=['enhance', 'bold'], default='bold', help='Image processing strategy to use', required=False)
    args = parser.parse_args()
    
    main(args.image_path, args.strategy)
    