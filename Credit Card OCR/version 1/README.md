# PersicaGlyph OCR Studio: Credit Card OCR System
This OCR system is tailored for extracting credit card information from images. It uses advanced image processing techniques to enhance the legibility of credit card numbers and employs OCR technology to read and interpret the card details.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Build and Test](#build-and-test)
- [Parameter Explanation](#parameter-explanation)
- [Demonstration](#demonstration)
- [Explore Our Kernel 🚀](#explore-our-kernel-)
- [Technology Stack](#technology-stack)
- [License](#license)
- [Contributing](#contributing)
- [Credits and Acknowledgements](#credits-and-acknowledgements)
- [Contact Information](#contact-information)

## Introduction
The Credit Card OCR System is a sophisticated tool that leverages Optical Character Recognition (OCR) to extract credit card information from images. Using a combination of image processing strategies and OCR, it can identify card numbers, expiration dates, CVV codes, and more with high accuracy. It's built with the ease of integration in mind, making it suitable for financial applications, automated form filling, and fraud prevention systems. It also uses image processing strategies such as image enhancement and bolding of numbers to prepare images for OCR and to improve the accuracy of text recognition.

## Features
- **OCR Technology**: Utilizes the `easyocr` library to recognize text within images.
- **Image Processing**: Employs strategies like number bolding and image enhancement to prepare images for OCR.
- **Information Extraction**: Extracts credit card details such as card number, IBAN, CVV2, expiry date, owner's name, and bank name.
- **Accuracy Enhancement**: Implements post-processing methods to correct common OCR errors.
- **Bank Identification**: Maps card prefixes to bank names for quick identification.
- **User-Selectable Image Processing Strategies**: Choose between `EnhanceImageStrategy` for overall image quality improvement or `MakeNumbersBolderStrategy` to increase the prominence of numbers for OCR.

### Explanation of Image Processing Strategies
- **EnhanceImageStrategy**: Improves the image's contrast and sharpness to aid OCR accuracy.
- **MakeNumbersBolderStrategy**: Thickens the numbers in the image, making them more distinguishable for the OCR process.


## Build and Test
To use the Credit Card OCR System, run the `main` script with the path to the image you want to process:

```bash
python main.py --image_path --strategy 
```

### Parameter Explanation
- `--image_path`: The path to the image file containing the credit card to be processed.
- `--strategy`: (Optional) Image processing strategy to use. Choose enhance for image enhancement or bold for making numbers bolder. Default is bold.

## Explanation of Source Code Components
- **OCRReader**: A singleton class that initializes the `easyocr` reader once and uses it to read text from images.
- **OCRDataProcessor**: Contains logic for parsing and structuring OCR results, mapping card numbers to bank names, and correcting OCR errors.
- **ImageProcessor**: A class that takes an image processing strategy as a parameter and applies it to the image.
- **EnhanceImageStrategy**: An image processing strategy that enhances the overall quality of the image, making it more suitable for OCR.
- **MakeNumbersBolderStrategy**: Specifically focuses on making numbers in the image bolder to improve OCR accuracy.
- **Main Script**: The entry point of the application, which uses `argparse` to accept an image path, processes the image, and then prints the extracted credit card information.

## Demonstration
The image showcases the robust detection capabilities of PersicaGlyphOCR. Our model is designed to handle a diverse array of license plate designs and formats, as evidenced by the multiple examples displayed. While the plates differ in background color, text style, and arrangement, our system can reliably identify and extract the plate region from the vehicle's image.

It's important to note that while the demonstration focuses on the character recognition aspect of the license plates, PersicaGlyphOCR is equally adept at detecting the plates themselves from a larger image of a vehicle. In these examples, we've chosen to highlight plates that represent a typical format the model can process. However, the underlying detection technology is capable of locating and reading plates of various sizes and styles, including those not depicted here.

![Demo](./Assets/Demo.png)

# Explore Our Kernel 🚀
We are thrilled to unveil our cutting-edge kernel, an embodiment of innovation that integrates the audio manipulation capabilities of VoxArte Studio! It's not just a repository; it's a revolution in audio processing, built with our audio projects at its heart.

## Catch the Wave of Audio Innovation
Don't miss out on this opportunity to be a part of the audio evolution. Click the link blow, star the repo for future updates, and let your ears be the judge. If you're as passionate about audio as we are, we look forward to seeing you there!

Remember, the future of audio is not just heard; it's shared and shaped by enthusiasts and professionals alike. Let's make waves together with VoxArte Studio and our Kernel. 🚀

🔗 [Kernel Repository](https://github.com/Meta-Intelligence-Services)

---

## Technology Stack
Before setting up Credit Card OCR, it is important to understand the role of each dependency in the system:

- **easyocr**: For reading text from images using pre-trained neural network models.
- **OpenCV (opencv-python)**: For image processing tasks such as image reading, converting color spaces, thresholding, and contour detection.
- **Numpy**: For efficient array operations, used in image manipulation.
- **singleton_decorator**: Ensures classes like `OCRReader` and image processing strategies are instantiated only once.
- **Argparse**: For parsing command-line options and arguments in the script.
---
Each of these dependencies plays a crucial role in ensuring Credit Card OCR operates effectively, from model loading and image processing to parsing user commands and ensuring efficient resource usage.


## License
Credit Card OCR is open-sourced under the MIT License. See [LICENSE](LICENSE) for more details.

## Contributing
While we deeply value community input and interest in Credit Card OCR, the project is currently in a phase where we're mapping out our next steps and are not accepting contributions just yet. We are incredibly grateful for your support and understanding. Please stay tuned for future updates when we'll be ready to welcome contributions with open arms.

## Contact Information
Although we're not open to contributions at the moment, your feedback and support are always welcome. Please feel free to star the project or share your thoughts through the Issues tab on GitHub, and we promise to consider them carefully.please [open an issue](https://github.com/Amir-Nassimi/PersicaGlyph-OCR-Studio/issues) in the Credit Card OCR repository, and we will assist you.