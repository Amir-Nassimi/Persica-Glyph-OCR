import re
import easyocr
import numpy as np
from singleton_decorator import singleton

# OCR Reader Singleton
@singleton
class OCRReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en', 'fa'], gpu=True)

    def read_text(self, image):
        return self.reader.readtext(np.array(image), paragraph=False)


# OCR Data Processing
class OCRDataProcessor:
    def __init__(self):
        self.bank_prefixes = {
            "6219": "Saman",
            "6037": "Melli",
            "5892": "Sepah",
            "5022": "Pasargad",
            "6274": "Eghtesad Novin",
            "6395": "Ghavamin",
            "6221": "Parsian",
            "6362": "Ayande",
            "5894": "Refah",
            "6104": "Melat"
        }

        self.cv2_list = [
            'CV2', 'vv2', 'Cvv2', 'CVV2', 'cvv2', 'cv2', 'cw2', 'Cw2', 'cw', 'CV/2', 'C"v2'
        ]

    @staticmethod
    def correct_ocr_errors(text):
        return text.replace('1R', 'IR').replace('CV2:', 'CVV2:')

    @staticmethod
    def group_and_sort_ocr_data(ocr_data, y_threshold=60):
        grouped_data = {}
        for item in ocr_data:
            bbox, _, _ = item
            y_coord = bbox[0][1]
            found = False
            for key in grouped_data:
                if abs(key - y_coord) <= y_threshold:
                    grouped_data[key].append(item)
                    found = True
                    break
            if not found:
                grouped_data[y_coord] = [item]

        for key in grouped_data:
            grouped_data[key] = sorted(grouped_data[key], key=lambda x: x[0][0][0])

        return grouped_data

    def extract_card_info(self, grouped_ocr_results):
        extracted_info = {
            'card_number': [],
            'iban': '',
            'cvv2': '',
            'expiry_date': '',
            'owner_name': '',
            'bank_name': ''
        }
        
        temp_num = []
        for group in grouped_ocr_results.values():
            temp_card = ''
            for _, text, _ in group:
                text = text.replace(' ',"")
                
                pattern = r'[٠-٩]'
                text = re.sub(pattern, 'v', text)
                
                if 'R' in text and re.search(r'\d+', text):
    #             if 'R' in text and len(re.sub(r'\D', '', text)) >= 24 and re.search(r'\d+', text):
                    extracted_info['iban'] = re.sub(r'\s+', '', text)[:26]  # IBAN length is 26 characters
                    extracted_info['iban'] = extracted_info['iban'].replace('1R', 'IR').replace('&','8').replace('{','0').replace('"','0').replace('|','')
                
                if (len(text) == 4 or len(text) == 8 or len(text) == 12 or len(text) >= 16) and not(':' in text):
                    if text.isdigit(): temp_card += text.replace(' ', "")
                
                elif re.match(r'^\D*(\d\D*){3}$', text) and text.isdigit():
                    extracted_info['cvv2'] = text
                
                elif any(keyword in text for keyword in self.cv2_list):
                    found_keyword = next((keyword for keyword in self.cv2_list if keyword in text), None)

                    if found_keyword is not None:
                        cvv2 = text.replace(found_keyword, "").replace(':', "").replace(' ', "")
  
                        if cvv2.isdigit(): extracted_info['cvv2'] = cvv2
                        
                elif re.match(r'^(\d{4}/\d{2}|\d{2}/\d{2})$', text):
                    extracted_info['expiry_date'] = text
                    
                elif re.match(r'^[\D]+$', text) and not any(text.startswith(prefix) for prefix in self.bank_prefixes.keys()):
                    extracted_info['owner_name'] = text
                    
            temp_num.append(temp_card)
        
        extract_digits = lambda s: ''.join(re.findall(r'\d', s))
        temp_num = list(map(extract_digits, temp_num))
        
        try:
            extracted_info['card_number'] = list(filter(lambda x: len(x) == 16 and x.isdigit(), temp_num))[0]
            extracted_info['bank_name'] = self.bank_prefixes[extracted_info['card_number'][:4]]
        except IndexError: 
            extracted_info['card_number'] = ''.join([num.zfill(4) for num in temp_num if num.isdigit()])
            extracted_info['card_number'] = extracted_info['card_number'][:16]
            
            try: extracted_info['bank_name'] = self.bank_prefixes[extracted_info['card_number'][:4]]
            except IndexError: extracted_info['card_number'] = ''
        
    #     if len(extracted_info['card_number']) != 16: 
    #         extracted_info['card_number'] = ''
    #         extracted_info['bank_name'] = ''
        
        if extracted_info['cvv2'] == '':
            try: extracted_info['cvv2'] = list(filter(lambda x: (len(x) == 4 or len(x) == 3) and x.isdigit(), temp_num))[0]
            except IndexError: pass
            
        return extracted_info