import os
import re
import cv2
import numpy as np
from PIL import Image
import pytesseract
from fuzzywuzzy import process

# Configuration for OCR and unit mapping
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

# Short form to full form mapping
short_form_map = {
    'cm': 'centimetre',
    'm': 'metre',
    'mm': 'millimetre',
    'kg': 'kilogram',
    'g': 'gram',
    'l': 'litre',
    'ft': 'foot',
    'in': 'inch',
    'oz': 'ounce',
    'lb': 'pound',
    'v': 'volt',
    'kw': 'kilowatt',
    'cubic ft': 'cubic foot',
    'cubic in': 'cubic inch',
    'fl oz': 'fluid ounce',
    'gal': 'gallon',
    'imp gal': 'imperial gallon',
    'pt': 'pint',
    'qt': 'quart'
}


# Functions
def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s.,-]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def standardize_units(text, short_form_map):
    for short_form, full_form in short_form_map.items():
        text = re.sub(r'\b{}\b'.format(short_form), full_form, text, flags=re.IGNORECASE)
    return text


def extract_units(text):
    pattern = r'(\d+\.?\d*)\s*([a-zA-Z\s]+)'
    matches = re.findall(pattern, text)
    return matches


def get_best_unit_match(unit, allowed_units):
    best_match, score = process.extractOne(unit.strip(), allowed_units)
    if score > 70:
        return best_match
    return None


def map_units(extracted_units, allowed_units):
    results = []
    for number, unit in extracted_units:
        standardized_unit = get_best_unit_match(unit, allowed_units)
        if standardized_unit:
            results.append(f"{number} {standardized_unit}")
    return results


def format_output(mapped_units):
    return [f"{item}" for item in mapped_units]


def show_image(image):
    # Convert PIL image to OpenCV format
    cv_image = np.array(image)
    cv_image = cv_image[:, :, ::-1]  # Convert RGB to BGR

    # Resize the image to a medium size (e.g., width=800)
    height, width = cv_image.shape[:2]
    new_width = 800
    new_height = int((new_width / width) * height)
    resized_image = cv2.resize(cv_image, (new_width, new_height))

    # Display the resized image
    cv2.imshow("Image Viewer", resized_image)
    cv2.waitKey(5000)  # Display for 5 seconds
    cv2.destroyWindow("Image Viewer")


# Main processing
def process_images_in_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if
                   os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        # Extract text and process it
        extracted_text = pytesseract.image_to_string(image)
        cleaned_text = clean_text(extracted_text)
        standardized_text = standardize_units(cleaned_text, short_form_map)
        extracted_units = extract_units(standardized_text)
        mapped_units = map_units(extracted_units, allowed_units)
        formatted_results = format_output(mapped_units)

        # Print the details of the extracted text
        print(f"Processed image: {image_file}")
        # print("Extracted text:")
        # print(extracted_text)
        # print("Cleaned text:")
        # print(cleaned_text)
        # print("Standardized text:")
        # print(standardized_text)
        print("Extracted units:")
        print(extracted_units)
        print("Mapped units:")
        print(formatted_results)

        # Display the image
        show_image(image)


# Example usage
folder_path = './dataset/train_images'
process_images_in_folder(folder_path)
