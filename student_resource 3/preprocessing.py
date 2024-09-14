import os
import re
import pytesseract
import pandas as pd
from PIL import Image
from fuzzywuzzy import process
import matplotlib.pyplot as plt

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


def process_images_in_folder(folder_path, csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Dictionary to map image URLs to actual values
    actual_values = {os.path.basename(row['image_link']): row['entity_value'] for _, row in df.iterrows()}

    # List all images in the directory
    image_files = [f for f in os.listdir(folder_path) if
                   os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Open the image
        image = Image.open(image_path)

        # Save the image to a temporary file
        temp_image_path = os.path.join(folder_path, 'temp_image.jpg')
        image.save(temp_image_path)

        # Display the image
        img = Image.open(temp_image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(temp_image_path)  # Save the image plot
        plt.close()  # Close the plot to avoid overlapping images

        # Extract text and process
        extracted_text = pytesseract.image_to_string(image)
        cleaned_text = clean_text(extracted_text)
        standardized_text = standardize_units(cleaned_text, short_form_map)
        extracted_units = extract_units(standardized_text)
        mapped_units = map_units(extracted_units, allowed_units)
        formatted_results = format_output(mapped_units)

        # Get actual value for comparison
        actual_value = actual_values.get(image_file, "Not Found")

        # Print results
        print(f"Processed image: {image_file}")
        print("Extracted and mapped units:")
        print(formatted_results)
        print(f"Actual value: {actual_value}")

        # Wait for user input
        try:
            input("Press Enter to continue to the next image...")
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            break

        # Clean up temporary files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# Example usage
folder_path = './dataset/train_images'
csv_file = './dataset/train.csv'
process_images_in_folder(folder_path, csv_file)
