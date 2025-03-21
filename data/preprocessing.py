import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

def load_ndjson(file_path, max_items=None):
    """Load data from NDJSON file"""
    data = []
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            count += 1
            if max_items and count >= max_items:
                break
    return data

def drawing_to_image(drawing, image_size=64):
    """Convert drawing to numpy image array"""
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    # Each drawing consists of multiple strokes
    for stroke in drawing:
        # Each stroke contains x, y coordinates
        for i in range(len(stroke[0]) - 1):
            x1, y1 = int(stroke[0][i] * image_size / 1024), int(stroke[1][i] * image_size / 1024)
            x2, y2 = int(stroke[0][i + 1] * image_size / 1024), int(stroke[1][i + 1] * image_size / 1024)
            cv2.line(image, (x1, y1), (x2, y2), 255, 1)

    return image

def get_local_categories(data_dir):
    """Get list of categories from local directory"""
    categories = []
    for file in os.listdir(data_dir):
        if file.endswith('.ndjson'):
            category = file.split('.')[0]
            categories.append(category)
    return categories

def process_categories(categories, data_dir, samples_per_category, image_size):
    """Process multiple categories of drawing data from local files"""
    X = []
    y = []

    for idx, category in enumerate(categories):
        file_path = os.path.join(data_dir, f"{category}.ndjson")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping category {category}.")
            continue
            
        # Load data
        print(f"Loading {category} data...")
        drawings = load_ndjson(file_path, max_items=samples_per_category)

        # Process drawings
        for drawing_data in tqdm(drawings, desc=f"Processing {category}"):
            if drawing_data.get('recognized', False):  # Use only recognized drawings
                image = drawing_to_image(drawing_data['drawing'], image_size)
                X.append(image)
                y.append(idx)  # Use category index as label

    # Convert to numpy arrays
    X = np.array(X).reshape(-1, image_size, image_size)
    y = np.array(y)

    return X, y

def split_data(X, y, test_size=0.2, val_size=0.5, random_state=42):
    """Split data into train, validation, and test sets while preserving class distribution"""
    # First split: train and temp (val+test)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, temp_idx in sss.split(X, y):
        X_train, X_temp = X[train_idx], X[temp_idx]
        y_train, y_temp = y[train_idx], y[temp_idx]

    # Second split: validation and test
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    for val_idx, test_idx in sss_val.split(X_temp, y_temp):
        X_val, X_test = X_temp[val_idx], X_temp[test_idx]
        y_val, y_test = y_temp[val_idx], y_temp[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_data(X_train, X_val, X_test):
    """Normalize pixel values to [0,1]"""
    return X_train / 255.0, X_val / 255.0, X_test / 255.0