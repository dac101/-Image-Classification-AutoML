import json
import os
import numpy as np
import pandas as pd
from supervised.automl import AutoML
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Save category data to individual JSON files
def save_category_to_json(category_data, category_labels, category_name, results_dir):
    """
    Save processed data for a single category to a JSON file.
    Args:
        category_data (pd.DataFrame): Features for the category.
        category_labels (pd.Series): Labels for the category.
        category_name (str): Name of the category.
        results_dir (str): Directory to save the JSON file.
    """
    try:
        print(f"Saving category {category_name} to JSON...")
        ensure_dir_exists(results_dir)  # Ensure the directory exists

        file_path = os.path.join(results_dir, f"{category_name}_data.json")

        # Combine data and labels
        data = category_data.to_dict(orient='records')
        for idx, record in enumerate(data):
            record['target'] = category_labels.iloc[idx]

        # Write to JSON file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Successfully saved category {category_name} to {file_path}")
    except Exception as e:
        print(f"Error saving category {category_name} to JSON: {e}")


# Function to check if a folder exists
def check_folder_exists(folder_path):
    """
    Check if a folder exists and raise an error if it does not.
    Args:
        folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    print(f"Folder found: {folder_path}")


# Function to preprocess images and extract features
def process_images(image_dir, label=None, image_size=(24, 24)):
    """
    Extract image features from a directory using Pillow and return them as a DataFrame.
    Args:
        image_dir (str): Directory containing image files.
        label (str): Optional label for the images (used for training).
        image_size (tuple): Target size for scaling images (width, height).
    Returns:
        pd.DataFrame: A DataFrame with extracted features and the label (if provided).
    """
    data = []
    labels = []
    files = os.listdir(image_dir)
    total_files = len(files)
    print(f"Processing {total_files} images in category: {label}")

    # Process up to 100 images
    for idx, file in enumerate(files[:10]):
        file_path = os.path.join(image_dir, file)
        if file.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            try:
                # Load the image using Pillow
                with Image.open(file_path) as img:
                    # Convert to RGB (to handle grayscale or inconsistent formats)
                    img = img.convert("RGB")
                    # Resize the image
                    img_resized = img.resize(image_size)
                    # Flatten the image into a 1D array
                    img_array = np.array(img_resized).flatten()
                    data.append(img_array)
                    if label is not None:
                        labels.append(label)

                # Print progress every 10 images
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{total_files} images")
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")

    print(f"Finished processing category: {label}")
    if label is not None:
        return pd.DataFrame(data), pd.Series(labels)
    else:
        return pd.DataFrame(data)


# Function to preprocess validation/test folders
def process_folder(image_dir, image_size=(24, 24)):
    """
    Preprocess images from a folder with subdirectories for each label.
    Args:
        image_dir (str): Root directory containing subfolders (labels).
        image_size (tuple): Target size for scaling images (width, height).
    Returns:
        pd.DataFrame, pd.Series: Features and corresponding labels.
    """
    data = []
    labels = []
    check_folder_exists(image_dir)
    categories = os.listdir(image_dir)
    print(f"Found {len(categories)} categories in {image_dir}")
    for category in categories:  # Subfolders as categories
        category_dir = os.path.join(image_dir, category)
        if os.path.isdir(category_dir):
            print(f"Processing category: {category}")
            category_data, category_labels = process_images(category_dir, label=category, image_size=image_size)
            data.extend(category_data.values.tolist())
            labels.extend(category_labels.tolist())
    return pd.DataFrame(data), pd.Series(labels)


# Function to combine all JSON files into one dataset
def combine_json_files(folder_path, combined_file_dir, combined_file_name="combined_data.json"):
    """
    Combine all JSON files in a folder into one dataset.
    Args:
        folder_path (str): Path to the folder containing JSON files.
        combined_file_dir (str): Directory to save the combined file.
        combined_file_name (str): Name of the combined JSON file.
    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    os.makedirs(combined_file_dir, exist_ok=True)
    combined_features = []
    combined_labels = []

    # Iterate through JSON files in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith('.json'):
            print(f"Loading file: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract features and labels
                features = [list(record.values())[:-1] for record in data]  # All except the target
                labels = [record['target'] for record in data]
                combined_features.append(np.array(features))
                combined_labels.append(np.array(labels))

    # Combine all loaded arrays
    combined_features = np.vstack(combined_features)
    combined_labels = np.concatenate(combined_labels)

    # Convert back to pandas DataFrame
    combined_data = pd.DataFrame(combined_features)
    combined_data["target"] = combined_labels

    # Save the combined data to JSON
    combined_file_path = os.path.join(combined_file_dir, combined_file_name)
    combined_data_dict = combined_data.to_dict(orient='records')
    with open(combined_file_path, 'w') as f:
        json.dump(combined_data_dict, f, indent=4)

    print(f"Combined data saved to {combined_file_path}")
    return combined_data


# Ensure results directory exists
def ensure_dir_exists(directory):
    """
    Ensure a directory exists. Create it if it doesn't.
    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def clean_directory(directory_path):
    """
    Deletes all files in the specified directory.
    Args:
        directory_path (str): Path to the directory to clean.
    """
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


# Paths to datasets
train_dir = "dataset/Train/Train"
validation_dir = "dataset/Validation/Validation"
test_dir = "dataset/Test/Test"
converted_dir = "training_data_converted"
results_dir = "AutoML_results"
combined_file_dir = "combined_data_dir"
combined_file_name = "final_combined_data.json"


# Ensure directories exist and clean them
clean_directory(converted_dir)
clean_directory(results_dir)
clean_directory(combined_file_dir)


# Ensure directories exist
os.makedirs(converted_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Check if necessary directories exist
check_folder_exists(train_dir)
check_folder_exists(validation_dir)
check_folder_exists(test_dir)

label_encoder = LabelEncoder()

# ====================
# Step 1: Training
# ====================


# Define the directory paths
train_dir = "dataset/Train/Train"
validation_dir = "dataset/Validation/Validation"
test_dir = "dataset/Test/Test"

# Define image transformations (resize, normalize, etc.)
image_transforms = transforms.Compose([
    transforms.Resize((24, 24)),  # Resize images to 224x224 for consistency
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load the datasets using ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir, transform=image_transforms)
validation_dataset = datasets.ImageFolder(root=validation_dir, transform=image_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=image_transforms)

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Verify the data
class_names = train_dataset.classes  # Get the class labels
print(f"Class labels: {class_names}")

# Extract data as NumPy arrays
# Train data
# Extract images and labels as NumPy arrays
def extract_data_as_numpy(data_loader):
    """
    Extract images and labels from a DataLoader as NumPy arrays.
    Args:
        data_loader (DataLoader): PyTorch DataLoader.
    Returns:
        images_np (numpy.ndarray): Images as NumPy array.
        labels_np (numpy.ndarray): Labels as NumPy array.
    """
    all_images = []
    all_labels = []

    for images, labels in data_loader:
        all_images.append(images.numpy())  # Convert images to NumPy array
        all_labels.append(labels.numpy())  # Convert labels to NumPy array

    images_np = np.concatenate(all_images, axis=0)  # Combine all batches
    labels_np = np.concatenate(all_labels, axis=0)
    return images_np, labels_np


# ====================
# Step 1: Training and Testing
# ====================
# Extract train, validation, and test data as NumPy arrays
train_images_np, train_labels_np = extract_data_as_numpy(train_loader)
validation_images_np, validation_labels_np = extract_data_as_numpy(validation_loader)
test_images_np, test_labels_np = extract_data_as_numpy(test_loader)

# Reshape and normalize images
train_images_flat = train_images_np.reshape(train_images_np.shape[0], -1)  # Flatten
validation_images_flat = validation_images_np.reshape(validation_images_np.shape[0], -1)
test_images_flat = test_images_np.reshape(test_images_np.shape[0], -1)

# Save the preprocessed data for future use
np.save('train_images.npy', train_images_flat)
np.save('train_labels.npy', train_labels_np)
np.save('validation_images.npy', validation_images_flat)
np.save('validation_labels.npy', validation_labels_np)
np.save('test_images.npy', test_images_flat)
np.save('test_labels.npy', test_labels_np)

# Print dataset details
print(f"Train images shape: {train_images_flat.shape}")  # e.g., (num_samples, 3 * 224 * 224)
print(f"Train labels shape: {train_labels_np.shape}")
print(f"Validation images shape: {validation_images_flat.shape}")
print(f"Validation labels shape: {validation_labels_np.shape}")
print(f"Test images shape: {test_images_flat.shape}")
print(f"Test labels shape: {test_labels_np.shape}")

# Example: Verify a batch
for images, labels in train_loader:
    print(f"Batch image tensor shape: {images.shape}")  # Should be [64, 3, 224, 224]
    print(f"Batch label tensor: {labels}")  # Class labels for the batch

automl = AutoML(mode="Perform", results_path=results_dir)
automl.fit(train_images_flat, train_labels_np)  # Pass train images and labels


# ====================
# Step 3: Validation
# ====================

# Evaluate on validation data (optional)
validation_predictions = automl.predict(validation_images_flat)

# Evaluate performance
from sklearn.metrics import classification_report, accuracy_score

print("\nValidation Results:")
print(classification_report(validation_labels_np, validation_predictions))
print(f"Validation Accuracy: {accuracy_score(validation_labels_np, validation_predictions):.2f}")

# Optionally save predictions
np.save('validation_predictions.npy', validation_predictions)


# ====================
# Step 5: Validation
# ====================


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load validation labels and predictions
validation_labels_np = np.load('validation_labels.npy')  # If saved earlier
validation_predictions = np.load('validation_predictions.npy')  # If saved earlier

# Confusion Matrix
conf_matrix = confusion_matrix(validation_labels_np, validation_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
disp.plot(cmap='viridis', xticks_rotation=45, ax=plt.gca())
plt.title('Confusion Matrix')
plt.show()

# Classification Report Metrics
from sklearn.metrics import classification_report
report = classification_report(validation_labels_np, validation_predictions, target_names=class_names, output_dict=True)

# Plot Precision, Recall, and F1-Score
metrics = ['precision', 'recall', 'f1-score']
metric_data = {metric: [report[class_name][metric] for class_name in class_names] for metric in metrics}

plt.figure(figsize=(12, 8))
for metric, values in metric_data.items():
    plt.plot(class_names, values, marker='o', label=metric)
plt.title('Classification Metrics (Precision, Recall, F1-score)')
plt.xlabel('Class')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Overall Accuracy Bar Chart
accuracy = accuracy_score(validation_labels_np, validation_predictions)
plt.figure(figsize=(6, 4))
plt.bar(['Validation Accuracy'], [accuracy], color='blue', alpha=0.7)
plt.title('Overall Validation Accuracy')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()