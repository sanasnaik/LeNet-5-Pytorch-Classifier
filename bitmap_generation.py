# generate_bitmaps.py

import os
from PIL import Image
import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

def get_transform(image_size=(7, 12)):
    # Returns a transform to grayscale, resize, and convert image to tensor
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

def load_digit_images(folder_path, transform):
    # Loads and transforms all .png images in a folder
    bitmaps = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img_tensor = transform(img).squeeze(0)  # Shape: [H, W]
            bitmaps.append(img_tensor)
    return bitmaps

def compute_average_bitmap(bitmaps):
    # Returns the flattened average of a list of image tensors
    if not bitmaps:
        return None
    stacked = torch.stack(bitmaps)
    avg = stacked.mean(dim=0)
    return avg.flatten()

def generate_bitmap_prototypes(root_dir, image_size=(7, 12)):
    # Generates a prototype (average bitmap) for each digit 0–9
    transform = get_transform(image_size)
    height, width = image_size
    prototypes = torch.zeros((10, height * width))

    for digit in range(10):
        folder_path = os.path.join(root_dir, str(digit))
        bitmaps = load_digit_images(folder_path, transform)
        avg_bitmap = compute_average_bitmap(bitmaps)
        if avg_bitmap is not None:
            prototypes[digit] = avg_bitmap

    return prototypes


def visualize_prototypes(prototypes, image_size=(7, 12)):
    # Displays the bitmap prototypes for digits 0–9
    num_digits = prototypes.size(0)
    plt.figure(figsize=(10, 4))
    for i in range(num_digits):
        plt.subplot(2, 5, i + 1)
        plt.imshow(prototypes[i].reshape(image_size), cmap='gray')
        plt.title(f"Digit {i}")
        plt.axis("off")
    plt.suptitle("Bitmap Prototypes")
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(__file__), "digits updated")
    prototypes = generate_bitmap_prototypes(root_dir)
    print("Bitmap prototypes generated:", prototypes.shape)  # [10, 84]
    visualize_prototypes(prototypes, image_size=(7, 12))
