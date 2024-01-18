import os
import scipy.io
from PIL import Image
import numpy as np

def read_and_save_images(mat_file_path, output_dir):
    # Load the .mat file
    data = scipy.io.loadmat(mat_file_path)

    # Extract the images under the key 'coco_file'
    images = data['coco_file']

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process and save each image
    for i, img_cell in enumerate(images):
        # Extract the image from the cell
        img = img_cell[0]

        # Convert the image data to a format suitable for saving (if necessary)
        # Assuming the image is in the format (224, 224, 3)
        if img.shape != (224, 224, 3):
            raise ValueError(f"Image {i} is not in the expected format")

        # Convert the image data to uint8 if it's not already
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Create an Image object
        image = Image.fromarray(img)

        # Save the image
        image.save(os.path.join(output_dir, f"{i}.png"))

# Example usage
mat_file_path = './datasets/coco_file_224_shared.mat' 
output_dir = './datasets/coco'
read_and_save_images(mat_file_path, output_dir)
