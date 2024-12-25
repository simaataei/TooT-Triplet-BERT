import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Directory containing the PNG files
image_dir = 'attention/P06703'

# Number of columns and rows
cols = 5
rows = 6

image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

# Number of images
num_images = len(image_files)

# Define the number of rows and columns for the grid

# Create a figure to hold the subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through each image and add it to the corresponding subplot
for i, image_file in enumerate(image_files):
    img = Image.open(image_file)
    axes[i].imshow(img)
    axes[i].axis('off')  # Hide the axis

# Hide any remaining empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust the layout
plt.tight_layout()

# Save the final figure as a PNG file
output_path = './attention/combined_images.png'
plt.savefig(output_path, dpi=300)

# Close the plot to free up memory
plt.close()

print(f'Saved the final image grid to {output_path}')
