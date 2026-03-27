import numpy as np
import matplotlib.pyplot as plt

# Path to one of your preprocessed/cropped files
file_path = "/Volumes/pelvis/projects/tiago/DWI_IQA/data/preprocessed/Task001_ProstateQualityDWI/universal_classifier_plans_v1.0_stage0/ProstateQualityDWI_0002.npz"

# Load the .npz file
with np.load(file_path) as data:
    images = data['data']  # Adjust key if needed

# Check shape to understand layout (e.g., [channels, depth, height, width])
print("Image shape:", images.shape)

# Visualize a few slices
for i in range(min(20, images.shape[1])):  # limit slices to display
    plt.imshow(images[0, i], cmap="gray")  # Adjust channel index as necessary
    plt.title(f"Slice {i}")
    plt.axis("off")
    plt.show()
