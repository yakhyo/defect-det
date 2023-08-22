import torch
from torchvision import transforms
from PIL import Image
from models import UNet
import numpy as np
from inference import resize

model = UNet(in_channels=3, out_channels=7)

# Load the pre-trained model
ckpt = torch.load("./weights/base_best_focal.pt")
model.load_state_dict(ckpt["model"].float().state_dict())
model.eval()  # Set the model to evaluation mode

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet values
])

# Load and preprocess the input image
input_image = Image.open("./data/test/images/122021417103241-37_5_side2.jpg")
gt_mask = Image.open("./data/test/masks/122021417103241-37_5_side2.png")
if ckpt.get("use_crop", False):
    input_image = input_image.crop((840, 512, 1640, 1312))
    gt_mask = gt_mask.crop((840, 512, 1640, 1312))
input_image = resize(input_image)
gt_mask = resize(gt_mask)

input_tensor = transform(input_image)
input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Get the predicted class labels (assuming the model output is logits)
predicted_labels = torch.argmax(output, dim=1)

# Convert the predicted labels to a color-coded segmentation mask
# You might need a mapping between class indices and corresponding colors
# Then, use this mapping to assign colors to each pixel based on predicted class
# Finally, save or display the color-coded mask

# Example mapping (adjust this according to your classes and color preferences)
class_colors = {
    0: [0, 0, 0],  # BACKGROUND
    1: [255, 0, 0],  # RED
    2: [0, 255, 0],  # GOLD
    3: [0, 0, 255],  # GLUE
    4: [128, 128, 128],  # STABBED
    5: [128, 0, 0],  # CLAMP
    6: [0, 0, 128],  # GREY
}

# Convert class indices to color values
color_coded_mask_colored = np.zeros((3, predicted_labels.shape[1], predicted_labels.shape[2]), dtype=np.uint8)
for class_idx, color in class_colors.items():
    mask = (predicted_labels[0] == class_idx).numpy()  # Convert the mask tensor to a numpy array
    color_coded_mask_colored[:, mask] = np.array(color).reshape(3, 1)  # Reshape the color to match the mask dimensions

# Create a PIL Image from the colored mask array
color_coded_mask_image = Image.fromarray(color_coded_mask_colored.transpose(1, 2, 0))

# Save or display the color-coded mask image
color_coded_mask_image.show()


# Define the alpha value for overlay (adjust as needed)
alpha = 0.5

# Overlay the color-coded mask on the input image
overlay = Image.blend(input_image, color_coded_mask_image, alpha)

# Save or display the overlay image
# overlay.show()

import matplotlib.pyplot as plt
# Display the images in a single row using matplotlib
plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

# Input Image
plt.subplot(1, 4, 1)
plt.imshow(input_image)
plt.title("Input Image")
plt.axis("off")

# Color-Coded Mask
plt.subplot(1, 4, 2)
plt.imshow(color_coded_mask_image)
plt.title("Predicted Mask")
plt.axis("off")

# Overlay Image
plt.subplot(1, 4, 3)
plt.imshow(overlay)
plt.title("Overlay Image")
plt.axis("off")

# Overlay Image
plt.subplot(1, 4, 4)
plt.imshow(gt_mask)
plt.title("GT mask")
plt.axis("off")

plt.tight_layout()
plt.show()