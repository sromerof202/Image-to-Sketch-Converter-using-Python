import cv2
import numpy as np
from PIL import Image

# Load the image
image_path = "0.jpg"
img = cv2.imread(image_path)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Invert the grayscale image
inverted_img = cv2.bitwise_not(gray_img)

# Blur the inverted image
blurred_img = cv2.GaussianBlur(inverted_img, (11, 11), sigmaX=0, sigmaY=0)

# Invert the blurred image
inverted_blurred = cv2.bitwise_not(blurred_img)

# Create the pencil sketch image
sketch_img = cv2.divide(gray_img, inverted_blurred, scale=240.0)

# Enhance lines by morphological dilation( join together broken parts of an object.)
kernel = np.ones((1, 1), np.uint8)
sketch_img_dilated = cv2.dilate(sketch_img, kernel, iterations=5)

# Increase line darkness by reducing the max pixel values to less than 255
sketch_img_darker = np.minimum(
    254, sketch_img_dilated * (255.0 / sketch_img_dilated.max())
)

# Save the image with thicker and blacker lines
cv2.imwrite("/Users/User/Downloads/pencil_sketch_thicker_darker.png", sketch_img_darker)

# Convert to a PIL image and show the final output
sketch_pil_image_thicker_darker = Image.fromarray(sketch_img_darker.astype("uint8"))

sketch_pil_image_thicker_darker.show()
