import cv2
import os

# Load one of your images
image_path = "images/level/frame_0000.jpg"

image = cv2.imread(image_path)

if image is None:
    raise RuntimeError(f"Could not load {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect features and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

print(f"Image: {image_path}")
print(f"Number of keypoints: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape}")

# Draw the detected features
output = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("SIFT Features", output)

print("Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
