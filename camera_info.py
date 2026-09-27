import cv2

image = cv2.imread("images/level/frame_0000.jpg")

if image is None:
    raise RuntimeError("Could not load image.")

height, width = image.shape[:2]

print(f"Image width:  {width}")
print(f"Image height: {height}")
