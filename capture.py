import cv2
import os

# Create output directory
os.makedirs("images", exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

count = 0

print("Press S to save an image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Display frame
    display = frame.copy()
    cv2.putText(
        display,
        f"Images: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("3DGS Dataset Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = f"images/level/frame_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
