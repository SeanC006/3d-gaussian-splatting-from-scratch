import cv2

# Load two images
img1 = cv2.imread("images/level/frame_0000.jpg")
img2 = cv2.imread("images/level/frame_0010.jpg")

if img1 is None or img2 is None:
    raise RuntimeError("Could not load one or both images.")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect SIFT features
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"Image 1 keypoints: {len(kp1)}")
print(f"Image 2 keypoints: {len(kp2)}")

# Match descriptors using Brute Force Matcher
bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Total candidate matches: {len(matches)}")
print(f"Good matches: {len(good_matches)}")

# Draw the good matches
match_image = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imshow("SIFT Feature Matches", match_image)

print("Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
