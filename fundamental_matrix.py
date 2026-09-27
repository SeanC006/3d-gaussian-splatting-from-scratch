import cv2
import numpy as np

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

# Match descriptors
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe ratio test
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Good matches: {len(good_matches)}")

# Need at least 8 points for fundamental matrix estimation
if len(good_matches) < 8:
    raise RuntimeError("Not enough matches to estimate fundamental matrix.")

# Extract matching pixel coordinates
pts1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_matches
])

pts2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_matches
])

# Estimate fundamental matrix using RANSAC
F, mask = cv2.findFundamentalMat(
    pts1,
    pts2,
    cv2.FM_RANSAC,
    1.0,
    0.99
)

if F is None:
    raise RuntimeError("Could not estimate fundamental matrix.")

# RANSAC mask:
# 1 = geometrically consistent match
# 0 = outlier
inlier_matches = [
    match
    for match, inlier in zip(good_matches, mask.ravel())
    if inlier
]

print(f"Geometric inliers: {len(inlier_matches)}")
print(f"Geometric outliers: {len(good_matches) - len(inlier_matches)}")

# Draw only geometrically valid matches
match_image = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imshow("Geometrically Valid Matches", match_image)

print("Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
