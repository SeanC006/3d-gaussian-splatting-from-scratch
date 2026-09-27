import cv2
import numpy as np

# --------------------------------------------------
# Load images
# --------------------------------------------------

img1 = cv2.imread("images/level/frame_0000.jpg")
img2 = cv2.imread("images/level/frame_0010.jpg")

if img1 is None or img2 is None:
    raise RuntimeError("Could not load one or both images.")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# Detect SIFT features
# --------------------------------------------------

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# --------------------------------------------------
# Match descriptors
# --------------------------------------------------

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Good matches: {len(good_matches)}")

# --------------------------------------------------
# Extract matched pixel coordinates
# --------------------------------------------------

pts1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_matches
])

pts2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_matches
])

# --------------------------------------------------
# Camera intrinsics
# --------------------------------------------------

fx = 640
fy = 640
cx = 320
cy = 240

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)

print("\nCamera matrix K:")
print(K)

# --------------------------------------------------
# Estimate essential matrix
# --------------------------------------------------

E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None:
    raise RuntimeError("Could not estimate essential matrix.")

# Keep only inliers
inlier_pts1 = pts1[mask.ravel() == 1]
inlier_pts2 = pts2[mask.ravel() == 1]

print(f"\nEssential matrix inliers: {len(inlier_pts1)}")

# --------------------------------------------------
# Recover relative camera pose
# --------------------------------------------------

_, R, t, pose_mask = cv2.recoverPose(
    E,
    inlier_pts1,
    inlier_pts2,
    K
)

print("\nRotation matrix R:")
print(R)

print("\nTranslation direction t:")
print(t)

# --------------------------------------------------
# Convert rotation matrix to more intuitive angles
# --------------------------------------------------

rotation_vector, _ = cv2.Rodrigues(R)

print("\nRotation vector:")
print(rotation_vector)
