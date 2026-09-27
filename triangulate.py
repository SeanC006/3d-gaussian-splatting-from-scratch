import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load images
# --------------------------------------------------

img1 = cv2.imread("images/level/frame_0000.jpg")
img2 = cv2.imread("images/level/frame_0010.jpg")

if img1 is None or img2 is None:
    raise RuntimeError("Could not load images.")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# SIFT
# --------------------------------------------------

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# --------------------------------------------------
# Feature matching
# --------------------------------------------------

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# --------------------------------------------------
# Enforce true one-to-one matching
# --------------------------------------------------

good_matches = sorted(
    good_matches,
    key=lambda m: m.distance
)

unique_matches = []

used_query_indices = set()
used_train_indices = set()

used_pts1 = set()
used_pts2 = set()

for m in good_matches:

    pt1 = kp1[m.queryIdx].pt
    pt2 = kp2[m.trainIdx].pt

    # Round coordinates so floating-point representation
    # doesn't cause tiny differences to count as unique
    pt1_key = (round(pt1[0], 3), round(pt1[1], 3))
    pt2_key = (round(pt2[0], 3), round(pt2[1], 3))

    if (
        m.queryIdx not in used_query_indices
        and m.trainIdx not in used_train_indices
        and pt1_key not in used_pts1
        and pt2_key not in used_pts2
    ):
        unique_matches.append(m)

        used_query_indices.add(m.queryIdx)
        used_train_indices.add(m.trainIdx)

        used_pts1.add(pt1_key)
        used_pts2.add(pt2_key)

good_matches = unique_matches

print(f"Good unique matches: {len(good_matches)}")

# --------------------------------------------------
# Extract pixel coordinates
# --------------------------------------------------

pts1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_matches
])

pts2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_matches
])

print("\nMatched keypoints:")
for i, m in enumerate(good_matches):
    print(
        f"Match {i}: "
        f"queryIdx={m.queryIdx}, "
        f"trainIdx={m.trainIdx}, "
        f"pt1={kp1[m.queryIdx].pt}, "
        f"pt2={kp2[m.trainIdx].pt}"
    )

# --------------------------------------------------
# Camera intrinsics
# --------------------------------------------------

K = np.array([
    [640, 0, 320],
    [0, 640, 240],
    [0, 0, 1]
], dtype=np.float64)

# --------------------------------------------------
# Essential matrix
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

# Only keep essential-matrix inliers
inlier_pts1 = pts1[mask.ravel() == 1]
inlier_pts2 = pts2[mask.ravel() == 1]

print(f"Essential matrix inliers: {len(inlier_pts1)}")

# --------------------------------------------------
# Recover camera pose
# --------------------------------------------------

_, R, t, pose_mask = cv2.recoverPose(
    E,
    inlier_pts1,
    inlier_pts2,
    K
)

print("\nR:")
print(R)

print("\nt:")
print(t)

# --------------------------------------------------
# Keep only recoverPose inliers
# --------------------------------------------------

pose_inlier_mask = pose_mask.ravel() > 0

pose_inlier_pts1 = inlier_pts1[pose_inlier_mask]
pose_inlier_pts2 = inlier_pts2[pose_inlier_mask]

print(f"Pose inliers: {len(pose_inlier_pts1)}")

# --------------------------------------------------
# Projection matrices
# --------------------------------------------------

P1 = K @ np.hstack((
    np.eye(3),
    np.zeros((3, 1))
))

P2 = K @ np.hstack((
    R,
    t
))

# --------------------------------------------------
# Triangulate
# --------------------------------------------------

points_4d = cv2.triangulatePoints(
    P1,
    P2,
    pose_inlier_pts1.T,
    pose_inlier_pts2.T
)

# Convert homogeneous coordinates to 3D
points_3d = points_4d[:3] / points_4d[3]

points_3d = points_3d.T

# --------------------------------------------------
# Check depth in both cameras
# --------------------------------------------------

# Depth in Camera 1
depth1 = points_3d[:, 2]

# Transform points into Camera 2 coordinates
points_cam2 = (R @ points_3d.T + t).T

# Depth in Camera 2
depth2 = points_cam2[:, 2]

print("\nDepth check:")

for i in range(len(points_3d)):
    print(
        f"Point {i}: "
        f"Z1={depth1[i]:.3f}, "
        f"Z2={depth2[i]:.3f}"
    )

# --------------------------------------------------
# Project 3D points back onto both images
# --------------------------------------------------

# Convert 3D points to homogeneous coordinates
points_3d_h = np.hstack((
    points_3d,
    np.ones((points_3d.shape[0], 1))
))

# Project into image 1
proj1_h = (P1 @ points_3d_h.T).T
proj1 = proj1_h[:, :2] / proj1_h[:, 2:3]

# Project into image 2
proj2_h = (P2 @ points_3d_h.T).T
proj2 = proj2_h[:, :2] / proj2_h[:, 2:3]

error1 = np.linalg.norm(pose_inlier_pts1 - proj1, axis=1)
error2 = np.linalg.norm(pose_inlier_pts2 - proj2, axis=1)

print("\nDetailed point information:")

for i in range(len(points_3d)):
    print(f"\nPoint {i}")
    print(f"  3D: {points_3d[i]}")
    print(f"  Original Frame 1: {pose_inlier_pts1[i]}")
    print(f"  Projected Frame 1: {proj1[i]}")
    print(f"  Error Frame 1: {error1[i]:.3f}")
    print(f"  Original Frame 2: {pose_inlier_pts2[i]}")
    print(f"  Projected Frame 2: {proj2[i]}")
    print(f"  Error Frame 2: {error2[i]:.3f}")

print("\nReprojection error:")
print(f"Frame 1 mean: {np.mean(error1):.3f} pixels")
print(f"Frame 2 mean: {np.mean(error2):.3f} pixels")

# --------------------------------------------------
# Plot projections on original images
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Image 1
axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

axes[0].scatter(
    pts1[:, 0],
    pts1[:, 1],
    s=20,
    label="Original SIFT points"
)

axes[0].scatter(
    proj1[:, 0],
    proj1[:, 1],
    s=40,
    marker="x",
    label="3D projections"
)

axes[0].set_title("Frame 0000")
axes[0].legend()


# Image 2
axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

axes[1].scatter(
    pts2[:, 0],
    pts2[:, 1],
    s=20,
    label="Original SIFT points"
)

axes[1].scatter(
    proj2[:, 0],
    proj2[:, 1],
    s=40,
    marker="x",
    label="3D projections"
)

axes[1].set_title("Frame 0010")
axes[1].legend()

plt.show()


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n3D points:")
for i, point in enumerate(points_3d):
    print(
        f"Point {i}: "
        f"X={point[0]:.3f}, "
        f"Y={point[1]:.3f}, "
        f"Z={point[2]:.3f}"
    )

X = points_3d[:, 0]
Y = points_3d[:, 1]
Z = points_3d[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X, Y, Z)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Triangulated 3D Feature Points")

plt.show()
