# 3D Gaussian Splatting from Scratch

A learning project exploring the computer vision pipeline behind 3D Gaussian Splatting, starting from feature matching and two-view geometry and progressively building toward multi-view reconstruction and Gaussian splatting.

## Current Progress

### Milestone 1 — Two-View Reconstruction

Currently implemented:

* SIFT feature detection and description
* BFMatcher feature matching
* Lowe's ratio test
* One-to-one correspondence filtering
* Duplicate pixel-coordinate filtering
* Essential matrix estimation
* RANSAC geometric outlier rejection
* Camera pose recovery
* Chirality / positive-depth filtering
* Linear triangulation
* 3D point reconstruction
* Reprojection error calculation
* 3D visualization of triangulated feature points

Current pipeline:

```text
Frame 1 + Frame 2
        ↓
       SIFT
        ↓
   Feature Matching
        ↓
   Lowe Ratio Test
        ↓
 One-to-One Filtering
        ↓
Duplicate Coordinate Filtering
        ↓
 Essential Matrix + RANSAC
        ↓
    Recover Pose
        ↓
 Chirality Check
        ↓
    Triangulation
        ↓
     3D Points
        ↓
 Reprojection Validation
```

## Example Results

The current two-view reconstruction uses a pair of webcam images and produces a sparse set of triangulated 3D feature points.

Because monocular two-view reconstruction does not provide an absolute scale, the reconstructed translation and 3D coordinates are expressed in an arbitrary scale.

## Next Steps

* Improve feature matching and correspondence robustness
* Extend from two frames to multiple frames
* Estimate a consistent camera trajectory
* Build a larger sparse 3D point cloud
* Explore bundle adjustment
* Investigate dense reconstruction
* Use the reconstructed scene as initialization for 3D Gaussian Splatting

## Technologies

* Python
* OpenCV
* NumPy
* Matplotlib
* SIFT
* Epipolar Geometry
* Essential Matrix
* RANSAC
* Triangulation
* Structure from Motion
* 3D Gaussian Splatting

## Goal

The goal is to understand the geometry and optimization underlying 3D reconstruction rather than treating 3D Gaussian Splatting as a black-box implementation.
