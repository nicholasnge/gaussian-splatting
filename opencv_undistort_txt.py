import os
import cv2
import numpy as np

# Paths
source_path = r"C:\Users\Nicholas\Desktop\3DGSDATASETS\tandt_intermediate\Francis"  # e.g., "C:/Users/Nicholas/Desktop/3DGSDATASETS/tandt_intermediate/Family"
sparse_path = os.path.join(source_path, r"sparse/txt")
image_path = os.path.join(source_path, r"images")
output_path = os.path.join(source_path, r"opencv_undistorted")

os.makedirs(output_path, exist_ok=True)

# === Parse cameras.txt ===
camera_file = os.path.join(sparse_path, "cameras.txt")
with open(camera_file, 'r') as f:
    for line in f:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split()
        camera_model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = list(map(float, parts[4:]))
        if camera_model == "SIMPLE_RADIAL":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
            k = params[3]
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])
            D = np.array([k, 0, 0, 0])  # OpenCV expects 4 or 5 coefficients
        else:
            raise NotImplementedError(f"Camera model {camera_model} not supported")

# === Parse images.txt to get image filenames ===
image_names = []
image_file = os.path.join(sparse_path, "images.txt")
with open(image_file, 'r') as f:
    for line in f:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split()
        if len(parts) == 10:  # first line of image block
            image_name = parts[-1]
            image_names.append(image_name)

# === Undistort each image using OpenCV ===
for name in image_names:
    img_in = cv2.imread(os.path.join(image_path, name))
    if img_in is None:
        print(f"Could not load {name}")
        continue
    undistorted = cv2.undistort(img_in, K, D)
    cv2.imwrite(os.path.join(output_path, name), undistorted)

print("✅ Done: undistorted images saved to", output_path)
