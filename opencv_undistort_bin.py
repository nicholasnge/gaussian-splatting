import os
import cv2
import numpy as np
import struct

# === Paths ===
source_path = r"C:\Users\Nicholas\Desktop\3DGSDATASETS\tandt_intermediate\Train"
sparse_path = os.path.join(source_path, "sparse", "0")  # path to .bin files
image_path = os.path.join(source_path, "images")
output_path = os.path.join(source_path, "undistorted_images")
os.makedirs(output_path, exist_ok=True)

# === Camera model ID to name mapping ===
CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}

def read_next_bytes(fid, num_bytes, format_sequence, endian="<"):
    return struct.unpack(endian + format_sequence, fid.read(num_bytes))

# === Parse cameras.bin ===
def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_id = read_next_bytes(f, 4, "I")[0]
            model_id = read_next_bytes(f, 4, "i")[0]
            width = read_next_bytes(f, 8, "Q")[0]
            height = read_next_bytes(f, 8, "Q")[0]
            num_params = {
                0: 3,  # SIMPLE_PINHOLE
                1: 4,  # PINHOLE
                2: 4,  # SIMPLE_RADIAL ← FIXED
                3: 5,  # RADIAL
                4: 8,  # OPENCV
            }.get(model_id, 0)
            params = read_next_bytes(f, 8 * num_params, "d" * num_params)
            cameras[cam_id] = {
                "model_id": model_id,
                "model_name": CAMERA_MODELS.get(model_id, "UNKNOWN"),
                "width": width,
                "height": height,
                "params": params
            }
    return cameras

# === Parse images.bin ===
def read_images_binary(path):
    images = []
    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(f, 4, "I")[0]
            qvec = read_next_bytes(f, 8 * 4, "d" * 4)
            tvec = read_next_bytes(f, 8 * 3, "d" * 3)
            camera_id = read_next_bytes(f, 4, "I")[0]

            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")

            num_points2D = read_next_bytes(f, 8, "Q")[0]
            f.read(24 * num_points2D)  # Skip 2D points

            images.append((name, camera_id))
    return images

# === Load data ===
cameras = read_cameras_binary(os.path.join(sparse_path, "cameras.bin"))
images = read_images_binary(os.path.join(sparse_path, "images.bin"))

# === Use first camera only (assume shared intrinsics) ===
_, cam_id = images[0]
camera = cameras[cam_id]
print(f"Camera ID: {cam_id}")
print(f"Model ID: {camera['model_id']} ({camera['model_name']})")
print(f"Params: {camera['params']}")

if camera["model_name"] != "SIMPLE_RADIAL":
    raise NotImplementedError(f"Camera model {camera['model_name']} not supported")

# === Intrinsics for SIMPLE_RADIAL ===
f, cx, cy, k1 = camera["params"]
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]], dtype=np.float32)
D = np.array([k1, 0, 0, 0], dtype=np.float32)



# === Desired undistorted output size ===
undistorted_size = (1960, 1085)

# Step 1: Get optimized camera matrix for full FoV (alpha=1) at custom size
new_K, valid_roi = cv2.getOptimalNewCameraMatrix(
    K, D,
    (camera["width"], camera["height"]),  # input size
    alpha=0,
    newImgSize=undistorted_size
)

# Step 2: Create remapping functions
map1, map2 = cv2.initUndistortRectifyMap(
    K, D,
    None,
    new_K,
    undistorted_size,
    cv2.CV_32FC1
)

# Step 3: Undistort and save
for name, _ in images:
    input_img_path = os.path.join(image_path, name)
    img_in = cv2.imread(input_img_path)
    if img_in is None:
        print(f"⚠️ Could not load {name}")
        continue

    undistorted = cv2.remap(img_in, map1, map2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_path, name), undistorted)

# Step 4: Save updated intrinsics
# np.savetxt(os.path.join(output_path, "camera_matrix.txt"), new_K)
# with open(os.path.join(output_path, "resolution.txt"), "w") as f:
#     f.write(f"{undistorted_size[0]} {undistorted_size[1]}\n")

print("✅ Done: undistorted images saved to", output_path)

print("\n📷 Final Camera Intrinsic Matrix (new_K):")
for row in new_K:
    print(" ".join(f"{val:.6f}" for val in row))

print("\n🖼️ Undistorted Image Resolution:")
print(f"Width: {undistorted_size[0]}, Height: {undistorted_size[1]}")



# === Undistort each image ===
# for name, _ in images:
#     input_img_path = os.path.join(image_path, name)
#     img_in = cv2.imread(input_img_path)
#     if img_in is None:
#         print(f"⚠️ Could not load {name}")
#         continue

#     undistorted = cv2.undistort(img_in, K, D)
#     cv2.imwrite(os.path.join(output_path, name), undistorted)


print("✅ Done: undistorted images saved to", output_path)
