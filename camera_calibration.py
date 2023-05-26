import cv2
import numpy as np
import glob
import os

square_size = 22.5  # mm
rows = 6
cols = 9

objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

print(objp)
obj_points = []
img_points = []

images = glob.glob("checkerboard_imgs/*.jpg")

for idx, img_path in enumerate(images):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))

    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        obj_points.append(objp)
        img_points.append(corners)

        cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
        
        save_path = os.path.join("checkerboard_imgs", f"corners_detected_{idx}.jpg")
        cv2.imwrite(save_path, img)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("Camera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(dist_coeffs)
