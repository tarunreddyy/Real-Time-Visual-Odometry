# Visual Odometry using IMU and Camera
This project implements visual odometry using an inertial measurement unit (IMU) and a camera. It estimates the camera's 3D pose (position and orientation) based on the accelerometer data from the IMU and the visual information captured by the camera.

## Requirements
To run this project, you need the following dependencies:

- Python 3.x
- NumPy
- OpenCV (cv2)
- Matplotlib
- SciPy
- FilterPy

## Installation

- Install the required dependencies using pip:
`pip install numpy opencv-python matplotlib filterpy`

## Usage
1. Connect the camera and IMU devices to your system (Android phone used: Droidcam for video feed and Sensor Server for IMU data).
2. Adjust the necessary parameters in the visual_odometry.py file, such as sensor_address, camera_address, camera_matrix, dist_coeffs, etc., according to your setup.
3. Run the visual_odometry.py script:

`python VO.py`

4. The script will start processing frames from the camera and estimating the camera's pose based on the IMU and visual data. The estimated pose will be displayed in a 3D plot.

## Notes
- Make sure the camera and IMU are properly calibrated and synchronized.
- The project assumes that the IMU and camera data are synchronized and aligned.
- Adjust the threshold values (KEYFRAME_THRESHOLD, threshold) according to your specific requirements and data.
- The project saves the accelerometer and gyroscope data, along with the estimated camera poses, to a file named IMU_data.txt. You can modify this behavior in the save_to_file method of the VisualOdometry class. This data can be used later for post processing with `refine_data.py`.