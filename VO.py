import numpy as np
import cv2
from camera import *
from imu import *
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag

sensor_address = "192.168.0.91:8081" # Change to the device IP address
camera_address = "192.168.0.91" # Change to the device IP address

camera_matrix = np.array([[883.18940021, 0.0, 636.00179082],
                          [0.0, 886.09959132, 498.82803173],
                          [0.0, 0.0, 1.0]])

dist_coeffs = np.array([ 2.53597019e-01, -1.86728946e+00, 1.07054010e-03, 7.92176184e-04, 4.55875910e+00])


projMatr1 = np.hstack((camera_matrix, np.zeros((3, 1))))
projMatr2 = np.hstack((camera_matrix, np.zeros((3, 1))))

# MATCH_THRESHOLD = 500
KEYFRAME_THRESHOLD = 50

def Q_continuous_white_noise(dim, dt=1., spectral_density=1.):
    if dim == 2:
        Q = np.array([[.25*dt**4, .5*dt**3],
                     [ .5*dt**3,    dt**2]], dtype=float)
    elif dim == 3:
        Q = np.array([[.25*dt**4, .5*dt**3, .5*dt**2],
                     [ .5*dt**3,    dt**2,       dt],
                     [ .5*dt**2,       dt,        1]], dtype=float)
    else:
        raise ValueError("dim must be 2 or 3")

    return Q * spectral_density

def block_diag(*arrs):
    bad_args = [k for k in range(len(arrs)) if not isinstance(arrs[k], np.ndarray)]
    if bad_args:
        raise ValueError("arguments in the following positions are not numpy arrays: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0))

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

dt = 1.0/60 
block = Q_continuous_white_noise(dim=3, dt=dt, spectral_density=1e-5)
Q = block_diag(block, block, block)


class VisualOdometry:

    def __init__(self, camera_matrix, dist_coeffs, sensor_address, kalman_enable, file_path):
        self.prev_frame = None
        self.prev_features = None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.detector = cv2.ORB_create()
        self.pose = np.eye(4)
        self.frame_count = 0
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.len_world_points = 0
        self.poses = []
        self.points_3d = []
        self.points_3d = np.array(self.points_3d)
        self.sensor_address = sensor_address
        self.kalman_enable = kalman_enable
        self.file_path = file_path

        self.prev_descriptors_list = [] 
        self.poses_list = [] 

        self.keyframe = None
        self.keypoints_keyframe = None
        self.descriptors_keyframe = None

        self.accel_sensor = Sensor(address=self.sensor_address, sensor_type="android.sensor.accelerometer")
        self.accel_sensor.connect()

        self.gyro_sensor = Sensor(address=self.sensor_address, sensor_type="android.sensor.gyroscope")
        self.gyro_sensor.connect()

        self.mag_sensor = Sensor(address=self.sensor_address, sensor_type="android.sensor.magnetic_field")
        self.mag_sensor.connect()

        self.kf = None
        if self.kalman_enable:
            self.kf = KalmanFilter(dim_x=12, dim_z=6) 
            self.kf.x[:6] = 0.0 # state Variables
            self.kf.F[0:3, 3:6] = np.eye(3) * dt # State Transition Matrix
            self.kf.F[3:6, 9:12] = np.eye(3) * dt  # state Transition Matrix
            self.kf.H[0:3, 0:3] = np.eye(3) # Measurenent Matrix
            self.kf.H[3:6, 3:6] = np.eye(3) # Measurenemt Matrix

            block = Q_continuous_white_noise(dim=3, dt=dt, spectral_density=1e-5)
            self.kf.Q = block_diag(block, block, block, block)

    def save_to_file(self, accel_data, gyro_data, pose, timestamp):
        with open(self.file_path, 'a') as f:
            f.write(f"{timestamp},{accel_data[0]},{accel_data[1]},{accel_data[2]},")
            f.write(f"{gyro_data[0]},{gyro_data[1]},{gyro_data[2]},")
            f.write(f"{pose[0]},{pose[1]},{pose[2]}\n")

    def process_frame(self, frame):
        timestamp = datetime.now().isoformat()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if self.keyframe is not None:
            matches = self.matcher.match(self.descriptors_keyframe, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32([self.keypoints_keyframe[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix, mask=mask)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.squeeze()

            self.pose = self.pose @ T

            if np.sum(mask) < KEYFRAME_THRESHOLD:
                self.keyframe = gray
                self.keypoints_keyframe = keypoints
                self.descriptors_keyframe = descriptors

        else:
            self.keyframe = gray
            self.keypoints_keyframe = keypoints
            self.descriptors_keyframe = descriptors

        accel_data, _ = self.accel_sensor.get_data()
        gyro_data, _ = self.gyro_sensor.get_data()
        mag_data, _ = self.mag_sensor.get_data()

        if accel_data is not None and gyro_data is not None and self.kalman_enable:
            self.kf.x[6:9] = np.reshape(accel_data, (3, 1)) 
            self.kf.x[9:12] = np.reshape(gyro_data, (3, 1))

            pose_measurement = np.hstack((self.pose[:3, 3], cv2.Rodrigues(self.pose[:3, :3])[0].squeeze()))
            self.kf.update(pose_measurement)

            self.pose[:3, 3] = self.kf.x[:3].flatten()
            self.pose[:3, :3] = cv2.Rodrigues(self.kf.x[3:6])[0] * dt + 0.5 * np.outer(self.kf.x[9:12], self.kf.x[9:12]) * dt**2
            point_3d = self.convert_transformation_matrix_to_open3d(self.pose)
            # print(point_3d)
            self.save_to_file(accel_data, gyro_data, point_3d[:3, 3], timestamp)
        else:
            point_3d = self.convert_transformation_matrix_to_open3d(self.pose)
            # print(point_3d)
            if accel_data is not None and gyro_data is not None:
                self.save_to_file(accel_data, gyro_data, point_3d[:3, 3], timestamp)
    
        self.poses.append(self.pose)
        self.frame_count += 1

        return self.pose, self.points_3d, point_3d

    def get_camera_pose(self):
        return self.pose
    
    def convert_transformation_matrix_to_open3d(self, matrix):
        result = np.eye(4)
        result[:3, :3] = matrix[:3, :3]
        result[:3, 3] = matrix[:3, 3]
        return result


with open("IMU_data.txt", 'w') as file:
    file.write("")

camera_trajectory = []
all_poses = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

def update_plot():
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    camera_trajectory_matplotlib_np = np.array(camera_trajectory)
    ax.plot(camera_trajectory_matplotlib_np[:, 0], camera_trajectory_matplotlib_np[:, 1], camera_trajectory_matplotlib_np[:, 2], 'b.-')
    plt.draw()
    plt.pause(0.001)

if __name__ == "__main__":
    try:
        with open("IMU_data.txt", 'w') as file:
            file.write("")
        camera = Camera(camera_address, 4747)
        camera.connect()

        visual_odometry = VisualOdometry(camera_matrix, dist_coeffs, sensor_address, True, "IMU_data.txt") # Can enable and disable Kalman Filter

        while True:
            frame = camera.get_frame_for_odometry()
            if frame is not None:
                pose, points_3d, pose_o3d  = visual_odometry.process_frame(frame)
                if pose is not None:

                    all_poses.append(pose_o3d[:3, 3])
                    camera_trajectory.append(pose_o3d[:3, 3])

                    update_plot()

            if camera.stopped:
                break

    except KeyboardInterrupt:
        print("Stopping visual odometry...")
    finally:
        print("Stopping visual odometry...")
        if 'visual_odometry' in globals():
            visual_odometry.accel_sensor.disconnect()
            visual_odometry.gyro_sensor.disconnect()
            visual_odometry.mag_sensor.disconnect()
        plt.close(fig)