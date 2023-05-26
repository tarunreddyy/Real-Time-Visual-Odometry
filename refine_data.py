import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('IMU_data.txt', delimiter=',')

timestamps = data[:, 0]
accel_x = data[:, 1]
accel_y = data[:, 2]
accel_z = data[:, 3]

smoothed_accel_z = np.convolve(accel_z, np.ones(10) / 10, mode='same')

threshold = 0.2

pose_x = [0]
pose_y = [0]
pose_z = [0]

for i in range(1, len(timestamps)):
    if abs(smoothed_accel_z[i] - smoothed_accel_z[i-1]) > threshold:
        pose_x.append(data[i, 7])
        pose_y.append(data[i, 8])
        pose_z.append(data[i, 9])

# for i in range(1, len(timestamps)):
#     if abs(smoothed_accel_z[i] - smoothed_accel_z[i-1]) > threshold:
#         pose_x.append(pose_x[-1] + accel_x[i])
#         pose_y.append(pose_y[-1] + accel_y[i])
#         pose_z.append(pose_z[-1] + accel_z[i])Z
#     else:
#         pose_x.append(pose_x[-1])
#         pose_y.append(pose_y[-1])
#         pose_z.append(pose_z[-1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pose_x, pose_y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.set_zlabel('Z')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(data[:, 7], data[:, 8])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# # ax.set_zlabel('Z')
# plt.show()