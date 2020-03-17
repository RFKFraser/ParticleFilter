"""
A particle filter which uses the beacon observations and 
odometry and/or commanded velocity/rotation rate to estimate the robot’s location

Ronan Fraser
Department of Electrical and Computer Engineering
University of Canterbury
"""

from __future__ import print_function, division
from numpy.random import uniform
import matplotlib; matplotlib.use("TkAgg")
from models import motion_model, sensor_model
from utils import *
from plot import *
from transform import *


# Load data

# data is a (many x 13) matrix. Its columns are:
# time_ns, velocity_command, rotation_command, map_x, map_y, map_theta, odom_x, odom_y, odom_theta,
# beacon_ids, beacon_x, beacon_y, beacon_theta
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

# Time in ns
t = data[:, 0]

# Velocity command in m/s, rotation command in rad/s
commands = data[:, 1:3]

# Position in map frame, from SLAM (this approximates ground truth)
slam_poses = data[:, 3:6]

# Position in odometry frame, from wheel encoders and gyro
odom_poses = data[:, 6:9]

# Id and measured position of beacon in camera frame
beacon_ids = data[:, 9]
beacon_poses = data[:, 10:13]
# Use beacon id of -1 if no beacon detected
beacon_ids[np.isnan(beacon_ids)] = -1
beacon_ids = beacon_ids.astype(int)
beacon_visible = beacon_ids >= 0

# map_data is a 16x13 matrix.  Its columns are:
# beacon_ids, x, y, theta, (9 columns of covariance)
map_data = np.genfromtxt('beacon_map.csv', delimiter=',', skip_header=1)

Nbeacons = map_data.shape[0]
beacon_locs = np.zeros((Nbeacons, 3))
for m in range(Nbeacons):
    id = int(map_data[m, 0])
    beacon_locs[id] = map_data[m, 1:4]

# Remove jumps in the pose history
slam_poses = clean_poses(slam_poses)

# Transform odometry poses into map frame using transform found from initial global
odom_to_map = find_transform(odom_poses[0], slam_poses[0])
print(odom_to_map)
odom_poses_g = transform_pose(odom_to_map, odom_poses)

plt.ion()
fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(111)

plot_beacons(axes, beacon_locs, label='Beacons')
plot_path(axes, slam_poses, '-', label='SLAM')
# Uncomment to show odometry when debugging
#plot_path(axes, odom_poses_g, 'b:', label='Odom')

axes.legend(loc='lower right')

axes.set_xlim([-6, None])
axes.axis('equal')

# Tweak axes to make plotting better
axes.invert_yaxis()
axes.set_xlabel('y (m)')
axes.set_ylabel('x (m)')
axes.figure.canvas.draw()
axes.figure.canvas.flush_events()

# TODO: Set this to avoid twirl at start
# When your algorithm works well set to 0
start_step = 50

# Number of particles, you may need more or fewer!
Nparticles = 100

# How many steps between display updates
display_steps = 100

# Set initial belief
start_pose = slam_poses[start_step]
Xmin = start_pose[0] - 0.1
Xmax = start_pose[0] + 0.1
Ymin = start_pose[1] - 0.1
Ymax = start_pose[1] + 0.1
Tmin = start_pose[2] - 0.1
Tmax = start_pose[2] + 0.1

weights = np.ones(Nparticles)
poses = np.zeros((Nparticles, 3))
print(*poses)
print(weights)
for m in range(Nparticles):
    poses[m] = (uniform(Xmin, Xmax),
                uniform(Ymin, Ymax),
                uniform(Tmin, Tmax))
#print(poses[1:5])

Nposes = odom_poses_g.shape[0]
est_poses = np.zeros((Nposes, 3))

display_step_prev = 0
for n in range(start_step + 1, Nposes):

    #  motion model function
    poses = motion_model(poses, commands[n-1], odom_poses_g[n], odom_poses_g[n - 1])

    if beacon_visible[n]:

        beacon_id = beacon_ids[n]
        beacon_loc = beacon_locs[beacon_id]
        beacon_pose = beacon_poses[n]
        curr_odom_poses = odom_poses[n,:]
        
        # sensor model function
        weights = sensor_model(poses, beacon_pose, beacon_loc, odom_to_map, curr_odom_poses)

        if sum(weights) < 1e-50:
            print('All weights are close to zero, you are lost...')
            # Do something to recover
            resample(poses, weights)
            break

        if is_degenerate(weights):
            print('Resampling %d' % n)
            resample(poses, weights)
            print('weights', weights)

    est_poses[n] = poses.mean(axis=0)

    if n > display_step_prev + display_steps:
        print(n)

        # Show particle cloud
        plot_particles(axes, poses, weights)

        # Leave breadcrumbs showing current odometry
        #plot_path(axes, odom_poses_g[n], 'k.')

        # Show mean estimate
        plot_path_with_visibility(axes, est_poses[display_step_prev-1 : n+1],
                                  '-', visibility=beacon_visible[display_step_prev-1 : n+1])
        display_step_prev = n

# Display final plot
print('Done, displaying final plot')
plt.ioff()
plt.show()

# Save final plot to file
plot_filename = 'path.pdf'
print('Saving final plot to', plot_filename)

plot_path(axes, est_poses, 'r-', label='Estimated trajectory')
axes.legend(loc='lower right')

fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(111)

plot_beacons(axes, beacon_locs, label='Beacons')
plot_path(axes, slam_poses, 'b-', label='SLAM')
plot_path(axes, odom_poses_g, 'b:', label='Odom')
plot_path(axes, est_poses, 'r-', label='Estimated trajectory')
axes.legend(loc='lower right')

axes.set_xlim([-6, None])
axes.axis('equal')

# Tweak axes to make plotting better
axes.invert_yaxis()
axes.set_xlabel('y (m)')
axes.set_ylabel('x (m)')
fig.savefig(plot_filename, bbox_inches='tight')
