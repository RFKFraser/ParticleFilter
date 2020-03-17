"""Particle filter sensor and motion model implementations.

Ronan Fraser
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
from utils import angle_difference
from utils import gauss
from utils import wraptopi
from scipy.stats import norm
import transform
import random


def motion_model(poses, command_prev, odom_pose, odom_pose_prev):
    """Apply motion model and return updated array of poses.

    Parameters
    ----------

    poses: an M x 3 array of robot poses where M is the number of
    particles.  Each pose is (x, y, theta) where x and y are in metres
    and theta is in radians.

    command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    Returns
    -------
    An M x 3 array of updated poses.

    """

    M = poses.shape[0]
    
    # For each particle calculate its predicted pose plus some
    # additive error to represent the process noise. 

    # Odometry model pose change parameterisation
    phi_1 = np.arctan2((odom_pose[1] - odom_pose_prev[1]), (odom_pose[0] - odom_pose_prev[0])) - odom_pose_prev[2],
    phi_2 = wraptopi(odom_pose[2] - odom_pose_prev[2] - phi_1)
    distance = np.sqrt((odom_pose[1] - odom_pose_prev[1])**2 + (odom_pose[0] - odom_pose_prev[0])**2)
    
    # Adding randomly sampled noise as s
    mu, sigma = 0, 0.02 # mean and standard deviation
    s = np.random.normal(mu, sigma, len(poses))    
    
    
    # Updating poses
    poses[:, 0] = poses[:, 0] + distance * np.cos(poses[:, 2] + phi_1) + s#odom_pose_prev[2]
    poses[:, 1] = poses[:, 1] + distance * np.sin(poses[:, 2] + phi_1) + s#odom_pose_prev[2]
    poses[:, 2] = poses[:, 2] + phi_1 + phi_2  + s

    return poses




def sensor_model(poses, beacon_pose, beacon_loc, tf, odom_pose):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    poses: an M x 3 array of robot poses where M is the number of
    particles.  Each pose is (x, y, theta) where x and y are in metres
    and theta is in radians.

    beacon_pose: the measured pose of the beacon (x, y, theta)
    relative to the robot's camera pose.

    beacon_loc: the known global pose of the beacon (x, y, theta).

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """
    
    # For each particle calculate its weight based on its pose,
    # the relative beacon pose, and the global beacon location.
    
    
    beacon_measured = np.zeros(3)
    
    # Distance to the beacon from the robot
    distance_b = np.sqrt(beacon_pose[0]**2 + beacon_pose[1]**2) 
    
    # This is bearing to beacon from local frame x axis
    bearing = odom_pose[2] + np.arctan2(beacon_pose[1], beacon_pose[0] )
    
    # Using correct trig transform for the quadrant the bearing is in to find x and y as local 
    if (np.pi/2) < bearing < -(np.pi/2):
        x_add = distance_b * np.sin( bearing % (np.pi/2))
        y_add = distance_b * np.cos( bearing % (np.pi/2))
    else:
        x_add = distance_b * np.cos( bearing )
        y_add = distance_b * np.sin( bearing )
        
    #print(x_add, ' = x global ', y_add, ' = y global ')
    #print(beacon_pose[0], ' = x ', beacon_pose[1], ' = y ')
    #print('range global = ', np.sqrt(x_add**2 + y_add**2), 'range measured = ', np.sqrt(beacon_pose[0]**2 + beacon_pose[1]**2))
        
    # This is the local beacon location from measurement 
    beacon_measured[0] = odom_pose[0] + x_add
    beacon_measured[1] = odom_pose[1] + y_add
    beacon_measured[2] = beacon_pose[2]
    
    # This transforms the local poses into global poses
    g_beacon_mes = transform.transform_pose(tf, beacon_measured)
    g_robot_pose = transform.transform_pose(tf, odom_pose)
    
    
    # This is the robot range and angle from pg. 122 of the notes.
    robot_range = np.sqrt((g_beacon_mes[0] - g_robot_pose[0])**2+(g_beacon_mes[1] - g_robot_pose[1])**2)
    robot_angle = angle_difference(g_robot_pose[2], np.arctan((g_beacon_mes[1] - g_robot_pose[1])/(g_beacon_mes[0] - g_robot_pose[0])))
    # This is the particle range and angle from pg. 122 of the notes.
    particle_range = np.sqrt((beacon_loc[0] - poses[:,0])**2+(beacon_loc[1] - poses[:,1])**2)
    particle_angle = angle_difference(poses[:,2], np.arctan((beacon_loc[1] - poses[:,1])/(beacon_loc[0] - poses[:,0])))    
    
    M = poses.shape[0]
    weights = np.ones(M)
   
    #print('robot_range - particle_range', robot_range-particle_range)
    #print('robot_angle - particle_angle', angle_difference(particle_angle, robot_angle))
    #print('gauss of range diff = ', gauss(robot_range - particle_range, 0.01))
    #print('gauss of angle diff = ', *gauss( angle_difference(particle_angle, robot_angle) , 0.01 ))

    # Weights calculated by multiplying PDFs. dont need to be normalised.
    weights = gauss(robot_range - particle_range, sigma = 0.1) * gauss(angle_difference(robot_angle, particle_angle), sigma = 0.1)
    
    return weights
