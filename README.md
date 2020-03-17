# ParticleFilter

The task is to implement a particle filter which uses the beacon observations and either
odometry and/or commanded velocity/rotation rate to estimate the robot’s location. We
are provided with one dataset as a CSV file, containing a “true” position from SLAM,
estimated positions from odometry, the commanded forward speed and rotation rate,
and the ID and position of any beacon observations. When two beacons are visible to
the camera, there are two rows with the same time and position but different beacon
observations. Additionally, we are provided with a CSV file containing the location of
each beacon.

The poses of the beacons are estimated in the camera-frame of the Turtlebot. This camera-
frame is offset from the base-frame of the Turtlebot. Full 6-D poses are measured but only
(x, y; θ) are given. Again x and y are in metres and θ is in radians. The camera-frame x
direction is in the Turtlebot’s forward direction of travel, the y direction is to its left, and
θ is measured anti-clockwise from the x direction of the base-frame.
