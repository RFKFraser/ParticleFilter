"""Utility functions for particle filter assignment.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
import bisect
from numpy.random import uniform


def gauss(mu=0, sigma=1):
    """Evaluate a Gaussian PDF."""
    
    return np.exp(-0.5 * (mu / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


###############################################################################
# Particle filter functions

def resample(particles, weights):
    """Resample particles in proportion to their weights.

    Particles and weights should be arrays, and will be updated in place.
    """
    cum_weights = np.cumsum(weights)

    if cum_weights[-1] == 0.0:
        print('All weights are zero, giving up...')
        return False

    cum_weights /= cum_weights[-1]

    new_particles = []
    for _ in particles:
        # Copy a particle into the list of new particles, choosing based
        # on weight
        m = bisect.bisect_left(cum_weights, uniform(0, 1))
        p = particles[m]
        new_particles.append(p)

    # Replace old particles with new particles
    for m, p in enumerate(new_particles):
        particles[m] = p

    # Reset weights
    weights[:] = 1
    return True


def is_degenerate(weights):
    """Return true if the particles are degenerate and need resampling."""
    
    weights_sum = np.sum(weights)
    w = weights / weights_sum
    return 1.0 / np.sum(w**2) < 0.5 * len(w)


###############################################################################
# Functions for working with angles (in radians)

def wrapto2pi(angle):
    """Convert angle into range [0, 2 * pi)."""
    return angle % (2 * np.pi)


def wraptopi(angle):
    """Convert angle into range [-pi, pi)."""
    return wrapto2pi(angle + np.pi) - np.pi


def angle_difference(angle1, angle2):
    """Return principal difference angle, angle2 - angle1  with return value in range [-pi, pi)."""
    return ((((angle2 - angle1) % (2 * np.pi)) + (3 * np.pi)) % (2 * np.pi)) - np.pi
