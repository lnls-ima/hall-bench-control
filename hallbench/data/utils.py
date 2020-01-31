# -*- coding: utf-8 -*-

"""Utils."""

import numpy as _np


def rotation_matrix(direction, angle):
    """Return the rotation matrix."""
    normalized_direction = normalized_vector(direction)
    ui = normalized_direction[0]
    uj = normalized_direction[1]
    uk = normalized_direction[2]
    c = _np.cos(angle)
    s = _np.sin(angle)
    R = _np.array([
        [c + (ui**2)*(1 - c), ui*uj*(1 - c) - uk*s, ui*uk*(1 - c) + uj*s],
        [uj*ui*(1 - c) + uk*s, c + (uj**2)*(1 - c), uj*uk*(1 - c) - ui*s],
        [uk*ui*(1 - c) - uj*s, uk*uj*(1 - c) + ui*s, c + (uk**2)*(1 - c)]
    ])
    return R


def parallel_vectors(vec1, vec2):
    """Return True if the vectors are parallel, False otherwise."""
    tol = 1e-15
    norm = _np.sum((_np.cross(vec1, vec2)**2))
    if norm < tol:
        return True
    else:
        return False


def vector_norm(vec):
    """Return the norm of the vector."""
    return _np.sqrt((_np.sum(_np.array(vec)**2)))


def normalized_vector(vec):
    """Return the normalized vector."""
    normalized_vec = vec / vector_norm(vec)
    return normalized_vec


def get_average_std(avgs, stds, nmeas):
    """Return the average and STD for a set of averages and STD values."""
    if len(avgs) == 0 and len(stds) == 0:
        return None, None

    if len(avgs) != len(stds):
        raise ValueError('Inconsistent size of input arguments')
        return None, None

    if nmeas == 0:
        raise ValueError('Invalid number of measurements.')
        return None, None

    elif nmeas == 1:
        avg = _np.mean(avgs)
        std = _np.std(avgs, ddof=1)

    else:
        n = len(avgs)*nmeas
        avgs = _np.array(avgs)
        stds = _np.array(stds)

        avg = _np.sum(avgs)*nmeas/n
        std = _np.sqrt((1/(n-1))*(
            _np.sum((nmeas-1)*(stds**2) + nmeas*(avgs**2)) -
            (1/n)*(_np.sum(avgs*nmeas)**2)))

    return avg, std
