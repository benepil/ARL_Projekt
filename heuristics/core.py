import numpy as np


MIN_DISTANCE: float = 0.15
BREAK_DISTANCE: float = 0.25
ACCELERATION_DISTANCE: float = 0.7
BETA: float = 1/4


def follow_the_center_line(observation: np.ndarray):
    frames: np.ndarray = observation.reshape(4, 12)
    last_frames: np.ndarray = frames[-1]
    distance_sensors: np.ndarray = last_frames[3:]
    # left column  = left sensor group
    # right column = right sensor group

    distance_sensors_groups: np.ndarray = (
        np.array([[distance_sensors[4], distance_sensors[1]],
                  [distance_sensors[8], distance_sensors[7]],
                  [distance_sensors[5], distance_sensors[2]],
                  [distance_sensors[6], distance_sensors[3]]]))

    total_distance: np.ndarray = distance_sensors_groups.sum(axis=1)
    assert all(total_distance > 0), "Divided by Zero!"
    # distance_ration of 0.5 -> agent is in the center of the road
    # distance_ration of 1.0 -> agent is to left side  of the road
    # distance_ration of 0.0 -> agent is to right side of the road
    distance_ratios: np.ndarray = distance_sensors_groups[:, 0] / total_distance
    center = 0.5
    # -1 == left | +1 == right
    direction = -2 * (distance_ratios > center) + 1
    ratios_centered = distance_ratios - center
    rotations = direction * abs(ratios_centered) ** BETA

    domain_maximum = (1 - MIN_DISTANCE - center) ** BETA
    rescale_factor_rotation = 1 / domain_maximum
    rotations *= rescale_factor_rotation
    rotation = rotations.mean()

    front_sensor: float = distance_sensors[0]
    if front_sensor < BREAK_DISTANCE:
        acceleration = 0.0
    elif front_sensor > ACCELERATION_DISTANCE:
        acceleration = 1.0
    else:
        acceleration = (front_sensor - BREAK_DISTANCE)
        rescale_factor_breaking = 1 / (ACCELERATION_DISTANCE - BREAK_DISTANCE)
        acceleration *= rescale_factor_breaking

    return np.array([rotation, acceleration])








