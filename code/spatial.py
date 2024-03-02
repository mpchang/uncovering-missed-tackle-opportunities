from scipy.spatial import voronoi_plot_2d, Voronoi
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_cell_area(cell):
    """
    Given a voronoi cell vertices, return the area of the cell using the shoelace algorithm

    :param cell: list of tuples representing the vertices of the cell
    :return: float representing the area of the cell
    """
    n = len(cell)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += cell[i][0] * cell[j][1]
        area -= cell[j][0] * cell[i][1]
    area = abs(area) / 2.0
    return area


def reflect_points(points, bounds):
    """
    Given a set of points and a bounding box, return the set of points and their reflections across the boundaries

    :param points: list of tuples representing the points
    :param bounds: tuple representing the corners of the bounding box

    :return points: list of tuples representing hte original points
    :return refl_points: list of tuples representing the reflected points
    """
    x_axes = np.unique(bounds[:, 0])
    y_axes = np.unique(bounds[:, 1])

    x_min = np.min(x_axes)
    x_max = np.max(x_axes)
    y_min = np.min(y_axes)
    y_max = np.max(y_axes)

    refl_points = []
    for pt in points:
        refl_points.append((pt[0], y_min - (pt[1] - y_min)))  # refl about y_min
        refl_points.append((pt[0], y_max + (y_max - pt[1])))  # refl about y_max
        refl_points.append((x_min - (pt[0] - x_min), pt[1]))  # refl about x_min
        refl_points.append((x_max + (x_max - pt[0]), pt[1]))  # refl about x_max

    return np.array(points), np.array(refl_points)


def in_bounds(points, bounds):
    """
    Given a set of points and a bounding box, return the set of points that are in-bounds

    Args:
        points (list): list of tuples representing the points
        bounds (list): list of tuples representing the corners of the bounding box

    Returns:
        in_bounds_pts (list): list of tuples representing the in-bounds points
    """
    in_bounds_pts = []
    x_min = np.min(bounds[:, 0])
    x_max = np.max(bounds[:, 0])
    y_min = np.min(bounds[:, 1])
    y_max = np.max(bounds[:, 1])

    for pt in points:
        if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max:
            in_bounds_pts.append(pt)

    return in_bounds_pts


def get_index_from_points(total_points, find_points):
    """
    Given a set of total points, return the indices in that list that correspond to find_points.

    Args:
        total_points (_type_): _description_
        find_points (_type_): _description_
    """
    pt_dict = {i: pt.tolist() for i, pt in enumerate(total_points)}
    find_points = [[p[0], p[1]] for p in find_points]
    idx = [k for k, v in pt_dict.items() if v in find_points]

    return idx


def get_regions_from_points(vor, points):
    """
    Given a set of points and a Voronoi object, return the regions that correspond to those points.

    Args:
        vor (scipy.spatial.Voronoi): A voronoi object created from scipy
        points (list): a list of tuples corresponding to the points to get the regions for

    Returns:
        regions (dict): a dict where the keys are points and the values are the list of voronoi vertices that correspond to that point
    """
    cell_dict = {}
    pt_regions = vor.point_region

    for pt in points:
        ix = get_index_from_points(vor.points, [pt])
        region_ix = pt_regions[ix[0]]
        cell_dict[(pt[0], pt[1])] = [vor.vertices[i] for i in vor.regions[region_ix]]

    return cell_dict


def _get_areas_from_points(points, bounds):
    """
    Given a set of points and bounds, return the areas of the regions that correspond to those points.
    A bounding box, given by bounds, will be applied such that only the areas within the bounds are returned.

    Args:
        points (list of tuples): a list of tuples corresponding to the points to get the regions for
        bounds (list of tuples): a list of tuples corresponding to the corners of the bounding box

    Returns:
        areas (dict): a dict where the keys are points and the values are the areas of the voronoi regions that correspond to that point
    """
    # first reflect points across boundary to bound all points inside the boundary
    points, refl_points = reflect_points(points, bounds)
    total_points = np.concatenate((points, refl_points))
    vor = Voronoi(total_points)

    # Identify the points that are in bounds
    pts_in_bounds = in_bounds(total_points, bounds)

    # Find the cells associated with those points.
    region_dict = get_regions_from_points(vor, pts_in_bounds)

    # Find the area of those cells
    area_dict = {}
    for k, v in region_dict.items():
        area_dict[k] = get_cell_area(v)

    return area_dict


def get_areas_from_points(point_dict, ballCarrierId=None, restriction=5) -> dict:
    """
    A wrapper function around _get_areas_from_points that directly takes the point_dict with keys as the nflId and points as the (x, y) tuple.
    If ballcarrierId and tacklerId are provided, they will be highlighted in the plot.
    If defenseIds and offenseIds are provided, they will be colored differently on the plot
    If ballCarrierId is provided, then the ballcarrier's voronoi region will be modified by placing a virtual player located at (pt[0] - 2*restriction, pt[1]), where pt is the ballcarrier's position
    This will effectively limit how far back behind the ballcarrier his voronoi region can extend.

    :param point_dict: dict where the keys are the nflIds and the values are the (x, y) positions of the players
    :param plot: boolean indicating whether to plot the voronoi diagram
    :param ballCarrierId: nflId representing the ball carrier.
    :param restriction (float): number of yards behind the ballcarrier's current x-position where his voronoi vertices will be included in his voronoi cell

    :return: dict where the keys are the nflIds and the values are the areas of the voronoi regions that correspond to that player
    """
    points = np.array([v for _, v in point_dict.items()])
    bounds = np.array([(0, 0), (120, 0), (120, 53), (120, 0)])

    if ballCarrierId is not None:
        assert (
            ballCarrierId in point_dict.keys()
        ), f"ballCarrierId {ballCarrierId} must be in point_dict.keys()"
        bc_point = point_dict[ballCarrierId]
        bc_mirror_pt = np.array([bc_point[0] - 2 * restriction, bc_point[1]])
        points = np.append(points, [bc_mirror_pt], axis=0)

    area_dict = _get_areas_from_points(points, bounds)

    # map cell area from points to nflIds
    new_area_dict = {}
    for k, v in point_dict.items():  # k = nflId, v = point
        new_area_dict[k] = area_dict[v]

    return new_area_dict


def plot_voronoi_cells(point_dict, xlim=[0, 120], ylim=[0, 53]) -> None:
    """
    Plots the voronoi cells of a frame. If offenseIds and defenseIds are provided, they will be colored differently on the plot.
    If ballcarrierId and tacklerId are provided, they will be highlighted in the plot.
    """
    points = [v for _, v in point_dict.items()]
    bounds = np.array([(0, 0), (120, 0), (120, 53), (120, 0)])
    points, refl_points = reflect_points(points, bounds)
    total_points = np.concatenate((points, refl_points))
    vor = Voronoi(total_points)
    voronoi_plot_2d(vor, show_vertices=False, show_points=False)

    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_players(
    point_dict,
    offenseIds=None,
    defenseIds=None,
    ballcarrierId=None,
    tacklerId=None,
    marker_size=3,
):
    """
    Given a dict with keys as nflIds and values as tuples of (x, y) position, plot players on an existing plot.
    If offenseIds and defenseIds are provided, they will be colored differently on the plot.
    If ballcarrierId and tacklerId are provided, they will be highlighted in the plot.

    :param offenseIds: list of nflIds representing the offense
    :param defenseIds: list of nflIds representing the defense
    :param ballcarrierId: nflId representing the ball carrier
    :param tacklerId: nflId representing the tackler
    """

    # plot offensive players
    if offenseIds is not None:
        offense_pts = [point_dict[offenseId] for offenseId in offenseIds]
        plt.scatter(
            [pt[0] for pt in offense_pts],
            [pt[1] for pt in offense_pts],
            color="red",
            alpha=0.5,
            s=marker_size,
            label="offense",
        )

    # plot defensive players
    if defenseIds is not None:
        defense_pts = [point_dict[defenseId] for defenseId in defenseIds]
        plt.scatter(
            [pt[0] for pt in defense_pts],
            [pt[1] for pt in defense_pts],
            color="blue",
            alpha=0.5,
            s=marker_size,
            label="defense",
        )

    # plot ballcarrier
    if ballcarrierId is not None:
        plt.scatter(
            point_dict[ballcarrierId][0],
            point_dict[ballcarrierId][1],
            color="red",
            marker="x",
            label="Ballcarrier",
        )

    # plot tackler
    if tacklerId is not None:
        plt.scatter(
            point_dict[tacklerId][0],
            point_dict[tacklerId][1],
            color="blue",
            marker="x",
            label="Tackler",
        )

    plt.legend()


def get_positions_from_dataframe(df: pd.DataFrame) -> list:
    """
    Given a dataframe of tracking data representing a single frame, return a dict of tuples where the key is the nflId and the value is the (x, y) position of the player

    Args:
        df (pd.DataFrame): a dataframe that represents a

    Returns:
        position_dict (dict): a dict where the keys are the nflIds and the values are the (x, y) positions of the players
    """

    position_dict = {
        nflId: (x, y) for nflId, x, y in zip(df["nflId"], df["x_clean"], df["y_clean"])
    }

    for k, v in position_dict.items():
        x_corr = min(max(0, v[0]), 120)
        y_corr = min(max(0, v[1]), 53)
        position_dict[k] = (x_corr, y_corr)

    return position_dict


# Functions for the influence model below


def single_player_influence(
    position: tuple, direction: float, speed: float
) -> Callable[[float, float], float]:
    """
    Given a player's position, direction, and speed, return a function that calculates the influence due to that player at any given point in space.
    The influence is calculated as a 2D Gaussian distribution that is rotated, stretched, and shifted based on a player's
    speed and direction.

    Args:
        position (tuple): tuple of floats representing the player's position
        direction (float): float representing the player's direction in degrees [0, 360]
        speed (float): float representing the player's speed (yards/s)

    Returns:
        influence (function): lambda function that takes a point and returns the influence at any arbitrary point due to this player's influence
    """

    cov = compute_cov_matrix(direction, speed)
    mean = np.array(
        [
            position[0] + 0.5 * speed * np.cos(np.pi * direction / 180),
            position[1] + 0.5 * speed * np.sin(np.pi * direction / 180),
        ]
    )

    det = np.linalg.det(cov)
    if det == 0:
        raise ValueError("Covariance matrix cannot be singular")
    cov_inv = np.linalg.inv(cov)

    return (
        lambda x_val, y_val: 1
        / np.sqrt((2 * np.pi) ** 2 * det)
        * np.exp(
            -0.5
            * (
                (np.array([x_val, y_val]) - mean)
                @ cov_inv
                @ (np.array([x_val, y_val]) - mean).T
            )
        )
    )


def multi_player_influence(
    positions: list, directions: list, speeds: list
) -> Callable[[float, float], float]:
    """
    Given a list of player positions, directions, and speeds, return a function that calculates the influence due to all players at any given point in space.
    The influence is calculated as the sum of the influence due to each individual player, which are each 2D gaussians given by single_player_influence.
    Note that this function only accepts influence of the same sign.

    Args:
        positions (list): list of tuples representing the player's positions
        directions (list): list of floats representing the player's directions in degrees [0, 360]
        speeds (list): list of floats representing the player's speeds (yards/s)

    Returns:
        influence (function): lambda function that takes a point and returns the influence at any arbitrary point due to all players' influences
    """
    influence_fns = [
        single_player_influence(position, direction, speed)
        for position, direction, speed in zip(positions, directions, speeds)
    ]

    return lambda x_val, y_val: sum(
        [influence_fn(x_val, y_val) for influence_fn in influence_fns]
    )


def team_influence(
    frame: pd.DataFrame, include_ballcarrier=False
) -> Callable[[float, float], float]:
    """
    Given a dataframe of a single frame, return the aggregate influence function due to all players on the field
    Offensive players contribute positive influence, while defensive players contribute negative influence

    Args:
        frame (pd.DataFrame): dataframe of a single frame, taken from tracking data.
        include_ballcarrier (boolean): If True, then ballcarrier is included in influence model. Otherwise, ballcarrier is ignored.

    Returns:
        lambda function that returns the net influence given a single x, y point
    """
    offenseIds = frame[frame.club == frame.possessionTeam].nflId.values
    if not include_ballcarrier:
        offenseIds = offenseIds[offenseIds != frame.ballCarrierId.values[0]]

    defenseIds = frame[frame.club == frame.defensiveTeam].nflId.values

    offensive_positions = frame[frame.nflId.isin(offenseIds)][
        ["x_clean", "y_clean"]
    ].values
    offensive_directions = frame[frame.nflId.isin(offenseIds)]["dir_clean"].values
    offensive_speeds = frame[frame.nflId.isin(offenseIds)]["s_clean"].values

    defensive_positions = frame[frame.nflId.isin(defenseIds)][
        ["x_clean", "y_clean"]
    ].values
    defensive_directions = frame[frame.nflId.isin(defenseIds)]["dir_clean"].values
    defensive_speeds = frame[frame.nflId.isin(defenseIds)]["s_clean"].values

    offensive_influence_fn = multi_player_influence(
        offensive_positions, offensive_directions, offensive_speeds
    )
    defensive_influence_fn = multi_player_influence(
        defensive_positions, defensive_directions, defensive_speeds
    )

    return lambda x_val, y_val: offensive_influence_fn(
        x_val, y_val
    ) - defensive_influence_fn(x_val, y_val)


def get_influence_for_frame(frame: pd.DataFrame) -> dict:
    """
    Given a dataframe of a single frame, return the influence at every single player point. Output is a dict with key = nflId and value = influence

    Args:
        frame (pd.DataFrame): dataframe of a single frame, taken from tracking data.

    Returns:
        dict: key = nflId and value = influence
        lambda: lambda function of the team_influence
    """
    team_influence_fn = team_influence(frame)
    position_dict = get_positions_from_dataframe(frame)

    influence_dict = {}
    for k, v in position_dict.items():
        influence_dict[k] = team_influence_fn(v[0], v[1])

    return influence_dict, team_influence_fn


def get_influence_for_blockers(frame: pd.DataFrame) -> dict:
    """
    Given a dataframe of a single frame, return the influence contributed only by the offensive players (blockers) at every single player point, including defenders. Output is a dict with key = nflId and value = influence

    Args:
        frame (pd.DataFrame): dataframe of a single frame, taken from tracking data.

    Returns:
        dict: key = nflId and value = influence
        lambda: lambda function of the team_influence
    """
    offense_frame = frame[frame.club == frame.possessionTeam]
    blocker_influence_fn = team_influence(offense_frame)
    position_dict = get_positions_from_dataframe(frame)

    blocker_influence_dict = {}
    for k, v in position_dict.items():
        blocker_influence_dict[k] = blocker_influence_fn(v[0], v[1])

    return blocker_influence_dict, blocker_influence_fn


def plot_influence_grid(influence_fn, grid, cmap_range=None):
    """
    Given an influence function and a grid, plot the influence intensity in a contour plot

    Args:
        influence_fn (function): function that takes an x, y point and returns the influence at that point
        grid (tuple): tuple of two np.arrays that represent the x and y values of the grid

    Returns:
        None
    """
    xx, yy = grid
    z = np.zeros_like(xx, dtype=np.float32)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            z[i, j] = influence_fn(xx[i, j], yy[i, j])

    if cmap_range is not None:
        c = plt.contourf(
            xx, yy, z, cmap="coolwarm", vmin=cmap_range[0], vmax=cmap_range[1]
        )
    else:
        c = plt.contourf(xx, yy, z, cmap="coolwarm")
    plt.colorbar(c, label="Team Influence (+ = offense)")


def compute_rotation_matrix(direction: float) -> np.array:
    """
    Given a direction, return a 2x2 rotation matrix

    Args:
        direction (float): float representing the player's direction in degrees.

    Returns:
        rotation_matrix (np.array): 2x2 matrix representing the rotation transformation
    """
    theta = np.radians(direction)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rotation_matrix


def compute_scaling_matrix(speed: float) -> np.array:
    """
    Given a speed, return a 2x2 scaling matrix

    Args:
        speed (float): float representing the player's speed in yards/s.

    Returns:
        scaling_matrix (np.array): 2x2 matrix representing the scaling transformation
    """
    max_speed = 18  # this is approximately the max speed seen over the tracking data
    influence_radius = 3  # this is a hyperparameter that says a player has an influence over 3 yards from where they are standing
    sx = influence_radius + influence_radius * speed / max_speed
    sy = influence_radius - influence_radius * speed / max_speed

    scaling_matrix = np.array([[sx / 2, 0], [0, sy / 2]])
    return scaling_matrix


def compute_cov_matrix(direction, speed):
    """
    Given a direction and speed, return the covariance matrix for the 2D Gaussian distribution that is
    rotated in the given direction, and scaled by the speed.

    Args:
        speed (float): float representing the player's speed in yards/s.
        direction (float): float representing the player's direction in degrees.

    Returns:
        cov_matrix (np.array): 2x2 matrix representing the covariance matrix
    """
    rotation_matrix = compute_rotation_matrix(direction)
    scaling_matrix = compute_scaling_matrix(speed)

    # COV = RSSR^(-1)
    return rotation_matrix @ scaling_matrix @ scaling_matrix @ rotation_matrix.T
