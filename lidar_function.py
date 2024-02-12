import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree


"""Calculates various features for the LiDAR point clouds

Arguments
---------
LiDAR point: numpy.ndarray
            The co-ordinates(x, y, z) ofLiDAR point cloud, stacked into a numpy array

Neighborhood radius: neighborhood radius to be given as a parameter in the KD-Tree search.

Returns
-------
numpy.ndarray
    Returns a numpy array for each feature calculation function. These array can be used as a feature for machine learning model.
"""


def calculate_planarity(lidar_data, neighborhood_radius, output_queue, tree):
    planarity_values = np.zeros(len(lidar_data))

    try:
        for point_idx, point in tqdm(enumerate(lidar_data), total=len(lidar_data), desc="Computing planarity"):
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            neighbors = lidar_data[neighbor_indices]

            if neighbors.shape[0] >= 3:
                cov_matrix = np.cov(neighbors, rowvar=False)
                eigenvalues, _ = np.linalg.eig(cov_matrix)
                sorted_eigenvalues = np.sort(eigenvalues)

                planarity = (sorted_eigenvalues[1] - sorted_eigenvalues[0]) / (
                            sorted_eigenvalues[0] + sorted_eigenvalues[1] + sorted_eigenvalues[2])
                planarity_values[point_idx] = planarity
            else:
                planarity_values[point_idx] = np.nan
    except Exception as e:
        print(f" An error occured in calculate_linearity: {e}")
        raise

    result = planarity_values
    output_queue.put(result)


def calculate_verticality(lidar_data, neighborhood_radius, output_queue, tree):
    verticality_values = np.zeros(len(lidar_data))
    valid_indices = []

    for point_idx, point in tqdm(enumerate(lidar_data), total=len(lidar_data), desc="Computing verticality"):
        # Query the octree to find neighbors within the specified radius
        neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
        neighbors = lidar_data[neighbor_indices]

        if neighbors.shape[0] >= 3:
            cov_matrix = np.cov(neighbors, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            sorted_eigenvalues = np.sort(eigenvalues)
            verticality = 1 - (
                        sorted_eigenvalues[0] / (sorted_eigenvalues[0] + sorted_eigenvalues[1] + sorted_eigenvalues[2]))
            verticality_values[point_idx] = verticality
            valid_indices.append(point_idx)
        else:
            verticality_values[point_idx] = np.nan

    result = verticality_values
    output_queue.put(result)


def calculate_sphericity(lidar_data, neighborhood_radius, output_queue, tree):
    sphericity_values = np.zeros(len(lidar_data))

    for point_idx, point in tqdm(enumerate(lidar_data), total=len(lidar_data), desc="Computing sphericity"):
        neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
        neighbors = lidar_data[neighbor_indices]

        if neighbors.shape[0] >= 3:
            cov_matrix = np.cov(neighbors, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            sorted_eigenvalues = np.sort(eigenvalues)

            sphericity = sorted_eigenvalues[0] / (sorted_eigenvalues[0] + sorted_eigenvalues[1] + sorted_eigenvalues[2])

            sphericity_values[point_idx] = sphericity
        else:
            sphericity_values[point_idx] = np.nan

    result = sphericity_values
    output_queue.put(result)


def calculate_anisotropy(lidar_data, neighborhood_radius, output_queue, tree):
    anisotropy_values = np.zeros(len(lidar_data))

    for point_idx, point in tqdm(enumerate(lidar_data), total=len(lidar_data), desc="Computing anisotropy"):
        neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
        neighbors = lidar_data[neighbor_indices]

        if neighbors.shape[0] >= 3:
            cov_matrix = np.cov(neighbors, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            sorted_eigenvalues = np.sort(eigenvalues)

            anisotropy = (sorted_eigenvalues[2] - sorted_eigenvalues[0]) / (
                        sorted_eigenvalues[0] + sorted_eigenvalues[1] + sorted_eigenvalues[2])
            anisotropy_values[point_idx] = anisotropy
        else:
            anisotropy_values[point_idx] = np.nan

    result = anisotropy_values
    output_queue.put(result)


def calculate_surface_variation(lidar_data, neighborhood_radius, output_queue, tree):
    surface_variation_values = np.zeros(len(lidar_data))

    for point_idx, point in tqdm(enumerate(lidar_data), total=len(lidar_data), desc="Computing surface variation"):
        neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
        neighbors = lidar_data[neighbor_indices]
        if neighbors.shape[0] >= 3:
            cov_matrix = np.cov(neighbors, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            sorted_eigenvalues = np.sort(eigenvalues)
            first_eigenvalue = sorted_eigenvalues[0]
            eigenvalue_sum = np.sum(eigenvalues)
            surface_variation = first_eigenvalue / eigenvalue_sum
            surface_variation_values[point_idx] = surface_variation
        else:
            surface_variation_values[point_idx] = np.nan

    result = surface_variation_values
    output_queue.put(result)


def calculate_curvature(lidar_data, neighborhood_radius, output_queue, tree):
    num_points = len(lidar_data)
    curvature_values = np.zeros(num_points)

    for point_idx, point in tqdm(enumerate(lidar_data), total=num_points, desc="Computing curvature"):
        neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
        neighbors = lidar_data[neighbor_indices]

        if neighbors.shape[0] >= 3:
            cov_matrix = np.cov(neighbors, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            lambda_ = np.min(eigenvalues)
            ind = np.argmin(eigenvalues)
            curvature_values[point_idx] = lambda_ / np.sum(eigenvalues)
        else:
            curvature_values[point_idx] = np.nan

    result = curvature_values
    output_queue.put(result)


def calculate_omnivariance(lidar_data, neighborhood_radius, output_queue, tree):
    omnivariance_values = np.zeros(len(lidar_data))
    
    for point_idx, point in tqdm(enumerate(lidar_data), total=len(lidar_data), desc="Computing omnivariance"):
        neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
        neighbors = lidar_data[neighbor_indices]

        if neighbors.shape[0] >= 3:
            cov_matrix = np.cov(neighbors, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            omnivariance = np.prod(eigenvalues + 1e-10) ** (1 / 3)
            omnivariance_values[point_idx] = omnivariance
        else:
            omnivariance_values[point_idx] = np.nan

    result = omnivariance_values
    output_queue.put(result)


def calculate_linearity(lidar_data, neighborhood_radius, output_queue, tree):
    linearity_values = np.zeros(len(lidar_data))
    valid_indices = []

    for point_idx, point in tqdm(enumerate(lidar_data), total=len(lidar_data), desc="Computing linearity"):
        # Query the octree to find neighbors within the specified radius
        neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
        neighbors = lidar_data[neighbor_indices]

        if neighbors.shape[0] >= 3:
            cov_matrix = np.cov(neighbors, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            sorted_eigenvalues = np.sort(eigenvalues)

            linearity = (sorted_eigenvalues[2] - sorted_eigenvalues[1]) / (
                        sorted_eigenvalues[0] + sorted_eigenvalues[1] + sorted_eigenvalues[2])
            linearity_values[point_idx] = linearity
            valid_indices.append(point_idx)
        else:
            linearity_values[point_idx] = np.nan

    result = linearity_values
    output_queue.put(result)