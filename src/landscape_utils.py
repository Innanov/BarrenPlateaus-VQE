"""
Landscape Computation Utilities Module 
======================================

This module provides specialized computational tools for analyzing quantum cost function
landscapes in the context of barren plateau research. It implements advanced techniques
for visualizing high-dimensional parameter spaces and computing landscape properties.

Key computational methods:
- 2D parameter space cross-sections with precise grid generation
- PCA-based dimensionality reduction for high-dimensional trajectory visualization  
- Gradient magnitude landscape computation for barren plateau identification

Author: Mostafa Atallah and Nouhaila Innan
Date: 2025
Version: 1.1.0 (Fixed)
License: Apache
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA


def compute_loss_landscape_2d(
    cost_function,
    center_params: np.ndarray,
    param_indices: Tuple[int, int] = None,
    param_range: float = 2.0,
    grid_size: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Compute 2D cross-section of the loss landscape around given parameters.

    Args:
        cost_function: The cost function to evaluate
        center_params: Center point for the landscape
        param_indices: Tuple of parameter indices to vary (if None, uses first two)
        param_range: Range of parameter values to explore (±param_range)
        grid_size: Resolution of the grid

    Returns:
        Dictionary containing X, Y coordinate grids and Z values (energies)
    """
    if param_indices is None:
        param_indices = (0, 1)  # Use first two parameters by default

    idx1, idx2 = param_indices

    # Create parameter grid with fixed spacing to ensure grid compatibility
    param1_values = np.linspace(
        center_params[idx1] - param_range, center_params[idx1] + param_range, grid_size
    )
    param2_values = np.linspace(
        center_params[idx2] - param_range, center_params[idx2] + param_range, grid_size
    )

    X, Y = np.meshgrid(param1_values, param2_values)
    Z = np.zeros_like(X)

    print(f"Computing {grid_size}x{grid_size} loss landscape...")

    # Evaluate cost function over the grid
    for i in range(grid_size):
        for j in range(grid_size):
            params_temp = center_params.copy()
            params_temp[idx1] = X[i, j]
            params_temp[idx2] = Y[i, j]

            try:
                Z[i, j] = cost_function(params_temp)
            except Exception as e:
                print(f"Error at grid point ({i}, {j}): {e}")
                Z[i, j] = np.inf

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{grid_size} rows")

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "param_indices": param_indices,
        "center_params": center_params,
        "param_range": param_range,
        "grid_size": grid_size,
    }


def compute_loss_landscape_pca(
    cost_function,
    trajectory_params: List[np.ndarray],
    grid_size: int = 30,
    scale_factor: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Compute 2D loss landscape using PCA projection of optimization trajectory.

    Args:
        cost_function: Quantum cost function C(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
        trajectory_params: List of parameter vectors from optimization trajectory
        grid_size: Resolution of the visualization grid (default: 30)
        scale_factor: Expansion factor for visualization bounds (default: 1.0)

    Returns:
        Dictionary containing:
        - 'X', 'Y': Coordinate grids in PCA space
        - 'Z': Energy values on the grid
        - 'trajectory_projected': Original trajectory in PCA coordinates
        - 'pca_components': Principal component vectors
        - 'explained_variance_ratio': Fraction of variance explained by each PC

    Theory:
        Given trajectory points {θ₁, θ₂, ..., θₜ}, PCA finds orthogonal directions
        v₁, v₂ that maximize the variance of projected points:

        max Var[θᵢ · v₁] subject to ||v₁|| = 1
        max Var[θᵢ · v₂] subject to ||v₂|| = 1, v₁ · v₂ = 0

        The 2D landscape is then C(θ₀ + α₁v₁ + α₂v₂) where θ₀ is the trajectory center.
    """
    # Check if trajectory is valid and has consistent dimensions
    if not trajectory_params or len(trajectory_params) < 1:
        raise ValueError("Trajectory must contain at least 1 parameter vector")

    # Convert to numpy arrays and handle dimension consistency
    trajectory_arrays = []
    for params in trajectory_params:
        if isinstance(params, (list, tuple)):
            trajectory_arrays.append(np.array(params))
        else:
            trajectory_arrays.append(params)

    # Handle mixed-dimension trajectories (e.g., from pretrained methods)
    param_lengths = [len(params) for params in trajectory_arrays]
    unique_lengths = list(set(param_lengths))

    if len(unique_lengths) > 1:
        print(f"  Warning: Trajectory has mixed parameter dimensions: {unique_lengths}")
        # Use the most common parameter dimension
        most_common_length = max(set(param_lengths), key=param_lengths.count)
        print(f"  Using parameters with dimension {most_common_length}")

        # Filter trajectory to only include parameters with the most common dimension
        filtered_trajectory = [
            params for params in trajectory_arrays if len(params) == most_common_length
        ]

        if len(filtered_trajectory) < 1:
            # Fallback: normalize all parameters to the first dimension
            target_length = unique_lengths[0]
            print(f"  Fallback: Normalizing all parameters to dimension {target_length}")

            normalized_trajectory = []
            for params in trajectory_arrays:
                if len(params) == target_length:
                    normalized_trajectory.append(params)
                elif len(params) > target_length:
                    # Truncate longer parameters
                    normalized_trajectory.append(params[:target_length])
                else:
                    # Pad shorter parameters with zeros
                    padded = np.zeros(target_length)
                    padded[: len(params)] = params
                    normalized_trajectory.append(padded)

            trajectory_arrays = normalized_trajectory
        else:
            trajectory_arrays = filtered_trajectory

    # Ensure we have at least 2 points for PCA
    if len(trajectory_arrays) == 1:
        # Create a second point by adding small perturbation
        original_point = trajectory_arrays[0]
        perturbed_point = original_point + np.random.normal(0, 0.01, len(original_point))
        trajectory_arrays = [original_point, perturbed_point]
        print("  Warning: Only 1 trajectory point, created perturbed second point for PCA")

    # Convert trajectory to numpy array
    try:
        trajectory = np.array(trajectory_arrays)
    except ValueError as e:
        print(f"  Error converting trajectory to array: {e}")
        # Create a simple 2-point trajectory
        param_dim = len(trajectory_arrays[0])
        trajectory = np.array([
            np.zeros(param_dim),
            np.random.normal(0, 0.1, param_dim)
        ])

    # Ensure we have a valid 2D array
    if trajectory.ndim != 2:
        raise ValueError(f"Trajectory must be 2D array, got shape {trajectory.shape}")

    # Perform PCA on the trajectory
    try:
        # Use minimum of 2 components or available dimensions
        n_components = min(2, trajectory.shape[0], trajectory.shape[1])
        pca = PCA(n_components=n_components)
        trajectory_projected = pca.fit_transform(trajectory)

        # Handle case where PCA returns only 1 component
        if trajectory_projected.shape[1] == 1:
            print("  Warning: PCA returned only 1 component, creating artificial 2nd component")
            second_component = np.random.normal(0, 0.1, trajectory_projected.shape[0])
            trajectory_projected = np.column_stack(
                [trajectory_projected[:, 0], second_component]
            )
            explained_variance_ratio = np.array([pca.explained_variance_ratio_[0], 0.0])
            pca_components = np.vstack([pca.components_[0], np.zeros(trajectory.shape[1])])
        else:
            explained_variance_ratio = pca.explained_variance_ratio_
            pca_components = pca.components_

    except Exception as e:
        print(f"  Error in PCA computation: {e}")
        # Fallback: use first two parameter dimensions if available
        print("  Fallback: Using first two parameter dimensions")
        if trajectory.shape[1] >= 2:
            trajectory_projected = trajectory[:, :2]
        else:
            # Create 2D projection artificially
            trajectory_projected = np.column_stack([
                trajectory[:, 0],
                np.random.normal(0, 0.1, trajectory.shape[0])
            ])
        explained_variance_ratio = np.array([1.0, 0.0])
        pca_components = np.eye(2, trajectory.shape[1])

    # Define grid bounds based on trajectory extent
    x_min, x_max = trajectory_projected[:, 0].min(), trajectory_projected[:, 0].max()
    y_min, y_max = trajectory_projected[:, 1].min(), trajectory_projected[:, 1].max()

    # Handle case where trajectory is a single point or line
    if x_max - x_min < 1e-6:
        x_range = 1.0
        x_center = x_min
    else:
        x_range = (x_max - x_min) * scale_factor
        x_center = (x_max + x_min) / 2

    if y_max - y_min < 1e-6:
        y_range = 1.0
        y_center = y_min
    else:
        y_range = (y_max - y_min) * scale_factor
        y_center = (y_max + y_min) / 2

    # Create standardized grid to ensure compatibility across methods
    x_vals = np.linspace(x_center - x_range / 2, x_center + x_range / 2, grid_size)
    y_vals = np.linspace(y_center - y_range / 2, y_center + y_range / 2, grid_size)

    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    print(f"Computing PCA-based loss landscape ({grid_size}x{grid_size})...")

    # Evaluate cost function over the PCA-projected grid
    trajectory_center = np.mean(trajectory, axis=0)
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Transform back to original parameter space
            try:
                pca_point = np.array([X[i, j], Y[i, j]])
                
                # Manual inverse transform: original_params = pca_point @ pca_components + center
                if pca_components.shape[0] >= 2:
                    original_params = np.dot(pca_point, pca_components) + trajectory_center
                else:
                    # Handle single component case
                    original_params = pca_point[0] * pca_components[0] + trajectory_center

                Z[i, j] = cost_function(original_params)
                
            except Exception as e:
                Z[i, j] = np.inf

        if (i + 1) % 5 == 0:
            print(f"Completed {i + 1}/{grid_size} rows")

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "trajectory_projected": trajectory_projected,
        "pca_components": pca_components,
        "explained_variance_ratio": explained_variance_ratio,
        "grid_size": grid_size,
        "x_range": x_range,
        "y_range": y_range,
        "x_center": x_center,
        "y_center": y_center,
        "trajectory_center": trajectory_center,
    }


def create_compatible_grid(
    center1: np.ndarray, center2: np.ndarray, param_range: float, grid_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a compatible grid for comparing two methods.

    This function ensures that both methods use identical X,Y coordinate grids,
    which is essential for accurate difference landscape computation.

    Args:
        center1: Center parameters for method 1
        center2: Center parameters for method 2
        param_range: Range of parameters to explore
        grid_size: Resolution of the grid

    Returns:
        X, Y meshgrids that are compatible for both methods
    """
    # Find the common bounds that encompass both methods
    min_param1 = min(center1[0] - param_range, center2[0] - param_range)
    max_param1 = max(center1[0] + param_range, center2[0] + param_range)
    min_param2 = min(center1[1] - param_range, center2[1] + param_range)
    max_param2 = max(center1[1] + param_range, center2[1] + param_range)

    # Create standardized grid with exact spacing
    param1_vals = np.linspace(min_param1, max_param1, grid_size)
    param2_vals = np.linspace(min_param2, max_param2, grid_size)

    X, Y = np.meshgrid(param1_vals, param2_vals)
    return X, Y


def compute_gradient_norm_function(cost_function, max_params: int = 10):
    """
    Create a gradient norm function with parameter limiting for efficiency.

    Args:
        cost_function: The cost function to differentiate
        max_params: Maximum number of parameters to include in gradient

    Returns:
        Function that computes gradient norm
    """

    def gradient_norm_function(params):
        """Compute the norm of the gradient at given parameters."""
        epsilon = 1e-6
        gradients = []

        # Limit to first N parameters for computational efficiency
        n_params = min(len(params), max_params)

        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon

            try:
                cost_plus = cost_function(params_plus)
                cost_minus = cost_function(params_minus)

                if np.isfinite(cost_plus) and np.isfinite(cost_minus):
                    grad_i = (cost_plus - cost_minus) / (2 * epsilon)
                    gradients.append(grad_i)
                else:
                    gradients.append(0.0)
            except Exception:
                gradients.append(0.0)

        return np.linalg.norm(gradients) if gradients else 0.0

    return gradient_norm_function


def compute_gradient_at_point(
    cost_function, params: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute gradient of cost function at given parameters using finite differences.

    Args:
        cost_function: Function to compute cost/energy
        params: Parameter values at which to compute gradient
        epsilon: Finite difference step size

    Returns:
        np.ndarray: Gradient vector ∇C(θ)
    """
    gradients = np.zeros(len(params))

    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon

        try:
            cost_plus = cost_function(params_plus)
            cost_minus = cost_function(params_minus)

            if np.isfinite(cost_plus) and np.isfinite(cost_minus):
                gradients[i] = (cost_plus - cost_minus) / (2 * epsilon)
            else:
                gradients[i] = 0.0
        except Exception as e:
            gradients[i] = 0.0

    return gradients


def validate_landscape_compatibility(
    landscape1: Dict, landscape2: Dict, tolerance: float = 1e-10
) -> bool:
    """
    Validate that two landscapes have compatible grids for difference computation.

    Args:
        landscape1: First landscape dictionary
        landscape2: Second landscape dictionary
        tolerance: Numerical tolerance for grid comparison

    Returns:
        True if landscapes are compatible, False otherwise
    """
    # Check if both landscapes exist
    if landscape1 is None or landscape2 is None:
        return False

    # Check grid shape compatibility
    if landscape1["X"].shape != landscape2["X"].shape:
        return False

    if landscape1["Y"].shape != landscape2["Y"].shape:
        return False

    # Check grid value compatibility
    x_compatible = np.allclose(landscape1["X"], landscape2["X"], rtol=tolerance)
    y_compatible = np.allclose(landscape1["Y"], landscape2["Y"], rtol=tolerance)

    return x_compatible and y_compatible


def compute_landscape_difference(
    landscape1: Dict, landscape2: Dict
) -> Dict[str, np.ndarray]:
    """
    Compute the difference between two compatible landscapes.

    Args:
        landscape1: First landscape (minuend)
        landscape2: Second landscape (subtrahend)

    Returns:
        Dictionary containing difference landscape data
    """
    if not validate_landscape_compatibility(landscape1, landscape2):
        raise ValueError("Landscapes are not compatible for difference computation")

    Z1, Z2 = landscape1["Z"], landscape2["Z"]
    finite_mask = np.isfinite(Z1) & np.isfinite(Z2)

    Z_diff = np.full_like(Z1, np.nan)

    # Compute difference only for finite values
    if np.any(finite_mask):
        Z_diff[finite_mask] = Z1[finite_mask] - Z2[finite_mask]

    return {
        "X": landscape1["X"],
        "Y": landscape1["Y"],
        "Z": Z_diff,
        "finite_mask": finite_mask,
        "max_abs_diff": (
            np.max(np.abs(Z_diff[finite_mask])) if np.any(finite_mask) else 0.0
        ),
    }


def analyze_landscape_features(landscape: Dict) -> Dict[str, Any]:
    """
    Analyze key features of a loss landscape.

    Args:
        landscape: Landscape dictionary containing X, Y, Z data

    Returns:
        Dictionary containing landscape analysis results
    """
    Z = landscape["Z"]
    finite_mask = np.isfinite(Z)

    if not np.any(finite_mask):
        return {
            "valid_points": 0,
            "min_energy": np.inf,
            "max_energy": np.inf,
            "energy_range": 0.0,
            "energy_variance": 0.0,
            "flatness_measure": 0.0,
        }

    Z_finite = Z[finite_mask]

    # Basic statistics
    min_energy = np.min(Z_finite)
    max_energy = np.max(Z_finite)
    energy_range = max_energy - min_energy
    energy_variance = np.var(Z_finite)

    # Flatness measure (smaller values indicate flatter landscapes)
    if energy_range > 0:
        # Normalize by range to get relative flatness
        flatness_measure = np.std(Z_finite) / energy_range
    else:
        flatness_measure = 0.0

    # Gradient magnitude estimation (simple finite difference)
    grad_x = np.gradient(Z, axis=1)
    grad_y = np.gradient(Z, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    avg_gradient = np.mean(grad_magnitude[finite_mask]) if np.any(finite_mask) else 0.0

    return {
        "valid_points": np.sum(finite_mask),
        "total_points": Z.size,
        "valid_fraction": np.sum(finite_mask) / Z.size,
        "min_energy": min_energy,
        "max_energy": max_energy,
        "energy_range": energy_range,
        "energy_variance": energy_variance,
        "energy_std": np.std(Z_finite),
        "flatness_measure": flatness_measure,
        "avg_gradient_magnitude": avg_gradient,
        "grid_size": landscape.get("grid_size", "unknown"),
    }


def safe_pca_transform(trajectory: List[np.ndarray], n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Safely perform PCA transformation on trajectory data with robust error handling.
    
    Args:
        trajectory: List of parameter vectors
        n_components: Number of PCA components to compute
        
    Returns:
        Tuple of (projected_trajectory, pca_object)
    """
    if not trajectory or len(trajectory) == 0:
        raise ValueError("Empty trajectory provided")
    
    # Convert to numpy array with dimension consistency
    trajectory_array = np.array(trajectory)
    
    # Handle single point case
    if len(trajectory) == 1:
        # Add a perturbed point for PCA
        perturbation = np.random.normal(0, 0.01, trajectory_array.shape[1])
        trajectory_array = np.vstack([trajectory_array, trajectory_array + perturbation])
    
    # Perform PCA with error handling
    try:
        pca = PCA(n_components=min(n_components, trajectory_array.shape[0], trajectory_array.shape[1]))
        projected = pca.fit_transform(trajectory_array)
        
        # Ensure we have 2D output
        if projected.shape[1] == 1:
            # Add artificial second dimension
            second_dim = np.random.normal(0, 0.1, projected.shape[0])
            projected = np.column_stack([projected, second_dim])
            
        return projected, pca
        
    except Exception as e:
        print(f"PCA failed: {e}, using fallback method")
        # Fallback: use first two dimensions or create artificial ones
        if trajectory_array.shape[1] >= 2:
            return trajectory_array[:, :2], None
        else:
            # Create 2D projection artificially
            projected = np.column_stack([
                trajectory_array[:, 0],
                np.random.normal(0, 0.1, len(trajectory_array))
            ])
            return projected, None