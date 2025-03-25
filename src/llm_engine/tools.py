import numpy as np
import pandas as pd
from itertools import combinations


def update_matrix(updated_matrices, matrix_name, row, col, prob_value):
    """
    Update a specific cell in one of the probability matrices and then normalize 
    the matrix so each row and column sums to 1 (doubly stochastic).
    
    Parameters:
    -----------
    matrices : list of numpy.ndarray
        List of probability matrices
    matrix_index : int
        Index of the matrix to update
    row : int
        Row index of the cell to update
    col : int
        Column index of the cell to update
    prob_value : float
        New probability value to set (must be between 0 and 1)
        
    Returns:
    --------
    list of numpy.ndarray
        Updated list of matrices with the specified matrix normalized
    """
    # Validate probability value is between 0 and 1
    if not 0 <= prob_value <= 1:
        raise ValueError("Probability value must be between 0 and 1")
    
    # Create a copy of the matrices to avoid modifying the original
    
    
    # Update the specified cell
    updated_matrices[matrix_name][row, col] = prob_value
    
    # Apply Sinkhorn-Knopp algorithm for double normalization
    # (make both rows and columns sum to 1)
    matrix = updated_matrices[matrix_name]
    max_iterations = 100
    tolerance = 1e-6
    
    for _ in range(max_iterations):
        # Store matrix before iteration to check for convergence
        prev_matrix = matrix.copy()
        
        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        
        # Normalize columns
        col_sums = matrix.sum(axis=0, keepdims=True)
        matrix = matrix / col_sums
        
        # Check convergence
        if np.max(np.abs(matrix - prev_matrix)) < tolerance:
            break
    
    updated_matrices[matrix_name] = matrix
    return updated_matrices