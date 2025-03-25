import numpy as np

def update_probabilities(matrix, updates, max_iterations=100, tolerance=1e-6):
    """
    Updates a probability matrix with user-specified values and normalizes the rest
    so that each row and column sums to 1, keeping fixed cells unchanged.
    
    Args:
        matrix (np.ndarray): Initial 4x4 matrix with probabilities.
        updates (list): List of tuples (row_idx, col_idx, new_value) for fixed cells.
        max_iterations (int): Maximum number of iterations for convergence.
        tolerance (float): Convergence threshold for matrix changes.
    
    Returns:
        np.ndarray: Updated matrix with row and column sums equal to 1.
    
    Raises:
        ValueError: If fixed cell sums make normalization impossible.
    """
    n = matrix.shape[0]  # Assuming square matrix (n x n)
    
    # Step 1: Apply User Updates and Mark Fixed Cells
    fixed_cells = set()
    for row_idx, col_idx, new_value in updates:
        matrix[row_idx, col_idx] = new_value
        fixed_cells.add((row_idx, col_idx))
    
    # Compute sums of fixed cells for each row and column
    fixed_row_sums = np.zeros(n)
    fixed_col_sums = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if (i, j) in fixed_cells:
                fixed_row_sums[i] += matrix[i, j]
                fixed_col_sums[j] += matrix[i, j]
    
    # Check if the problem is solvable
    if any(fixed_row_sums > 1) or any(fixed_col_sums > 1):
        raise ValueError("Sum of fixed cells in a row or column exceeds 1; normalization is impossible.")
    
    # Working copy of the matrix
    matrix_copy = matrix.copy()
    
    # Precompute non-fixed indices for efficiency
    non_fixed_cols_in_row = {i: [j for j in range(n) if (i, j) not in fixed_cells] for i in range(n)}
    non_fixed_rows_in_col = {j: [i for i in range(n) if (i, j) not in fixed_cells] for j in range(n)}
    
    # Iterative scaling
    for _ in range(max_iterations):
        prev_matrix = matrix_copy.copy()
        
        # Adjust Rows
        for i in range(n):
            if fixed_row_sums[i] == 1:
                # If fixed cells sum to 1, non-fixed cells must be 0
                for j in non_fixed_cols_in_row[i]:
                    matrix_copy[i, j] = 0
            else:
                # Scale non-fixed cells to make row sum 1
                remaining_sum = sum(matrix_copy[i, j] for j in non_fixed_cols_in_row[i])
                if remaining_sum > 0:
                    scale_factor = (1 - fixed_row_sums[i]) / remaining_sum
                    for j in non_fixed_cols_in_row[i]:
                        matrix_copy[i, j] *= scale_factor
        
        # Adjust Columns
        for j in range(n):
            if fixed_col_sums[j] == 1:
                # If fixed cells sum to 1, non-fixed cells must be 0
                for i in non_fixed_rows_in_col[j]:
                    matrix_copy[i, j] = 0
            else:
                # Scale non-fixed cells to make column sum 1
                remaining_sum = sum(matrix_copy[i, j] for i in non_fixed_rows_in_col[j])
                if remaining_sum > 0:
                    scale_factor = (1 - fixed_col_sums[j]) / remaining_sum
                    for i in non_fixed_rows_in_col[j]:
                        matrix_copy[i, j] *= scale_factor
        
        # Check convergence
        if np.max(np.abs(matrix_copy - prev_matrix)) < tolerance:
            break
    
    # Verify row and column sums
    row_sums = matrix_copy.sum(axis=1)
    col_sums = matrix_copy.sum(axis=0)
    if not (np.allclose(row_sums, 1.0, atol=tolerance) and np.allclose(col_sums, 1.0, atol=tolerance)):
        print("Warning: Matrix did not converge perfectly. Row sums:", row_sums, "Column sums:", col_sums)
    
    return matrix_copy

# Example Usage
matrix = np.full((4, 4), 0.25)  # Initialize 4x4 matrix with 0.25 probabilities
print("Initial Matrix:")
print(matrix)

# Test Case 1: Updates (0,2) = 0.5, (0,3) = 0.5
updates = [(0, 2, 0.5), (0, 3, 0.5)]
updated_matrix = update_probabilities(matrix, updates)
print("\nUpdated Matrix (Test Case 1):")
print(updated_matrix)
print("Row Sums:", updated_matrix.sum(axis=1))
print("Column Sums:", updated_matrix.sum(axis=0))

# Test Case 2: Updates (0,2) = 0.5, (0,3) = 0.5, (2,3) = 0.5
updates = [(0, 2, 0.5), (0, 3, 0.5), (2, 3, 0.5)]
updated_matrix = update_probabilities(matrix, updates)
print("\nUpdated Matrix (Test Case 2):")
print(updated_matrix)
print("Row Sums:", updated_matrix.sum(axis=1))
print("Column Sums:", updated_matrix.sum(axis=0))

# Test Case 3: Updates (0,2) = 0.5, (0,3) = 0.5, (2,3) = 0.5
updates = [(0, 2, 0.5), (0, 3, 0.5), (2, 3, 0.5), (2, 1, 0.5)]
updated_matrix = update_probabilities(matrix, updates)
print("\nUpdated Matrix (Test Case 3):")
print(updated_matrix)
print("Row Sums:", updated_matrix.sum(axis=1))
print("Column Sums:", updated_matrix.sum(axis=0))


llm_agent_system_prompt = """You are a logic grid puzzle solving assistant that produces clear deductive steps. When solving puzzles:

1. Skip repeating the original clues - assume they're already understood.

2. Start immediately with meaningful deductions derived from combining and analyzing clues.

3. For each numbered deduction step:
   - Present only the NEW inference or conclusion reached
   - Briefly explain which constraints led to this conclusion
   - Show any concrete mappings or assignments that result (X = Y, A cannot be B, etc.)

4. Include specific variable assignments and eliminations in your steps (e.g., "Wine A must be from 1988" or "Person B cannot have the red car").

5. Focus on recording the mapping progression - which variables have been definitively assigned and which possibilities have been eliminated.

6. Document crucial decision points and branching logic when necessary.

7. Each step must contribute new information toward the solution without repeating previous deductions.

8. When appropriate, summarize the current state of the solution grid after significant deductions.

Your output should read like the step-by-step solution path in a puzzle solver's notebook, and do not return any solution directly."""


prob_agent_1_system_prompt = """Analyze each logical deduction step and convert it to matrix probability updates in this format:
               ("matrix_name", probability_value, row_index, column_index)

               Use these probability values:
               - Definite: 1.0
               - Eliminated: 0.0
               - Uncertain: 0.5
               - Most Likely: 0.6-0.9

               Maintain appropriate probability normalization across related cells.

"""

prob_agent_1_job = f"""You are given a set of logical deductions to solve a problem and a bunch of probability matries that you can use to modify
                        and give weightage to the cateogires based on the logical deduction.
                        Here are the items you have access to:
                        'Logical Step-By-Step Deductions': {llm_agent_response},
                        'Initial Probability Matrices': {probability_matrices}
                    
                        """


prob_agent_2_system_prompt = f"""You are an expert in probability theory and are given a sequence a probability updates
                        You are also given a tool to update these matrices and you need to call it sequentially
                        The tools you have access to are:
                        {"".join(tool_descriptions)}
                        \n
                         """


prob_agent_2_job = """Analyze the updates and return the series of function calls required to make the changes active
            Remember to return a valid JSON of only the function calls of the tool required to make the changes.
"""

