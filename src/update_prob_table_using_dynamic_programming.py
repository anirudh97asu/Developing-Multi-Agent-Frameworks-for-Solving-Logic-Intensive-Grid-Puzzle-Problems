import numpy as np
from collections import defaultdict

def solve_logic_puzzle(categories, items_by_category, rules):
    """
    Solve a logic puzzle using constraint propagation and return probability matrices.
    
    Parameters:
    -----------
    categories : list of str
        List of category names (e.g., ['Vintage', 'Wine', 'Type'])
    
    items_by_category : dict
        Dictionary mapping each category to its list of items
        e.g., {'Vintage': [1984, 1988, 1992, 1996], 
               'Wine': ['Annata Branco', 'Bianca Flaux', 'Ece Suss', 'Vendemmia'],
               'Type': ['gewurztraminer', 'merlot', 'pinot noir', 'riesling']}
    
    rules : list of dict
        List of rule dictionaries, each containing:
        - 'type': str - Type of rule ('direct_assignment', 'order', 'exact_difference', 'exclusive_or')
        - Other keys depend on the rule type (see examples below)
    
    Returns:
    --------
    dict of numpy.ndarray
        Dictionary mapping pairs of categories to their probability matrices
    """
    # Initialize matrices with uniform probabilities
    matrices = {}
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i < j:  # Only create unique pairs
                items1 = items_by_category[cat1]
                items2 = items_by_category[cat2]
                n1, n2 = len(items1), len(items2)
                matrices[(cat1, cat2)] = np.full((n1, n2), 1.0/n2)
    
    # Apply rules iteratively to converge to solution
    for _ in range(20):  # Multiple iterations for convergence
        # Apply each rule
        for rule in rules:
            apply_rule(rule, matrices, items_by_category, categories)
        
        # Ensure consistency across matrices
        ensure_consistency(matrices, categories, items_by_category)
        
        # Normalize all matrices
        for key in matrices:
            matrices[key] = normalize_matrix(matrices[key])
    
    return matrices

def apply_rule(rule, matrices, items_by_category, categories):
    """Apply a specific rule to update matrices."""
    rule_type = rule['type']
    
    if rule_type == 'direct_assignment':
        # e.g., {'type': 'direct_assignment', 'cat1': 'Vintage', 'item1': 1988, 'cat2': 'Type', 'item2': 'pinot noir'}
        cat1, cat2 = rule['cat1'], rule['cat2']
        item1, item2 = rule['item1'], rule['item2']
        if (cat1, cat2) in matrices:
            matrix = matrices[(cat1, cat2)]
            i = items_by_category[cat1].index(item1)
            j = items_by_category[cat2].index(item2)
            matrix[i, :] = 0.01  # Small value instead of 0 for numerical stability
            matrix[i, j] = 1.0
        elif (cat2, cat1) in matrices:
            matrix = matrices[(cat2, cat1)]
            i = items_by_category[cat2].index(item2)
            j = items_by_category[cat1].index(item1)
            matrix[i, :] = 0.01
            matrix[i, j] = 1.0
    
    elif rule_type == 'order':
        # e.g., {'type': 'order', 'cat': 'Vintage', 'item1': 'Ece Suss', 'item2': 'Annata Branco', 'relation': '>'}
        cat = rule['cat']
        item1_cat = next(c for c in categories if c != cat and rule['item1'] in items_by_category[c])
        item2_cat = next(c for c in categories if c != cat and rule['item2'] in items_by_category[c])
        
        item1 = rule['item1']
        item2 = rule['item2']
        relation = rule['relation']
        
        # Find relevant matrices
        if (cat, item1_cat) in matrices and (cat, item2_cat) in matrices:
            apply_order_constraint(
                cat, item1_cat, item2_cat, 
                item1, item2, relation,
                matrices, items_by_category
            )
    
    elif rule_type == 'exact_difference':
        # e.g., {'type': 'exact_difference', 'cat': 'Vintage', 'item1': 'Bianca Flaux', 'item2': 'Vendemmia', 'diff': 4}
        cat = rule['cat']
        item1_cat = next(c for c in categories if c != cat and rule['item1'] in items_by_category[c])
        item2_cat = next(c for c in categories if c != cat and rule['item2'] in items_by_category[c])
        
        item1 = rule['item1']
        item2 = rule['item2']
        diff = rule['diff']
        
        apply_difference_constraint(
            cat, item1_cat, item2_cat,
            item1, item2, diff,
            matrices, items_by_category
        )
    
    elif rule_type == 'exclusive_or':
        # e.g., {'type': 'exclusive_or', 'cat1': 'Wine', 'items1': ['Annata Branco', 'Bianca Flaux'], 
        #        'cat2': 'Type', 'item2': 'merlot'}
        cat1, cat2 = rule['cat1'], rule['cat2']
        items1, item2 = rule['items1'], rule['item2']
        
        apply_exclusive_or_constraint(
            cat1, cat2, items1, item2,
            matrices, items_by_category
        )

def apply_order_constraint(cat, item1_cat, item2_cat, item1, item2, relation, matrices, items_by_category):
    """Apply ordering constraint (e.g., Y_E > Y_A)"""
    cat_items = items_by_category[cat]
    
    # Get indices
    idx1 = items_by_category[item1_cat].index(item1)
    idx2 = items_by_category[item2_cat].index(item2)
    
    # Find or create relevant matrices
    matrix1 = None
    matrix2 = None
    
    if (cat, item1_cat) in matrices:
        matrix1 = matrices[(cat, item1_cat)]
    elif (item1_cat, cat) in matrices:
        matrix1 = matrices[(item1_cat, cat)].T
    
    if (cat, item2_cat) in matrices:
        matrix2 = matrices[(cat, item2_cat)]
    elif (item2_cat, cat) in matrices:
        matrix2 = matrices[(item2_cat, cat)].T
    
    if matrix1 is not None and matrix2 is not None:
        # Apply constraint
        for i, val1 in enumerate(cat_items):
            for j, val2 in enumerate(cat_items):
                if relation == '>' and val1 <= val2:
                    matrix1[i, idx1] *= 0.5
                    matrix2[j, idx2] *= 0.5
                elif relation == '<' and val1 >= val2:
                    matrix1[i, idx1] *= 0.5
                    matrix2[j, idx2] *= 0.5
                elif relation == '=' and val1 != val2:
                    matrix1[i, idx1] *= 0.5
                    matrix2[j, idx2] *= 0.5

def apply_difference_constraint(cat, item1_cat, item2_cat, item1, item2, diff, matrices, items_by_category):
    """Apply exact difference constraint (e.g., Y_B = Y_V - 4)"""
    cat_items = items_by_category[cat]
    
    # Get indices
    idx1 = items_by_category[item1_cat].index(item1)
    idx2 = items_by_category[item2_cat].index(item2)
    
    # Find or create relevant matrices
    matrix1 = None
    matrix2 = None
    
    if (cat, item1_cat) in matrices:
        matrix1 = matrices[(cat, item1_cat)]
    elif (item1_cat, cat) in matrices:
        matrix1 = matrices[(item1_cat, cat)].T
    
    if (cat, item2_cat) in matrices:
        matrix2 = matrices[(cat, item2_cat)]
    elif (item2_cat, cat) in matrices:
        matrix2 = matrices[(item2_cat, cat)].T
    
    if matrix1 is not None and matrix2 is not None:
        # Apply constraint
        for i, val1 in enumerate(cat_items):
            valid_val2 = val1 + diff
            if valid_val2 in cat_items:
                j = cat_items.index(valid_val2)
                # Increase probability for valid combinations
                matrix1[i, idx1] *= 2
                matrix2[j, idx2] *= 2
            else:
                # Decrease probability for invalid combinations
                matrix1[i, idx1] *= 0.5

def apply_exclusive_or_constraint(cat1, cat2, items1, item2, matrices, items_by_category):
    """Apply exclusive OR constraint (e.g., merlot is either Annata Branco or Bianca Flaux)"""
    # Find the relevant matrix
    matrix = None
    if (cat1, cat2) in matrices:
        matrix = matrices[(cat1, cat2)]
    elif (cat2, cat1) in matrices:
        matrix = matrices[(cat2, cat1)].T
    
    if matrix is not None:
        # Get indices
        item2_idx = items_by_category[cat2].index(item2)
        
        # Increase probability for the specified items
        for item1 in items1:
            item1_idx = items_by_category[cat1].index(item1)
            matrix[item1_idx, item2_idx] *= 2
        
        # Decrease probability for all other items
        for i, item1 in enumerate(items_by_category[cat1]):
            if item1 not in items1:
                matrix[i, item2_idx] *= 0.1

def ensure_consistency(matrices, categories, items_by_category):
    """Ensure consistency across all matrices."""
    # Create consistency matrices for triplets of categories
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i < j:
                for k, cat3 in enumerate(categories):
                    if j < k:
                        # We have a triplet (cat1, cat2, cat3)
                        reconcile_triplet(cat1, cat2, cat3, matrices, items_by_category)

def reconcile_triplet(cat1, cat2, cat3, matrices, items_by_category):
    """Reconcile probabilities across a triplet of categories."""
    n1 = len(items_by_category[cat1])
    n2 = len(items_by_category[cat2])
    n3 = len(items_by_category[cat3])
    
    # Temporary storage for new values
    new_values = defaultdict(list)
    
    # Find relevant matrices
    m12 = get_matrix(cat1, cat2, matrices)
    m13 = get_matrix(cat1, cat3, matrices)
    m23 = get_matrix(cat2, cat3, matrices)
    
    if m12 is None or m13 is None or m23 is None:
        return
    
    # For each triplet of items
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                # Calculate consistency value
                p12 = m12[i, j]
                p13 = m13[i, k]
                p23 = m23[j, k]
                
                # Geometric mean for consistency
                new_val = (p12 * p13 * p23) ** (1/3)
                
                new_values[(cat1, cat2, i, j)].append(new_val)
                new_values[(cat1, cat3, i, k)].append(new_val)
                new_values[(cat2, cat3, j, k)].append(new_val)
    
    # Update matrices with new values
    for key, vals in new_values.items():
        cat_i, cat_j, i, j = key
        avg_val = np.mean(vals)
        
        if (cat_i, cat_j) in matrices:
            matrices[(cat_i, cat_j)][i, j] = avg_val
        elif (cat_j, cat_i) in matrices:
            matrices[(cat_j, cat_i)][j, i] = avg_val

def get_matrix(cat1, cat2, matrices):
    """Get the matrix for two categories, handling transposition if needed."""
    if (cat1, cat2) in matrices:
        return matrices[(cat1, cat2)]
    elif (cat2, cat1) in matrices:
        return matrices[(cat2, cat1)].T
    return None

def normalize_matrix(matrix):
    """Normalize a matrix so rows sum to 1."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Handle division by zero
    row_sums[row_sums == 0] = 1
    return matrix / row_sums

def get_most_likely_assignments(matrices, categories, items_by_category):
    """Get the most likely assignments based on the matrices."""
    assignments = {}
    
    # For each pair of categories
    for cat1, cat2 in matrices.keys():
        matrix = matrices[(cat1, cat2)]
        items1 = items_by_category[cat1]
        items2 = items_by_category[cat2]
        
        # For each item in cat1, find most likely item in cat2
        for i, item1 in enumerate(items1):
            most_likely_idx = np.argmax(matrix[i])
            most_likely_item = items2[most_likely_idx]
            assignments[(item1, cat2)] = most_likely_item
    
    return assignments

def format_matrix(matrix, row_names, col_names):
    """Format a matrix for printing."""
    result = "        " + " ".join(f"{col:<15}" for col in col_names) + "\n"
    for i, row in enumerate(row_names):
        result += f"{row:<8}: " + " ".join(f"{matrix[i, j]:.4f}        " for j in range(len(col_names))) + "\n"
    return result

# Example usage for the wine puzzle
def solve_wine_puzzle():
    categories = ['Vintage', 'Wine', 'Type']
    items_by_category = {
        'Vintage': [1984, 1988, 1992, 1996],
        'Wine': ['Annata Branco', 'Bianca Flaux', 'Ece Suss', 'Vendemmia'],
        'Type': ['gewurztraminer', 'merlot', 'pinot noir', 'riesling']
    }
    
    rules = [
        # Rule 1: Ece Suss was bottled after Annata Branco
        {'type': 'order', 'cat': 'Vintage', 'item1': 'Ece S