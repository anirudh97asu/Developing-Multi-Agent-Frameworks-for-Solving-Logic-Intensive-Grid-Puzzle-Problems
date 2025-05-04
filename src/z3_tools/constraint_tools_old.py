from z3 import (
    Int, Real, Solver, And, Or, Implies, If, Distinct, sat,
    ExprRef, BoolRef
)
from functools import reduce
from typing import List, Dict, Optional, Iterable, Any

def create_int_var(name: str) -> ExprRef:
    """
    Create an integer variable for use in Z3 constraints.
    
    Args:
        name (str): The name of the variable.
    
    Returns:
        ExprRef: A Z3 integer variable.
    """
    return Int(name)

def create_real_var(name: str) -> ExprRef:
    """
    Create a real variable for use in Z3 constraints.
    
    Args:
        name (str): The name of the variable.
    
    Returns:
        ExprRef: A Z3 real variable.
    """
    return Real(name)

def equal_constraint(a: ExprRef, b: ExprRef) -> BoolRef:
    """
    Create an equality constraint between two expressions.
    
    Args:
        a (ExprRef): First expression.
        b (ExprRef): Second expression.
        
    Returns:
        BoolRef: A constraint representing (a == b).
    """
    return a == b

def not_equal_constraint(a: ExprRef, b: ExprRef) -> BoolRef:
    """
    Create a non-equality constraint between two expressions.
    
    Args:
        a (ExprRef): First expression.
        b (ExprRef): Second expression.
        
    Returns:
        BoolRef: A constraint representing (a != b).
    """
    return a != b

def less_than_constraint(a: ExprRef, b: ExprRef) -> BoolRef:
    """
    Create a constraint ensuring that 'a' is strictly less than 'b'.
    
    Args:
        a (ExprRef): First expression.
        b (ExprRef): Second expression.
        
    Returns:
        BoolRef: A constraint representing (a < b).
    """
    return a < b

def less_equal_constraint(a: ExprRef, b: ExprRef) -> BoolRef:
    """
    Create a constraint ensuring that 'a' is less than or equal to 'b'.
    
    Args:
        a (ExprRef): First expression.
        b (ExprRef): Second expression.
        
    Returns:
        BoolRef: A constraint representing (a <= b).
    """
    return a <= b

def greater_than_constraint(a: ExprRef, b: ExprRef) -> BoolRef:
    """
    Create a constraint ensuring that 'a' is strictly greater than 'b'.
    
    Args:
        a (ExprRef): First expression.
        b (ExprRef): Second expression.
        
    Returns:
        BoolRef: A constraint representing (a > b).
    """
    return a > b

def greater_equal_constraint(a: ExprRef, b: ExprRef) -> BoolRef:
    """
    Create a constraint ensuring that 'a' is greater than or equal to 'b'.
    
    Args:
        a (ExprRef): First expression.
        b (ExprRef): Second expression.
        
    Returns:
        BoolRef: A constraint representing (a >= b).
    """
    return a >= b

def sum_constraint(variables: List[ExprRef], target_value: int) -> BoolRef:
    """
    Create a constraint that the sum of a list of variables equals a target value.
    
    Args:
        variables (List[ExprRef]): List of Z3 expressions.
        target_value (int): The numerical target value that the sum must equal.
        
    Returns:
        BoolRef: A constraint representing (sum(variables) == target_value).
    """
    return reduce(lambda a, b: a + b, variables) == target_value

def difference_constraint(a: ExprRef, b: ExprRef, target_value: int) -> BoolRef:
    """
    Create a constraint asserting the difference between two expressions equals a target value.
    
    Args:
        a (ExprRef): First expression.
        b (ExprRef): Second expression.
        target_value (int): The target value for (a - b).
        
    Returns:
        BoolRef: A constraint representing (a - b == target_value).
    """
    return a - b == target_value

def product_constraint(a: ExprRef, b: ExprRef, target_value: int) -> BoolRef:
    """
    Create a constraint asserting the product of two expressions equals a target value.
    
    Args:
        a (ExprRef): First expression.
        b (ExprRef): Second expression.
        target_value (int): The target product value.
        
    Returns:
        BoolRef: A constraint representing (a * b == target_value).
    """
    return a * b == target_value

def division_constraint(a: ExprRef, b: ExprRef, target_value: float) -> BoolRef:
    """
    Create a constraint asserting that the division of 'a' by 'b' equals a target value.
    
    Note:
        Ensure that 'b' is non-zero before using this constraint.
    
    Args:
        a (ExprRef): Numerator expression.
        b (ExprRef): Denominator expression.
        target_value (float): The target quotient.
        
    Returns:
        BoolRef: A constraint representing (a / b == target_value).
    """
    return a / b == target_value

def logical_and_constraint(constraints: List[BoolRef]) -> BoolRef:
    """
    Create a compound constraint that is the logical AND of the provided constraints.
    
    Args:
        constraints (List[BoolRef]): A list of Z3 boolean constraints.
        
    Returns:
        BoolRef: A constraint representing the conjunction (AND) of all constraints.
    """
    return And(constraints)

def logical_or_constraint(constraints: List[BoolRef]) -> BoolRef:
    """
    Create a compound constraint that is the logical OR of the provided constraints.
    
    Args:
        constraints (List[BoolRef]): A list of Z3 boolean constraints.
        
    Returns:
        BoolRef: A constraint representing the disjunction (OR) of all constraints.
    """
    return Or(constraints)

def implication_constraint(antecedent: BoolRef, consequent: BoolRef) -> BoolRef:
    """
    Create an implication constraint: if 'antecedent' holds, then 'consequent' must hold.
    
    Args:
        antecedent (BoolRef): The condition that implies another.
        consequent (BoolRef): The condition that is implied.
        
    Returns:
        BoolRef: A constraint representing (Implies(antecedent, consequent)).
    """
    return Implies(antecedent, consequent)

def abs_constraint(expr: ExprRef, expected: int) -> BoolRef:
    """
    Create a constraint to ensure the absolute value of an expression equals an expected value.
    
    Args:
        expr (ExprRef): A Z3 numeric expression.
        expected (int): The expected absolute value.
        
    Returns:
        BoolRef: A constraint representing (If(expr >= 0, expr, -expr) == expected).
    """
    return If(expr >= 0, expr, -expr) == expected

def in_domain_constraint(var: ExprRef, domain: Iterable[int]) -> BoolRef:
    """
    Create a constraint specifying that a variable must take one of the specified values.
    
    Args:
        var (ExprRef): The Z3 variable.
        domain (Iterable[int]): An iterable of allowable values.
        
    Returns:
        BoolRef: A constraint representing that var is equal to one of the values in domain.
    """
    return Or([var == value for value in domain])

def all_different_constraint(variables: List[ExprRef]) -> BoolRef:
    """
    Create a constraint ensuring that all variables in the provided list have distinct values.
    
    Args:
        variables (List[ExprRef]): List of Z3 expressions.
        
    Returns:
        BoolRef: A constraint that enforces all variables are different.
    """
    return Distinct(variables)

def solve_constraints(constraints: List[BoolRef], timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Solve a list of constraints using Z3 and return the solution model if satisfiable.
    
    Args:
        constraints (List[BoolRef]): A list of Z3 constraints.
        timeout (Optional[int]): Time limit in milliseconds for the solver. Defaults to None.
        
    Returns:
        Optional[Dict[str, Any]]: A dictionary mapping variable names to their values if the constraints 
                                  are satisfiable, or None if the constraints are unsatisfiable.
    """
    solver = Solver()
    if timeout is not None:
        solver.set("timeout", timeout)
    solver.add(constraints)
    if solver.check() == sat:
        model = solver.model()
        return {str(d): model[d] for d in model.decls()}
    else:
        return None