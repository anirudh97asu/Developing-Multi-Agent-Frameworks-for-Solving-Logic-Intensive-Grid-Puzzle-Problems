from z3 import Solver, Int, Bool, Or, And, Not, Xor, Implies, Distinct, If, sat, BoolRef
from typing import Dict, List, Union

class ConstraintSolver:
    """
    Wrapper around Z3 Solver providing category-based and individual constraint operations.

    Attributes:
        solver: Z3 Solver instance.
        categories: Mapping from category names to list of item identifiers.
        vars: Mapping from variable keys ("category|item") to Z3 Int/Bool variables.
    """
    def __init__(self, categories: Dict[str, List[str]]) -> None:
        """
        Initialize the wrapper with categories but do not auto-create variables.

        Inputs:
            categories: dict mapping category name to list of item names
        """
        self.solver = Solver()
        self.categories = categories
        self.vars: Dict[str, Union[Int, Bool]] = {}

    def create_category_vars(self, category: str, dtype: str = 'int') -> None:
        """
        Declare variables for all items in a category with specified data type and optional bounds.

        Inputs:
            category: Category identifier matching a key in self.categories
            dtype: 'int' for Int, 'bool' for Bool
        """
        
        if self.vars.get(category) is None:
            self.vars[category] = {}
        
        items = self.categories.get(category, [])
        
        for item in items:
            if dtype == 'bool':
                v = Bool(item)
            else:
                v = Int(item)
            self.vars[category][item] = v

        return

    def set_category_domain(self, category: str, lower: int = None, upper: int = None) -> None:
        """
        Adjust domain bounds for all integer variables in a given category using add_constraint_to_solver.

        Inputs:
            category: Category identifier
            lower: Minimum allowed value (inclusive)
            upper: Maximum allowed value (inclusive)
        """
        for item in self.categories.get(category, []):
            v = self.vars[category][item]
            if v is not None:
                if lower is not None:
                    self.add_constraint_to_solver(v >= lower)
                if upper is not None:
                    self.add_constraint_to_solver(v <= upper)

        return

    def add_distinct(self, category: str) -> None:
        """
        Add a distinct constraint for all variables within the specified category.
        
        This function collects all variables associated with items in the given category
        and adds a constraint to the solver ensuring that each variable must have a 
        distinct value from all others in that category.
        
        Args:
            category (str): The category name for which to add the distinct constraint.
                            Must be a valid key in the self.categories dictionary.
        Returns:
            None: This function does not return a value.
        """
        vars_ = []
        category_list = self.categories[category]
        for item in category_list:
            var = self.vars[category][item]
            vars_.append(var)
        
        self.add_constraint_to_solver(Distinct(vars_))

        return
    
    def add_constraint_to_solver(self, constraint: Union[BoolRef, list[BoolRef]]) -> None:
        """
        Adds a constraint to the Z3 solver instance.
        
        Args:
            constraint Union[BoolRef, list]: The Z3 constraint to add to the solver
        
        Returns:
            None
        """
        if isinstance(constraint, BoolRef):
            self.solver.add(constraint)
        elif isinstance(constraint, list):
            for constrain in constraint:
                self.solver.add(constrain)
        
        return

    def increment_val(self, category: str, item: str, value: str) -> int:
        """
        Increment a variable's value by the specified amount.
        
        Args:
            category (str): The category of the variable in self.vars
            item (str): The specific item/key within the category
            value (str): The value to increment by (will be converted to int)
        
        Returns:
            int: The updated value after incrementing
        """
        var_ = self.vars[category][item]
        if var_ is not None:
            var_ += int(value)
        return var_

    def decrement_val(self, category: str, item: str, value: str) -> int:
        """
        Decrement a variable's value by the specified amount.
        
        Args:
            category (str): The category of the variable in self.vars
            item (str): The specific item/key within the category
            value (str): The value to decrement by (will be converted to int)
        
        Returns:
            int: The updated value after decrementing
        """
        var_ = self.vars[category][item]
        if var_ is not None:
            var_ -= int(value)
        return var_

    def multiply_val(self, category: str, item: str, value: str) -> int:
        """
        Multiply a variable's value by the specified amount.
        
        Args:
            category (str): The category of the variable in self.vars
            item (str): The specific item/key within the category
            value (str): The value to multiply by (will be converted to int)
        
        Returns:
            int: The updated value after multiplication
        """
        var_ = self.vars[category][item]
        if var_ is not None:
            var_ *= int(value)
        return var_

    def divide_val(self, category: str, item: str, value: str) -> int:
        """
        Perform integer division on a variable's value by the specified amount.
        
        Args:
            category (str): The category of the variable in self.vars
            item (str): The specific item/key within the category
            value (str): The divisor (will be converted to int)
        
        Returns:
            int: The updated value after integer division
        """
        var_ = self.vars[category][item]
        if var_ is not None:
            var_ = var_ // int(value)
        return var_

    def modulo_val(self, category: str, item: str, value: str) -> int:
        """
        Calculate the modulo of a variable's value by the specified amount.
        
        Args:
            category (str): The category of the variable in self.vars
            item (str): The specific item/key within the category
            value (str): The modulo value (will be converted to int)
        
        Returns:
            int: The updated value after modulo operation
        """
        var_ = self.vars[category][item]
        if var_ is not None:
            var_ %= int(value)
        return var_

    def add_eq(self, category_a: str, a: str, category_b: str, b: Union[str, int]) -> BoolRef:
        """
        Creates a constraint that variable a equals b.
        
        Args:
            category_a (str): The first category to match in self.vars lookup table
            item_a (str): The specific item from the matched category
            category_b (str): The second category to match in self.vars lookup table
            b (Union[str, int]): The specific item from the matched category. Either a (str) or a literal value (int)
        
        Returns:
            BoolRef: A Z3 boolean reference representing the constraint (a == b)
        """
        left = self.vars[category_a][a]
        right = self.vars[category_b][b] if isinstance(b, str) else b
        return left == right

    def add_ne(self, category_a: str, item_a: str, category_b: str, item_b: Union[str, int]) -> BoolRef:
        """
        Creates a constraint that variable a is not equal to b.
        
        Args:
            category_a (str): The first category to match in self.vars lookup table
            item_a (str): The specific item from the matched category
            category_b (str): The second category to match in self.vars lookup table
            item_b (Union[str, int]): The specific item from the matched category. Either a (str) or a literal value (int)
        
        Returns:
            BoolRef: A Z3 boolean reference representing the constraint (a != b)
        """
        left = self.vars[category_a][item_a]
        right = self.vars[category_b][item_b] if isinstance(item_b, str) else item_b
        return left != right

    def add_lt(self, category_a: str, a: str, category_b: str, b: Union[str, int]) -> BoolRef:
        """
        Creates a constraint that variable a is less than b.
        
        Args:
            category_a (str): The first category to match in self.vars lookup table
            item_a (str): The specific item from the matched category
            category_b (str): The second category to match in self.vars lookup table
            b (Union[str, int]): The specific item from the matched category. Either a (str) or a literal value (int)
        
        Returns:
            BoolRef: A Z3 boolean reference representing the constraint (a < b)
        """
        left = self.vars[category_a][a]
        right = self.vars[category_b][b] if isinstance(b, str) else b
        return left < right

    def add_le(self, category_a: str, a: str, category_b: str, b: Union[str, int]) -> BoolRef:
        """
        Creates a constraint that variable a is less than or equal to b.
        
        Args:
            category_a (str): The first category to match in self.vars lookup table
            item_a (str): The specific item from the matched category
            category_b (str): The second category to match in self.vars lookup table
            b (Union[str, int]): The specific item from the matched category. Either a (str) or a literal value (int)
        
        Returns:
            BoolRef: A Z3 boolean reference representing the constraint (a <= b)
        """
        left = self.vars[category_a][a]
        right = self.vars[category_b][b] if isinstance(b, str) else b
        return left <= right

    def add_gt(self, category_a: str, a: str, category_b: str, b: Union[str, int]) -> BoolRef:
        """
        Creates a constraint that variable a is greater than b.
        
        Args:
            category_a (str): The first category to match in self.vars lookup table
            item_a (str): The specific item from the matched category
            category_b (str): The second category to match in self.vars lookup table
            b (Union[str, int]): The specific item from the matched category. Either a (str) or a literal value (int)
        
        Returns:
            BoolRef: A Z3 boolean reference representing the constraint (a > b)
        """
        left = self.vars[category_a][a]
        right = self.vars[category_b][b] if isinstance(b, str) else b
        return left > right

    def add_ge(self, category_a: str, a: str, category_b: str, b: Union[str, int]) -> BoolRef:
        """
        Creates a constraint that variable a is greater than or equal to b.
        
        Args:
            category_a (str): The first category to match in self.vars lookup table
            item_a (str): The specific item from the matched category
            category_b (str): The second category to match in self.vars lookup table
            b (Union[str, int]): The specific item from the matched category. Either a (str) or a literal value (int)
        
        Returns:
            BoolRef: A Z3 boolean reference representing the constraint (a >= b)
        """
        left = self.vars[category_a][a]
        right = self.vars[category_b][b] if isinstance(b, str) else b
        return left >= right

    def add_and(self, constraint_A: BoolRef, constraint_B: BoolRef) -> BoolRef:
        """
        This function takes in two constraints from Z3 solver library and performs And 
        on top of it.
        
        Args:
            constraint_A (BoolRef): The first Z3 boolean constraint
            constraint_B (BoolRef): The second Z3 boolean constraint
        
        Returns:
            BoolRef: A Z3 boolean reference representing (constraint_A AND constraint_B)
        """
        return And(constraint_A, constraint_B)
    
    def add_or(self, constraint_A: BoolRef, constraint_B: BoolRef) -> BoolRef:
        """
        This function takes in two constraints from Z3 solver library and performs Or 
        on top of it.
        
        Args:
            constraint_A (BoolRef): The first Z3 boolean constraint
            constraint_B (BoolRef): The second Z3 boolean constraint
        
        Returns:
            BoolRef: A Z3 boolean reference representing (constraint_A OR constraint_B)
        """
        return Or(constraint_A, constraint_B)

    def add_xor(self, constraint_A: BoolRef, constraint_B: BoolRef) -> BoolRef:
        """
        This function takes in two constraints from Z3 solver library and performs Xor 
        on top of it.
        
        Args:
            constraint_A (BoolRef): The first Z3 boolean constraint
            constraint_B (BoolRef): The second Z3 boolean constraint
        
        Returns:
            BoolRef: A Z3 boolean reference representing (constraint_A XOR constraint_B)
        """
        return Xor(constraint_A, constraint_B)

    def add_nand(self, constraint_A: BoolRef, constraint_B: BoolRef) -> BoolRef:
        """
        This function takes in two constraints from Z3 solver library and performs Nand
        on top of it.
        
        Args:
            constraint_A (BoolRef): The first Z3 boolean constraint
            constraint_B (BoolRef): The second Z3 boolean constraint
        
        Returns:
            BoolRef: A Z3 boolean reference representing NOT(constraint_A AND constraint_B)
        """
        return Not(And(constraint_A, constraint_B))

    def add_nor(self, constraint_A: BoolRef, constraint_B: BoolRef) -> BoolRef:
        """
        This function takes in two constraints from Z3 solver library and performs Nor 
        on top of it.
        
        Args:
            constraint_A (BoolRef): The first Z3 boolean constraint
            constraint_B (BoolRef): The second Z3 boolean constraint
        
        Returns:
            BoolRef: A Z3 boolean reference representing NOT(constraint_A OR constraint_B)
        """
        return Not(Or(constraint_A, constraint_B))
    

    def add_implies(self, category_a: str, item_a: str, category_b: str, item_b: str) -> BoolRef:
        """Constrain implication a -> b over variable keys."""
        return Implies(self.vars[category_a][item_a], self.vars[category_b][item_b])

    def check(self) -> bool:
        """Check if current constraints are satisfiable."""
        return self.solver.check() == sat

    def model(self):
        """Get the model after satisfiability check."""
        return self.solver.model() if self.solver.check() == sat else None
