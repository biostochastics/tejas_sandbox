#!/usr/bin/env python3
"""
Safe Expression Evaluator for DOE Benchmark Framework

This module provides a secure alternative to eval() for evaluating
configuration conditions without executing arbitrary code.
"""

import ast
import operator as op
from typing import Dict, Any, Union, Optional


class SafeEvaluator:
    """
    Safe expression evaluator using AST parsing.
    
    This class safely evaluates boolean expressions on configuration
    dictionaries without the security risks of eval().
    """
    
    # Define allowed operations
    ALLOWED_OPS = {
        # Comparison operators
        ast.Eq: op.eq,
        ast.NotEq: op.ne,
        ast.Lt: op.lt,
        ast.LtE: op.le,
        ast.Gt: op.gt,
        ast.GtE: op.ge,
        ast.Is: op.is_,
        ast.IsNot: op.is_not,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
        
        # Boolean operators
        ast.And: lambda *args: all(args),
        ast.Or: lambda *args: any(args),
        
        # Unary operators
        ast.Not: op.not_,
        ast.UAdd: op.pos,
        ast.USub: op.neg,
        
        # Binary operators
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
    }
    
    @classmethod
    def safe_eval(cls, expr: str, variables: Dict[str, Any]) -> Any:
        """
        Safely evaluate an expression with given variables.
        
        Args:
            expr: Expression string to evaluate
            variables: Dictionary of variables available in expression
            
        Returns:
            Result of expression evaluation
            
        Raises:
            ValueError: If expression contains unsafe operations
            SyntaxError: If expression is malformed
        """
        if not expr or not isinstance(expr, str):
            return False
            
        try:
            # Parse the expression into an AST
            tree = ast.parse(expr, mode='eval')
            
            # Validate that the AST only contains safe operations
            cls._validate_ast(tree)
            
            # Evaluate the AST
            return cls._eval_node(tree.body, variables)
            
        except (SyntaxError, ValueError) as e:
            # Re-raise security exceptions, don't mask them
            if "not allowed" in str(e) or "Dunder" in str(e):
                raise
            # For other errors, log and return False
            import warnings
            warnings.warn(f"Failed to evaluate expression: {type(e).__name__}")
            return False
    
    @classmethod
    def _validate_ast(cls, tree: ast.AST) -> None:
        """
        Validate that AST only contains safe operations.
        
        Args:
            tree: AST to validate
            
        Raises:
            ValueError: If unsafe operations are found
        """
        for node in ast.walk(tree):
            # Check for dangerous operations
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Import statements are not allowed")
            elif isinstance(node, ast.FunctionDef):
                raise ValueError("Function definitions are not allowed")
            elif isinstance(node, ast.ClassDef):
                raise ValueError("Class definitions are not allowed")
            elif isinstance(node, ast.Delete):
                raise ValueError("Delete statements are not allowed")
            elif isinstance(node, (ast.While, ast.For)):
                raise ValueError("Loops are not allowed")
            elif isinstance(node, ast.Lambda):
                raise ValueError("Lambda functions are not allowed")
            elif isinstance(node, ast.ListComp):
                raise ValueError("List comprehensions are not allowed")
            elif isinstance(node, ast.DictComp):
                raise ValueError("Dict comprehensions are not allowed")
    
    @classmethod
    def _eval_node(cls, node: ast.AST, variables: Dict[str, Any]) -> Any:
        """
        Recursively evaluate AST nodes.
        
        Args:
            node: AST node to evaluate
            variables: Available variables
            
        Returns:
            Result of node evaluation
            
        Raises:
            ValueError: If unsupported node type is encountered
        """
        if isinstance(node, ast.Constant):
            # Python 3.8+ uses ast.Constant for literals
            return node.value
            
        elif isinstance(node, ast.Num):
            # Python 3.7 compatibility
            return node.n
            
        elif isinstance(node, ast.Str):
            # Python 3.7 compatibility
            return node.s
            
        elif isinstance(node, ast.NameConstant):
            # Python 3.7 compatibility (True, False, None)
            return node.value
            
        elif isinstance(node, ast.Name):
            # Variable reference
            if node.id in variables:
                return variables[node.id]
            else:
                raise ValueError(f"Undefined variable: {node.id}")
        
        elif isinstance(node, ast.Compare):
            # Comparison operations (e.g., x > 5, y == 'test')
            left = cls._eval_node(node.left, variables)
            
            for op, comparator in zip(node.ops, node.comparators):
                op_type = type(op)
                if op_type not in cls.ALLOWED_OPS:
                    raise ValueError(f"Unsupported operation: {op_type.__name__}")
                    
                right = cls._eval_node(comparator, variables)
                
                if not cls.ALLOWED_OPS[op_type](left, right):
                    return False
                    
                left = right  # For chained comparisons
                
            return True
        
        elif isinstance(node, ast.BoolOp):
            # Boolean operations (and, or)
            op_type = type(node.op)
            if op_type not in cls.ALLOWED_OPS:
                raise ValueError(f"Unsupported boolean operation: {op_type.__name__}")
            
            values = [cls._eval_node(value, variables) for value in node.values]
            
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
        
        elif isinstance(node, ast.UnaryOp):
            # Unary operations (not, -, +)
            op_type = type(node.op)
            if op_type not in cls.ALLOWED_OPS:
                raise ValueError(f"Unsupported unary operation: {op_type.__name__}")
                
            operand = cls._eval_node(node.operand, variables)
            return cls.ALLOWED_OPS[op_type](operand)
        
        elif isinstance(node, ast.BinOp):
            # Binary operations (+, -, *, /, etc.)
            op_type = type(node.op)
            if op_type not in cls.ALLOWED_OPS:
                raise ValueError(f"Unsupported binary operation: {op_type.__name__}")
                
            left = cls._eval_node(node.left, variables)
            right = cls._eval_node(node.right, variables)
            return cls.ALLOWED_OPS[op_type](left, right)
        
        elif isinstance(node, ast.Call):
            # Function calls - only allow specific safe functions
            func = cls._eval_node(node.func, variables)
            
            # Handle dict.get() method
            if isinstance(node.func, ast.Attribute):
                obj = cls._eval_node(node.func.value, variables)
                
                if isinstance(obj, dict) and node.func.attr == 'get':
                    # dict.get(key, default)
                    args = [cls._eval_node(arg, variables) for arg in node.args]
                    if len(args) >= 1:
                        key = args[0]
                        # Block access to dangerous keys
                        if isinstance(key, str) and (key.startswith('__') or key in ['eval', 'exec', 'compile', '__import__']):
                            raise ValueError(f"Access to key '{key}' not allowed")
                    if len(args) == 1:
                        return obj.get(args[0])
                    elif len(args) == 2:
                        return obj.get(args[0], args[1])
                    else:
                        raise ValueError("dict.get() takes 1 or 2 arguments")
                        
                elif node.func.attr == 'startswith' and isinstance(obj, str):
                    # str.startswith(prefix)
                    args = [cls._eval_node(arg, variables) for arg in node.args]
                    if len(args) == 1:
                        return obj.startswith(args[0])
                    else:
                        raise ValueError("str.startswith() takes 1 argument")
                        
                elif node.func.attr == 'endswith' and isinstance(obj, str):
                    # str.endswith(suffix)
                    args = [cls._eval_node(arg, variables) for arg in node.args]
                    if len(args) == 1:
                        return obj.endswith(args[0])
                    else:
                        raise ValueError("str.endswith() takes 1 argument")
                        
                else:
                    raise ValueError(f"Unsupported method call: {node.func.attr}")
            else:
                raise ValueError("Direct function calls are not allowed")
        
        elif isinstance(node, ast.Attribute):
            # Attribute access (e.g., obj.attr)
            # Block all dunder attributes
            if node.attr.startswith('__'):
                raise ValueError(f"Dunder attributes not allowed: {node.attr}")
            
            obj = cls._eval_node(node.value, variables)
            
            # Only allow safe attributes on safe types
            safe_types = (str, dict)
            if not isinstance(obj, safe_types):
                raise ValueError(f"Attribute access not allowed on type {type(obj).__name__}")
            
            # Only allow specific safe methods
            if node.attr in ['get', 'startswith', 'endswith', 'keys', 'values', 'items']:
                # Additional validation for dict methods
                if isinstance(obj, dict) and node.attr in ['get', 'keys', 'values', 'items']:
                    return getattr(obj, node.attr)
                elif isinstance(obj, str) and node.attr in ['startswith', 'endswith']:
                    return getattr(obj, node.attr)
                else:
                    raise ValueError(f"Method '{node.attr}' not allowed on type {type(obj).__name__}")
            else:
                raise ValueError(f"Attribute not allowed: {node.attr}")
        
        elif isinstance(node, ast.Subscript):
            # Subscript access (e.g., dict['key'], list[0])
            obj = cls._eval_node(node.value, variables)
            
            if isinstance(node.slice, ast.Index):
                # Python 3.7-3.8 compatibility
                key = cls._eval_node(node.slice.value, variables)
            else:
                # Python 3.9+
                key = cls._eval_node(node.slice, variables)
                
            try:
                return obj[key]
            except (KeyError, IndexError, TypeError):
                return None
        
        elif isinstance(node, ast.List):
            # List literal
            return [cls._eval_node(elem, variables) for elem in node.elts]
        
        elif isinstance(node, ast.Dict):
            # Dict literal
            keys = [cls._eval_node(k, variables) for k in node.keys]
            values = [cls._eval_node(v, variables) for v in node.values]
            return dict(zip(keys, values))
        
        elif isinstance(node, ast.Tuple):
            # Tuple literal
            return tuple(cls._eval_node(elem, variables) for elem in node.elts)
        
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")


def safe_eval_condition(condition: str, config: Dict[str, Any]) -> bool:
    """
    Convenience function to evaluate a condition against a config dict.
    
    Args:
        condition: Condition string to evaluate
        config: Configuration dictionary
        
    Returns:
        Boolean result of condition evaluation
    """
    return SafeEvaluator.safe_eval(condition, {"config": config})
