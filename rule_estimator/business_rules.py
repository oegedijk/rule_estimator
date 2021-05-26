__all__ = [
    'EmptyRule',
    'PredictionRule',
    'GreaterThan', 
    'GreaterEqualThan', 
    'LesserThan', 
    'LesserEqualThan',
    
    'GreaterThanNode', 
    'GreaterEqualThanNode', 
    'LesserThanNode', 
    'LesserEqualThanNode', 
]

from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from .core import BusinessRule, BinaryDecisionNode


class EmptyRule(BusinessRule):
    def __init__(self, prediction=None, default=None):
        super().__init__()

    def __rule__(self, X):
        return pd.Series(np.full(len(X), False))

    def __rulerepr__(self):
        return f"EmptyRule: Always predict {self.default}"

class PredictionRule(BusinessRule):
    def __init__(self, prediction=None):
        super().__init__()

    def __rule__(self, X):
        return pd.Series(np.full(len(X), True))

    def __rulerepr__(self):
        return f"PredictionRule: Always predict {self.prediction}"


class GreaterThan(BusinessRule):
    def __init__(self, col:str, cutoff:float, prediction:Union[float, int], default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame):
        return X[self.col] > self.cutoff

    def __rulerepr__(self):
        return f"GreaterThan: If {self.col} > {self.cutoff} then predict {self.prediction}"


class GreaterEqualThan(BusinessRule):
    def __init__(self, col:str, cutoff:float, prediction:Union[float, int], default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame):
        return X[self.col] >= self.cutoff

    def __rulerepr__(self):
        return f"GreaterEqualThan: If {self.col} >= {self.cutoff} then predict {self.prediction}"


class LesserThan(BusinessRule):
    def __init__(self, col:str, cutoff:float, prediction:Union[float, int], default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame):
        return X[self.col] < self.cutoff

    def __rulerepr__(self):
        return f"LesserThan: If {self.col} < {self.cutoff} then predict {self.prediction}"


class LesserEqualThan(BusinessRule):
    def __init__(self, col:str, cutoff:float, prediction:Union[float, int], default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame):
        return X[self.col] <= self.cutoff

    def __rulerepr__(self):
        return f"LesserEqualThan: If {self.col} <= {self.cutoff} then predict {self.prediction}"


class GreaterThanNode(BinaryDecisionNode):
    def __init__(self, col:str, cutoff:float,
                 if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame)->pd.Series:
        return X[self.col] > self.cutoff

    def __rulerepr__(self)->str:
        return f"GreaterThanNode: {self.col} > {self.cutoff}"


class GreaterEqualThanNode(BinaryDecisionNode):
    def __init__(self, col:str, cutoff:float,
                 if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame)->pd.Series:
        return X[self.col] >= self.cutoff

    def __rulerepr__(self)->str:
        return f"GreaterEqualThanNode: {self.col} >= {self.cutoff}"

class LesserThanNode(BinaryDecisionNode):
    def __init__(self, col:str, cutoff:float,
                 if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame)->pd.Series:
        return X[self.col] < self.cutoff

    def __rulerepr__(self)->str:
        return f"LesserThanNode: {self.col} < {self.cutoff}"

class LesserEqualThanNode(BinaryDecisionNode):
    def __init__(self, col:str, cutoff:float, 
                 if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame)->pd.Series:
        return X[self.col] <= self.cutoff

    def __rulerepr__(self)->str:
        return f"LesserEqualThanNode: {self.col} <= {self.cutoff}"