__all__ = [
    'CaseWhen',
    'DummyRule',
    'GreaterThan', 
    'GreaterEqualThan', 
    'LesserThan', 
    'LesserEqualThan',
    

    'GreaterThanNode', 
    'GreaterEqualThanNode', 
    'LesserThanNode', 
    'LesserEqualThanNode', 
]

from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .core import BusinessRule, BinaryDecisionNode


class DummyRule(BusinessRule):
    def __init__(self, default=None):
        super().__init__()

    def predict(self, X:pd.DataFrame):
        return np.full(len(X), self.default)

    def __rulerepr__(self):
        return f"DummyRule: Always predict {self.default}"


class GreaterThan(BusinessRule):
    def __init__(self, col:str, cutoff:float, prediction:Union[float, int], default=None):
        super().__init__()

    def predict(self, X:pd.DataFrame):
        return np.where(X[self.col] > self.cutoff, self.prediction, self.default)

    def __rulerepr__(self):
        return f"If {self.col} > {self.cutoff} then predict {self.prediction}"


class GreaterEqualThan(BusinessRule):
    def __init__(self, col:str, cutoff:float, prediction:Union[float, int], default=None):
        super().__init__()

    def predict(self, X:pd.DataFrame):
        return np.where(X[self.col] >= self.cutoff, self.prediction, self.default)

    def __rulerepr__(self):
        return f"If {self.col} >= {self.cutoff} then predict {self.prediction}"


class LesserThan(BusinessRule):
    def __init__(self, col:str, cutoff:float, prediction:Union[float, int], default=None):
        super().__init__()

    def predict(self, X:pd.DataFrame):
        return np.where(X[self.col] < self.cutoff, self.prediction, self.default)

    def __rulerepr__(self):
        return f"If {self.col} < {self.cutoff} then predict {self.prediction}"


class LesserEqualThan(BusinessRule):
    def __init__(self, col:str, cutoff:float, prediction:Union[float, int], default=None):
        super().__init__()

    def predict(self, X:pd.DataFrame):
        return np.where(X[self.col] <= self.cutoff, self.prediction, self.default)

    def __rulerepr__(self):
        return f"If {self.col} <= {self.cutoff} then predict {self.prediction}"



class CaseWhen(BusinessRule):
    """Execute a list of rules one-by-one. 

    Args:
        BusinessRule ([type]): [description]
    """
    def __init__(self, rules:List[BusinessRule], default=None):
        super().__init__()

    def predict(self, X):
        y = np.full(len(X), np.nan)
        for rule in self.rules:
            y[np.isnan(y)] = rule.predict(X[np.isnan(y)])
        if not np.isnan(self.default):
            y = np.where(np.isnan(y), self.default, y)
        return y

    def __rulerepr__(self):
        return "CaseWhen"


class GreaterThanNode(BinaryDecisionNode):
    def __init__(self, col:str, cutoff:float,
                 if_true:BusinessRule, if_false:BusinessRule):
        super().__init__()

    def predict(self, X):
        return np.where(X[self.col] > self.cutoff,
                        self.if_true.predict(X), self.if_false.predict(X))

    def __rulerepr__(self):
        return f"BinaryDecisionNode {self.col} > {self.cutoff}"


class GreaterEqualThanNode(BinaryDecisionNode):
    def __init__(self, col:str, cutoff:float,
                 if_true:BusinessRule, if_false:BusinessRule):
        super().__init__()

    def predict(self, X):
        return np.where(X[self.col] >= self.cutoff,
                        self.if_true.predict(X), self.if_false.predict(X))

    def __rulerepr__(self):
        return f"BinaryDecisionNode {self.col} >= {self.cutoff}"


class LesserThanNode(BinaryDecisionNode):
    def __init__(self, col:str, cutoff:float,
                 if_true:BusinessRule, if_false:BusinessRule):
        super().__init__()

    def predict(self, X):
        return np.where(X[self.col] < self.cutoff, self.if_true.predict(X), self.if_false.predict(X))

    def __rulerepr__(self):
        return f"BinaryDecisionNode {self.col} < {self.cutoff}"


class LesserEqualThanNode(BinaryDecisionNode):
    def __init__(self, col:str, cutoff:float, if_true:BusinessRule, if_false:BusinessRule):
        super().__init__()

    def predict(self, X):
        return np.where(X[self.col] <= self.cutoff, self.if_true.predict(X), self.if_false.predict(X))

    def __rulerepr__(self):
        return f"BinaryDecisionNode {self.col} <= {self.cutoff}"