__all__ = [
    'BinarySplit',
    'IsInSplit',
    'GreaterThanSplit', 
    'GreaterEqualThanSplit', 
    'LesserThanSplit', 
    'LesserEqualThanSplit', 
    'RangeSplit',
    'MultiRangeSplit',
    'MultiRangeAnySplit'
]

from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from igraph import Graph

from .businessrule import BusinessRule, generate_range_mask
from .rules import EmptyRule


class BinarySplit(BusinessRule):
    def __init__(self):
        self._store_child_params(level=2)
        
        if not hasattr(self, "default"):
            self.default = None
        if self.default is None:
            self.default = np.nan
            
        assert hasattr(self, "if_true")
        if self.if_true is None:
            self.if_true = EmptyRule()
            self._stored_params['if_true'] = self.if_true
        assert isinstance(self.if_true, BusinessRule)
        
        assert hasattr(self, "if_false")
        if self.if_false is None:
            self.if_false = EmptyRule()
            self._stored_params['if_false'] = self.if_false
        assert isinstance(self.if_false, BusinessRule) 

    def predict(self, X:pd.DataFrame)->np.ndarray:
        y = np.full(len(X), np.nan)
        mask = self.__rule__(X)
        y[mask] = self.if_true.predict(X[mask])
        y[~mask] = self.if_false.predict(X[~mask])
        
        if not np.isnan(self.default):
            y = np.where(np.isnan(y), self.default, y)
            
        return y    
    
    def score_rule(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray], 
                   scores_df:pd.DataFrame=None, is_classifier:bool=False)->pd.DataFrame:
        # first predict without filling in the default
        if not np.isnan(self.default):
            old_default = self.default
            self.default = np.nan
            y_preds = self.predict(X)
            self.default = old_default 
        else:
            y_preds = self.predict(X)
            
        mask = np.invert(np.isnan(y_preds))
        scores_df = self._score_rule(y, y_preds, mask,
                                     prediction=np.nan, default=self.default, 
                                     scores_df=scores_df, is_classifier=is_classifier)
                
        rule_mask = pd.Series(self.__rule__(X)).values
        scores_df = self.if_true.score_rule(X[rule_mask], y[rule_mask], scores_df, is_classifier)
        scores_df = self.if_false.score_rule(X[~rule_mask], y[~rule_mask], scores_df, is_classifier)
        return scores_df
    
    def set_rule_id(self, rule_id:int=0)->int:
        rule_id = super().set_rule_id(rule_id)
        rule_id = self.if_true.set_rule_id(rule_id) 
        rule_id = self.if_false.set_rule_id(rule_id) 
        return rule_id
    
    def get_max_rule_id(self, max_rule_id:int=0)->int:
        max_rule_id = super().get_max_rule_id(max_rule_id)
        
        max_rule_id = self.if_true.get_max_rule_id(max_rule_id)
        max_rule_id = self.if_false.get_max_rule_id(max_rule_id)
        return max_rule_id
    
    def get_rule(self, rule_id:int)->BusinessRule:
        if self._rule_id is not None and self._rule_id == rule_id:
            return self
        
        if_true_rule = self.if_true.get_rule(rule_id)
        if if_true_rule is not None:
            return if_true_rule

        if_false_rule = self.if_false.get_rule(rule_id)
        if if_false_rule is not None:
            return if_false_rule
        
    def get_rule_input(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None
                      )->Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]:
        if y is not None:
            input_X, input_y = super().get_rule_input(rule_id, X, y)
            if input_X is not None:
                return input_X, input_y
        else:
            input_X = super().get_rule_input(rule_id, X)
            if input_X is not None:
                return input_X
            
        rule_mask = pd.Series(self.__rule__(X)).values
        

        if y is not None:
            input_X, input_y = self.if_true.get_rule_input(rule_id, X[rule_mask], y[rule_mask])
            if input_X is not None:
                return input_X, input_y
        else:
            input_X = self.if_true.get_rule_input(rule_id, X[rule_mask])
            if input_X is not None:
                return input_X
                  
        if y is not None:
            input_X, input_y = self.if_false.get_rule_input(rule_id, X[~rule_mask], y[~rule_mask])
            if input_X is not None:
                return input_X, input_y
        else:
            input_X = self.if_false.get_rule_input(rule_id, X[~rule_mask])
            if input_X is not None:
                return input_X
               
        if y is not None:
            return None, None
        else:
            return None
        
    def get_rule_leftover(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None
                         )->Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]:
        if self._rule_id is not None and self._rule_id == rule_id:
            y_preds = self.predict(X)
            mask = np.isnan(y_preds)
            if y is not None:
                return X[mask], y[mask]
            else:
                return X[mask]
            
        rule_mask = pd.Series(self.__rule__(X)).values
        
        if y is not None:
            leftover_X, leftover_y = self.if_true.get_rule_leftover(rule_id, X[rule_mask], y[rule_mask])
            if leftover_X is not None:
                return leftover_X, leftover_y
        else:
            leftover_X = self.if_true.get_rule_leftover(rule_id, X[rule_mask])
            if leftover_X is not None:
                return leftover_X
          
        if y is not None:
            leftover_X, leftover_y = self.if_false.get_rule_leftover(rule_id, X[~rule_mask], y[~rule_mask])
            if leftover_X is not None:
                return leftover_X, leftover_y
        else:
            leftover_X = self.if_false.get_rule_leftover(rule_id, X[~rule_mask])
            if leftover_X is not None:
                return leftover_X
               
        if y is not None:
            return None, None
        else:
            return None
        
    def replace_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        replace_rule = super().replace_rule(rule_id, new_rule)
        if replace_rule is None and hasattr(self, "if_true"):
            replace_rule = self.if_true.replace_rule(rule_id, new_rule)
        if replace_rule is None and hasattr(self, "if_false"):
            replace_rule = self.if_false.replace_rule(rule_id, new_rule)
        return replace_rule

    def remove_rule(self, rule_id:int):
        if self.if_true._rule_id is not None and self.if_true._rule_id == rule_id:
            self.replace_rule(self.if_true._rule_id, EmptyRule())
            self._stored_params['if_true'] = self.if_true
            return self.if_true
        elif self.if_false._rule_id is not None and self.if_false._rule_id == rule_id:
            self.replace_rule(self.if_false._rule_id, EmptyRule())
            self._stored_params['if_false'] = self.if_false
            return self.if_false
        else:
            removed = self.if_true.remove_rule(rule_id)
            if not removed:
                removed = self.if_false.remove_rule(rule_id)
            return removed
       
    def get_rule_params(self, rule_id:int)->dict:
        params = super().get_rule_params(rule_id)
        if params is not None: 
            return params

        params = self.if_true.get_rule_params(rule_id)
        if params is not None: 
            return params

        params = self.if_false.get_rule_params(rule_id)
        if params is not None: 
            return params
            
    def set_rule_params(self, rule_id:int, **params)->None:
        super().set_rule_params(rule_id, **params)
        
        self.if_true.set_rule_params(rule_id, **params)
        self.if_false.set_rule_params(rule_id, **params)

    def _get_casewhens(self, casewhens:dict=None):
        casewhens = super()._get_casewhens(casewhens)
        casewhens = self.if_true._get_casewhens(casewhens)
        casewhens = self.if_false._get_casewhens(casewhens)
        return casewhens

    def _get_binarynodes(self, binarynodes:dict=None):
        binarynodes = super()._get_binarynodes(binarynodes)
        if self._rule_id is not None:
            binarynodes[self._rule_id] = dict(if_true=self.if_true._rule_id, if_false=self.if_false._rule_id)
        binarynodes = self.if_true._get_binarynodes(binarynodes)
        binarynodes = self.if_false._get_binarynodes(binarynodes)
        return binarynodes
        
    def add_to_igraph(self, graph:Graph=None)->Graph:
        graph = super().add_to_igraph(graph)
        self.if_true.add_to_igraph(graph)
        self.if_false.add_to_igraph(graph)
        
        if self._rule_id is not None and self.if_true._rule_id is not None:
            graph.add_edge(self._rule_id, self.if_true._rule_id, binary_node="if_true")
        if self._rule_id is not None and self.if_false._rule_id is not None:
            graph.add_edge(self._rule_id, self.if_false._rule_id, binary_node="if_false")
        return graph
              
    def __rulerepr__(self):
        return "BinarySplit"


class IsInSplit(BinarySplit):
    def __init__(self, col:str, cats:List[str],
                 if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        super().__init__()
        if not isinstance(self.cats, list):
            self.cats = [self.cats]

    def __rule__(self, X:pd.DataFrame)->pd.Series:
        return X[self.col].isin(self.cats)
    
    def __rulerepr__(self)->str:
        return f"Split if {self.col} in {self.cats}"

class GreaterThanSplit(BinarySplit):
    def __init__(self, col:str, cutoff:float,
                 if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame)->pd.Series:
        return X[self.col] > self.cutoff

    def __rulerepr__(self)->str:
        return f"Split if {self.col} > {self.cutoff}"


class GreaterEqualThanSplit(BinarySplit):
    def __init__(self, col:str, cutoff:float,
                 if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame)->pd.Series:
        return X[self.col] >= self.cutoff

    def __rulerepr__(self)->str:
        return f"Split if {self.col} >= {self.cutoff}"

class LesserThanSplit(BinarySplit):
    def __init__(self, col:str, cutoff:float,
                 if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame)->pd.Series:
        return X[self.col] < self.cutoff

    def __rulerepr__(self)->str:
        return f"Split if {self.col} < {self.cutoff}"

class LesserEqualThanSplit(BinarySplit):
    def __init__(self, col:str, cutoff:float, 
                 if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame)->pd.Series:
        return X[self.col] <= self.cutoff

    def __rulerepr__(self)->str:
        return f"Split if {self.col} <= {self.cutoff}"

class RangeSplit(BinarySplit):
    def __init__(self, col:str, min:float, max:float, 
                if_true:BusinessRule=None, if_false:BusinessRule=None, 
                default=None):
        super().__init__()

    def __rule__(self, X:pd.DataFrame):
        return (X[self.col] >= self.min) & (X[self.col] <= self.max)

    def __rulerepr__(self):
        return f"Split if {self.min} <= {self.col} <= {self.max} "

class MultiRangeSplit(BinarySplit):
    def __init__(self, range_dict, if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        """
        Switches to if_true rule if all range conditions hold for all cols in
        range_dict. range_dict can contain multiple ranges per col.

        range_dict should be of the format 
        ```
        range_dict = {
            'petal length (cm)': [[4.1, 4.7], [5.2, 7.5]],
            'petal width (cm)': [1.6, 2.6]
        }
        ```

        """
        super().__init__()
    
    def __rule__(self, X):
        return generate_range_mask(self.range_dict, X, kind='all')
        
    def __rulerepr__(self):
        return ("Split if " + " AND ".join([f"{k} in {v}" for k, v in self.range_dict.items()]))


class MultiRangeAnySplit(BinarySplit):
    def __init__(self, range_dict, if_true:BusinessRule=None, if_false:BusinessRule=None, default=None):
        """
        Switches to if_true rule if any range conditions hold for any cols in
        range_dict. range_dict can contain multiple ranges per col.

        range_dict should be of the format 
        ```
        range_dict = {
            'petal length (cm)': [[4.1, 4.7], [5.2, 7.5]],
            'petal width (cm)': [1.6, 2.6]
        }
        ```

        """
        super().__init__()
    
    def __rule__(self, X):
        return generate_range_mask(self.range_dict, X, kind='any')
        
    def __rulerepr__(self):
        return ("Split if " + " OR ".join([f"{k} in {v}" for k, v in self.range_dict.items()]))
