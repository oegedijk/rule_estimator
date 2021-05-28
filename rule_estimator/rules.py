__all__ = [
    'CaseWhen',
    'EmptyRule',
    'PredictionRule',
    'GreaterThan', 
    'GreaterEqualThan', 
    'LesserThan', 
    'LesserEqualThan',
    'MultiRangeAndRule',
    'MultiRangeOrRule'
]

from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from igraph import Graph

from .businessrule import BusinessRule


class CaseWhen(BusinessRule):
    def __init__(self, rules:List[BusinessRule], default=None):
        super().__init__()
        if not isinstance(self.rules, list):
            self.rules = [self.rules]

    def append_rule(self, new_rule:BusinessRule, rule_id:int=None)->None:
        if rule_id is not None:
            rule_ids = [rule._rule_id for rule in self.rules]
            if rule_id in rule_ids:
                self.rules.insert(rule_ids.index(rule_id)+1, new_rule)
            else:
                raise ValueError(f"rule_id {rule_id} can not be found in this CaseWhen!")
        else:
            self.rules.append(new_rule)
        self._stored_params['rules'] = self.rules
        
    def predict(self, X:pd.DataFrame)->np.ndarray:
        y = np.full(len(X), np.nan)
        for rule in self.rules:
            y[np.isnan(y)] = rule.predict(X[np.isnan(y)])
            
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
  
        y_temp = np.full(len(X), np.nan)
        for rule in self.rules:
            scores_df = rule.score_rule(X[np.isnan(y_temp)], y[np.isnan(y_temp)], scores_df, is_classifier)
            y_temp[np.isnan(y_temp)] = rule.predict(X[np.isnan(y_temp)])
            
        return scores_df
    
    def set_rule_id(self, rule_id:int=0)->int:
        rule_id = super().set_rule_id(rule_id)
        for rule in self.rules:
            rule_id = rule.set_rule_id(rule_id) 
        return rule_id
    
    def get_max_rule_id(self, max_rule_id:int=0)->int:
        max_rule_id = super().get_max_rule_id(max_rule_id)
        
        for rule in self.rules:
            max_rule_id = rule.get_max_rule_id(max_rule_id)
        return max_rule_id
    
    def get_rule(self, rule_id:int)->BusinessRule:
        if self._rule_id is not None and self._rule_id == rule_id:
            return self
        
        for rule in self.rules:
            return_rule = rule.get_rule(rule_id)
            if return_rule is not None:
                return return_rule
    
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
            
            
        y_temp = np.full(len(X), np.nan)
        for rule in self.rules:
            mask = np.isnan(y_temp)
            if y is not None:
                input_X, input_y = rule.get_rule_input(rule_id, X[mask], y[mask])
                if input_X is not None:
                    return input_X, input_y
            else:
                input_X = rule.get_rule_input(rule_id, X[mask])
                if input_X is not None:
                    return input_X
            y_temp[mask] = rule.predict(X[mask])     
        
        if y is not None:
            return None, None
        else:
            return None
        
    def get_rule_leftover(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None
                         )->Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]:
        if self._rule_id is not None and self._rule_id == rule_id:
            # first predict without filling in the default to get the mask
            old_default = self.default
            self.default = np.nan
            y_preds = self.predict(X)
            self.default = old_default 
            mask = np.isnan(y_preds)
            if y is not None:
                return X[mask], y[mask]
            else:
                return X[mask]
            
        y_temp = np.full(len(X), np.nan)
        for rule in self.rules:
            mask = np.isnan(y_temp)
            if y is not None:
                leftover_X, leftover_y = rule.get_rule_leftover(rule_id, X[mask], y[mask])
                if leftover_X is not None:
                    return leftover_X, leftover_y
            else:
                leftover_X = rule.get_rule_leftover(rule_id, X[mask])
                if leftover_X is not None:
                    return leftover_X
            y_temp[mask] = rule.predict(X[mask])   
            
        if y is not None:
            return None, None
        else:
            return None
        
    def replace_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        super().replace_rule(rule_id, new_rule)
        
        if hasattr(self, "rules"):
            for rule in self.rules:
                rule.replace_rule(rule_id, new_rule)
            
    def get_rule_params(self, rule_id:int)->dict:
        if self._rule_id is not None and self._rule_id == rule_id:
            return self.get_params()

        for rule in self.rules:
            params = rule.get_rule_params(rule_id)
            if params is not None: 
                return params
            
    def set_rule_params(self, rule_id:int, **params)->None:
        super().set_rule_params(rule_id, **params)
        
        for rule in self.rules:
            rule.set_rule_params(rule_id, **params)

    def _get_casewhens(self, casewhens:dict=None):
        if self._rule_id is not None:
            if casewhens is None:
                casewhens =  {self._rule_id:[rule._rule_id for rule in self.rules]}
            else:
                casewhens[self._rule_id] = [rule._rule_id for rule in self.rules]
        for rule in self.rules:
            casewhens = rule._get_casewhens(casewhens)
        return casewhens
            
    def add_to_igraph(self, graph:Graph=None)->Graph:
        graph = super().add_to_igraph(graph)

        for rule in self.rules:
            graph = rule.add_to_igraph(graph)
            if self._rule_id is not None and rule._rule_id is not None:
                graph.add_edge(self._rule_id, rule._rule_id)
        return graph
          
    def __rulerepr__(self)->str:
        return "CaseWhen"


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


class MultiRangeAndRule(BusinessRule):
    def __init__(self, range_dict, prediction, default=None):
        """
        Predicts prediction if all range conditions hold for all cols in
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
        
        def AND_masks(masks):
            for i, mask in enumerate(masks):
                combined_mask = mask if i == 0 else combined_mask & mask # noqa: F82
            return combined_mask
        
        def OR_masks(masks):
            for i, mask in enumerate(masks):
                combined_mask = mask if i == 0 else combined_mask | mask # noqa: F82
            return combined_mask
        
        def generate_range_mask(X, col, col_range):
            if isinstance(col_range[0], list):
                return OR_masks([(X[col] > l) & (X[col] < h) for l, h in col_range])
            return (X[col] > col_range[0]) & (X[col] < col_range[1])
        
        return AND_masks([generate_range_mask(X, col, col_range) for col, col_range in self.range_dict.items()])
        
    def __rulerepr__(self):
        return ("MultiRangeAndRule: If " 
                    + " AND ".join([f"{k} in {v}" for k, v in self.range_dict.items()])
                    + f" then predict {self.prediction}")


class MultiRangeOrRule(BusinessRule):
    def __init__(self, range_dict, prediction, default=None):
        """
        Predicts prediction if any range conditions hold for any cols in
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
        
        def AND_masks(masks):
            for i, mask in enumerate(masks):
                combined_mask = mask if i == 0 else combined_mask & mask # noqa: F82
            return combined_mask
        
        def OR_masks(masks):
            for i, mask in enumerate(masks):
                combined_mask = mask if i == 0 else combined_mask | mask # noqa: F82
            return combined_mask
        
        def generate_range_mask(X, col, col_range):
            if isinstance(col_range[0], list):
                return OR_masks([(X[col] > l) & (X[col] < h) for l, h in col_range])
            return (X[col] > col_range[0]) & (X[col] < col_range[1])
        
        return OR_masks([generate_range_mask(X, col, col_range) for col, col_range in self.range_dict.items()])
        
    def __rulerepr__(self):
        return ("MultiRangeOrRule: If " 
                    + " OR ".join([f"{k} in {v}" for k, v in self.range_dict.items()])
                    + f" then predict {self.prediction}")
