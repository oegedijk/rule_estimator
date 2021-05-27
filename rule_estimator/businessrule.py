# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['BusinessRule']

from typing import Union, List, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error

from igraph import Graph

from .storable import Storable


def describe_businessrule(obj, spaces:int=0)->str:
    """If obj has a __rulerepr__ method then adds it to a string,
    and then recursively finds all attributes with __rulerepr__ methods
    and adds them to the string with appropriate indentation.

    Args:
        obj (BusinessRule): BusinessRule instance to be recursively described.

        spaces (int, optional): Number of spaces of indentation. Gets recursively
            increased. Defaults to 0.

    Returns:
        str: description of the entire tree of businessrules inside obj.
    """
    rulerepr = ""
    if isinstance(obj, BusinessRule):
        rule_id = f"{obj._rule_id}: " if obj._rule_id is not None else ""
        rulerepr += " " * spaces + rule_id + obj.__rulerepr__() # + "\n"
        if hasattr(obj, "default") and not np.isnan(obj.default):
            rulerepr += f" (default={obj.default})\n"
        else:
            rulerepr += "\n"
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):
                rulerepr += describe_businessrule(v, spaces=spaces+2)
    elif isinstance(obj, dict):
        for v in obj.values():
            rulerepr += describe_businessrule(v, spaces=spaces+2)
    elif isinstance(obj, list):
        for v in obj:
            rulerepr += describe_businessrule(v, spaces=spaces+1)
    return rulerepr


class BusinessRule(BaseEstimator, Storable):
    def __init__(self, prediction=None, default=None):
        self._store_child_params(level=2)
        if not hasattr(self, "prediction"):
            self.prediction = prediction
        if not hasattr(self, "default"):
            self.default = default
            
        if self.prediction is None:
            self.prediction = np.nan
        if self.default is None:
            self.default = np.nan
            
        self._rule_id = None
        
    def fit(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None):
        pass

    def predict(self, X:pd.DataFrame)->np.ndarray:
        assert hasattr(self, "__rule__"), "You need to implement the __rule__ method first!"
        return np.where(self.__rule__(X), self.prediction, self.default)
    
    def _score_rule(self, y, y_preds, mask, prediction, default, 
                    scores_df=None, is_classifier=False)->pd.DataFrame:
        if scores_df is None:
            scores_df = pd.DataFrame(columns=[
                'rule_id', 'name','description', 'prediction', 
                'n_inputs', 'n_outputs','coverage', 
                'accuracy' if is_classifier else 'rmse'
            ])
        score_dict = dict(
                rule_id=self._rule_id, name=self.__class__.__name__,
                description=self.__rulerepr__(),
                prediction = prediction,
                n_inputs=len(y), n_outputs=mask.sum(), 
                coverage = mask.mean() if len(mask)>0 else np.nan,
        )
        
        if is_classifier:
            if len(y[mask]) > 0:
                score_dict['accuracy'] = accuracy_score(y[mask], y_preds[mask])
            else:
                score_dict['accuracy'] = np.nan
        else:
            if len(y[mask]) > 0:
                score_dict['rmse'] = mean_squared_error(y[mask], y_preds[mask], squared=False)
            else:
                score_dict['rmse'] = np.nan
        
        scores_df = scores_df.append(score_dict, ignore_index=True)
        
        if not np.isnan(default):
            default_score_dict = dict(
                    rule_id=self._rule_id, 
                    name=chr(int("21B3", 16)), #self.__class__.__name__,
                    description=f"default: predict {self.default}",
                    prediction=default,
                    n_inputs=len(y), n_outputs=np.invert(mask).sum(), 
                    coverage = np.invert(mask).mean() if len(mask)>0 else np.nan,       
            )
            if is_classifier:
                if np.invert(mask).sum() > 0:
                    default_score_dict['accuracy'] = accuracy_score(
                        y[~mask], np.full(np.invert(mask).sum(), default))
                else:
                    default_score_dict['accuracy'] = np.nan      
            else:
                if np.invert(mask).sum() > 0:
                    default_score_dict['rmse'] = mean_squared_error(
                        y[~mask], np.full(np.invert(mask).sum(), default), squared=False)
                else:
                    default_score_dict['rmse'] = np.nan
                    
            scores_df = scores_df.append(default_score_dict, ignore_index=True)
        return scores_df
    
    def score_rule(self, X:pd.DataFrame, y:Union[np.ndarray, pd.Series], 
                   scores_df:pd.DataFrame=None, is_classifier:bool=False)->pd.DataFrame:
        mask = pd.Series(self.__rule__(X)).values
        y_preds = self.predict(X)
        return self._score_rule(y, y_preds, mask, 
                                self.prediction, self.default, 
                                scores_df, is_classifier)
    
    def set_rule_id(self, rule_id:int=0)->int:
        self._rule_id = rule_id
        return rule_id+1
    
    def get_max_rule_id(self, max_rule_id:int=0)->int:
        if self._rule_id is not None and self._rule_id > max_rule_id:
            return self._rule_id
        return max_rule_id
    
    def get_rule(self, rule_id:int):
        if self._rule_id is not None and self._rule_id == rule_id:
            return self
        
    def replace_rule(self, rule_id:int, new_rule)->None:
        assert isinstance(new_rule, BusinessRule)
        if self._rule_id is not None and self._rule_id == rule_id:
            self.__class__ = new_rule.__class__
            self.__dict__.update(new_rule.__dict__)
            
    def get_rule_params(self, rule_id:int)->dict:
        if self._rule_id is not None and self._rule_id == rule_id:
            return self.get_params()
    
    def set_rule_params(self, rule_id:int, **params)->None:
        if self._rule_id is not None and self._rule_id == rule_id:
            self.set_params(**params)
          
    def get_rule_input(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None
                      )->Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]:
        if self._rule_id is not None and self._rule_id == rule_id:
            if y is not None:
                return X, y
            else:
                return X
            
        if y is not None:
            return None, None
        else:
            return None
        
    def get_rule_leftover(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None
                         )->Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]:
        if self._rule_id is not None and self._rule_id == rule_id:
            mask = np.invert(pd.Series(self.__rule__(X)).values)
            if y is not None:
                return X[mask], y[mask]
            else:
                return X[mask]
            
        if y is not None:
            return None, None
        else:
            return None
                    
    def get_params(self, deep:bool=True)->dict:
        """ """
        return self._stored_params

    def set_params(self, **params)->None:
        """ """
        for k, v in params.items():
            if k in self._stored_params:
                self._stored_params[k] = v
                setattr(self, k, v)
                
    def add_to_igraph(self, graph:Graph=None)->Graph:
        if graph is None:
            graph = Graph()
            graph.vs.set_attribute_values('rule_id', [])
            graph.vs.set_attribute_values('name', [])
            graph.vs.set_attribute_values('description', [])
            graph.vs.set_attribute_values('rule', [])
        graph.add_vertex(
            rule_id=self._rule_id,
            name=self.__class__.__name__,
            description=self.__rulerepr__(),
            rule=self
        )
        return graph
        
    def __rulerepr__(self)->str:
        return "BusinessRule"

    def describe(self)->str:
        return describe_businessrule(self)

    def to_yaml(self, filepath:Union[Path, str]=None, return_dict:bool=False):
        """Store object to a yaml format.

        Args:
            filepath: file where to store the .yaml file. If None then just return the
                yaml as a str.
            return_dict: instead of return a yaml str, return the raw dict.

        """
        return super().to_yaml(filepath, return_dict, comment=self.describe())


