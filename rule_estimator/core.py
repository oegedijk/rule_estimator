# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['BusinessRule', 
           'BinaryDecisionNode',
           'RuleClassifier', 
           'RuleRegressor']

from typing import Union, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd



from sklearn.base import BaseEstimator

from .storable import Storable


def describe_businessrule(obj, spaces:int=0):
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
        rulerepr += " " * spaces + obj.__rulerepr__() + "\n"
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
    if isinstance(obj, BusinessRule) and hasattr(obj, "default") and not np.isnan(obj.default):
        rulerepr += f"{' '*(spaces+1)}Default: {obj.default} \n"
    return rulerepr


class BusinessRule(BaseEstimator, Storable):
    def __init__(self):
        self._store_child_params(level=2)
        if (not hasattr(self, "default") or
            (hasattr(self, "default") and (self.default is None or np.isnan(self.default)))):
            self.default = np.nan

    def fit(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None):
        pass

    def predict(self, X:pd.DataFrame):
        raise NotImplementedError

    def get_params(self, deep=True):
        """ """
        return self._stored_params

    def set_params(self, **params):
        """ """
        self._stored_params.update(params)

    def __rulerepr__(self):
        return "BusinessRule"

    def describe(self):
        return describe_businessrule(self)

    def to_yaml(self, filepath:Union[Path, str]=None, return_dict:bool=False):
        """Store object to a yaml format.

        Args:
            filepath: file where to store the .yaml file. If None then just return the
                yaml as a str.
            return_dict: instead of return a yaml str, return the raw dict.

        """
        return super().to_yaml(filepath, return_dict, comment=self.describe())


class BinaryDecisionNode(BusinessRule):
    def __init__(self):
        self._store_child_params(level=2)
        assert hasattr(self, "if_true")
        assert hasattr(self, "if_false")

    def __rulerepr__(self):
        return "BinaryDecisionNode"


class RuleEstimator(BusinessRule):
    def __init__(self, rules:Union[BusinessRule, List[BusinessRule]],
                 default:Union[int, float]=None, 
                 final_estimator:BaseEstimator=None, fit_remaining_only:bool=True):
        """[summary]

        Args:
            rules (Union[BusinessRule, List[BusinessRule]]): (nested list) of rules 
            that will be applied to the incoming DataFrame.
            default ({float, int} optional): [description]. If rules do not result
                in any prediction apply this default prediction. Defaults to None 
                (i.e. defaults to np.nan)
            final_estimator (BaseEstimator, optional): You can specify a scikit-learn
                estimator such as a DecisionTree or RandomForest to handle all
                the cases that were not covered by rules (i.e. got a np.nan prediction). 
                Defaults to None.
            fit_remaining_only (bool, optional): When fitting the final_estimator,
                only fit on the data rows that were not covered by a rule. 
                Defaults to True. When False then fits the final_estimator on the
                entire data tuple X, y.
        """
        super().__init__()
        if not isinstance(self.rules, list):
            self.rules = [self.rules]
        self.fitted = False

    def fit(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]):
        if self.final_estimator is not None:
            if self.fit_remaining_only:
                y_pred = self._predict_rules(X)
                X = X.copy()[np.isnan(y_pred)]
                y = y.copy()[np.isnan(y_pred)]
            print("Fitting final_estimator...")
            self.final_estimator.fit(X, y)
        self.fitted = True

    def _predict_rules(self, X:pd.DataFrame):
        """Applies the rules in self.rules to DataFrame X, and returns
        an array of predictions y. When no rule applies prediction equals np.nan.

        Args:
            X (pd.DataFrame): input DataFrame. 

        Returns:
            np.ndarray: array of predictions
        """
        y = np.full(len(X), np.nan)
        for rule in self.rules:
            y = np.where(np.isnan(y), rule.predict(X), y)

        y = np.where(np.isnan(y), self.default, y)
        return y

    def predict(self, X:pd.DataFrame):
        assert self.final_estimator is None or self.fitted, \
            "If you have a final_estimator you need to .fit(X, y) first!"

        y = self._predict_rules(X)
        if self.final_estimator is not None:
            X_remainder = X.copy()[np.isnan(y)]
            y[np.isnan(y)] = self.final_estimator.predict(X_remainder)
        return y

    def __rulerepr__(self):
        return "RulesEstimator"

    def describe(self):
        description = super().describe()
        if self.final_estimator is not None:
            description += f"final_estimator = {self.final_estimator}\n"
        return description


class RuleClassifier(RuleEstimator):

    def __rulerepr__(self):
        return "RulesClassifier"


class RuleRegressor(RuleEstimator):

    def __rulerepr__(self):
        return "RulesRegressor"