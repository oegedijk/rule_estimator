__all__ = [
    'RuleEstimator',
    'RuleClassifier',
    'RuleRegressor'
]
from typing import Union, List, Dict, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error

from igraph import Graph 

from .businessrule import BusinessRule 
from .nodes import BinaryDecisionNode
from .rules import CaseWhen, EmptyRule


class RuleEstimator(BusinessRule):
    def __init__(self, rules:Union[BusinessRule, List[BusinessRule]]=None,
                 final_estimator:BaseEstimator=None, fit_remaining_only:bool=True):
        super().__init__()
        if rules is None:
            print("WARNING: No rules passed! Setting a EmptyRule that predicts nan!")
            self.rules = EmptyRule()
        
        self._reset_rule_ids()
        self.fitted = False
        
    def _reset_rule_ids(self)->None:
        self.rules.set_rule_id(0)
        
    def fit(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]):
        if self.final_estimator is not None:
            print("Fitting final_estimator...")
            if self.fit_remaining_only:
                y_pred = self.rules.predict(X)
                self.final_estimator.fit(X[np.isnan(y_pred)], y[np.isnan(y_pred)])
            else:
                self.final_estimator.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X:pd.DataFrame)->np.ndarray:
        assert self.final_estimator is None or self.fitted, \
            "If you have a final_estimator you need to .fit(X, y) first!"

        y = self.rules.predict(X)
         
        if self.final_estimator is not None:
            y[np.isnan(y)] = self.final_estimator.predict(X[np.isnan(y)])
        return y
    
    def get_max_rule_id(self)->int:
        return self.rules.get_max_rule_id(max_rule_id=0)
    
    def append_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        rule = self.get_rule(rule_id)
        if isinstance(rule, CaseWhen):
            rule.append_rule(new_rule)
        else:
            rule.replace_rule(rule._rule_id, CaseWhen([deepcopy(rule), new_rule]))
        self.reset_rule_ids()
        
    def replace_binarynode_true_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        binary_node = self.get_rule(rule_id)
        if isinstance(binary_node, BinaryDecisionNode):
            binary_node.if_true = new_rule
            self.reset_rule_ids()
        else:
            raise ValueError(f"rule {rule_id} is not a BinaryDecisionNode!")
        
    def replace_binarynode_false_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        binary_node = self.get_rule(rule_id)
        if isinstance(binary_node, BinaryDecisionNode):
            binary_node.if_false = new_rule
            self.reset_rule_ids()
        else:
            raise ValueError(f"rule {rule_id} is not a BinaryDecisionNode!")
        
    def score_rule(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray])->pd.DataFrame:
        return self.score_rules(X, y)
    
    def score_rules(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray])->pd.DataFrame:
        return self.rules.score_rule(X, y)
    
    def get_rule(self, rule_id:int)->BusinessRule:
        return self.rules.get_rule(rule_id)
        
    def get_rule_input(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None
            )->Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]:
        return self.rules.get_rule_input(rule_id, X, y)
        
    def get_rule_leftover(self, rule_id, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None
            )->Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]:
        return self.rules.get_rule_leftover(rule_id, X, y)
    
    def replace_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        self.rules.replace_rule(rule_id, new_rule)
        self._reset_rule_ids()
     
    def get_rule_params(self, rule_id:int)->dict:
        return self.rules.get_rule_params(rule_id)
        
    def set_rule_params(self, rule_id:int, **params)->None:
        self.rules.set_rule_params(rule_id, **params)
        
    def get_igraph(self)->Graph:
        self._reset_rule_ids()
        return self.rules.add_to_igraph()
    
    def plot(self):
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Failed to load plotly, the plotting backend. "
                              "You need to install it seperately to with pip install plotly")

        graph = self.get_igraph()
        layout = graph.layout_reingold_tilford(mode="in", root=0)
        nodes_x = [6*pos[0] for pos in layout]
        nodes_y = [pos[1] for pos in layout]
        #find the max Y in order to invert the graph:
        nodes_y = [2*max(nodes_y)-y for y in nodes_y]

        connections_x, connections_y = [], []
        for edge in graph.es:
            connections_x += [nodes_x[edge.tuple[0]], nodes_x[edge.tuple[1]], None]
            connections_y += [nodes_y[edge.tuple[0]], nodes_y[edge.tuple[1]], None]


        fig = go.Figure()

        fig.add_trace(go.Scatter(
                           x=connections_x,
                           y=connections_y,
                           mode='lines',
                           name='connections',
                           line=dict(color='rgb(210,210,210)', width=1),
                           hoverinfo='none'
                           ))

        fig.add_trace(go.Scatter(
                          x=nodes_x,
                          y=nodes_y,
                          mode='markers+text',
                          name='nodes',
                          marker=dict(symbol='circle',
                                        size=18,
                                        color='#6175c1',    
                                        line=dict(color='rgb(50,50,50)', width=1)
                                        ),
                          text=[f"{id}: {desc}" for id, desc in zip(graph.vs['rule_id'], graph.vs['name'])],
                          textposition="top right",
                          hovertext=graph.vs['description'],
                          opacity=0.8
                          ))

        fig.update_layout(showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig
          
    def __rulerepr__(self)->str:
        return "RuleEstimator"

    def describe(self)->str:
        description = super().describe()
        if self.final_estimator is not None:
            description += f"final_estimator = {self.final_estimator}\n"
        return description


class RuleClassifier(RuleEstimator):
    
    def score_rules(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray])->pd.DataFrame:
        return self.rules.score_rule(X, y, is_classifier=True)
    
    def suggest_rule(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray], 
                     kind='rule', after:bool=False)->BusinessRule:
        if after:
            X, y = self.get_rule_leftover(rule_id, X, y)
            if len(X) == 0:
                raise ValueError(f"No samples after application of rule {rule_id}! Try setting after=False.")
        else:
            X, y = self.get_rule_input(rule_id, X, y)
            
        dt = DecisionTreeClassifier(max_depth=1, max_features=1.0, max_leaf_nodes=2).fit(X, y)
        y_most_frequent = pd.Series(y).value_counts().index[0]
        col = X.columns.tolist()[dt.tree_.feature[0]]
        cutoff = dt.tree_.threshold[0]
        prediction = dt.classes_[dt.tree_.value[1].argmax()]
        default = dt.classes_[dt.tree_.value[2].argmax()]

        if kind=='rule':
            return f"LesserThan(col='{col}', cutoff={cutoff}, prediction={prediction}, default={default})"
        elif kind=='prediction':
            return f"PredictionRule(prediction={y_most_frequent})"
        elif kind=='node':
            return f"LesserThanNode(col='{col}', cutoff={cutoff})"

    def __rulerepr__(self)->str:
        return "RuleClassifier"

class RuleRegressor(RuleEstimator):
    
    def suggest_rule(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray], 
                     kind='rule', after:bool=False)->BusinessRule:
        if after:
            X, y = self.get_rule_leftover(rule_id, X, y)
            if len(X) == 0:
                raise ValueError(f"No samples after application of rule {rule_id}! Try setting after=False.")
        else:
            X, y = self.get_rule_input(rule_id, X, y)
            
        dt = DecisionTreeRegressor(max_depth=1, max_features=1.0, max_leaf_nodes=2).fit(X, y)
        y_mean = pd.Series(y).mean()
        col = X.columns.tolist()[dt.tree_.feature[0]]
        cutoff = dt.tree_.threshold[0]
        prediction = dt.tree_.value[1].mean()
        default = dt.tree_.value[2].mean()
        
        if kind=='rule':
            return f"LesserThan(col='{col}', cutoff={cutoff}, prediction={prediction}, default={default})"
        elif kind=='prediction':
            return f"PredictionRule(prediction={y_mean})"
        elif kind=='node':
            return f"LesserThanNode(col='{col}', cutoff={cutoff})"
    
    def __rulerepr__(self)->str:
        return "RuleRegressor"