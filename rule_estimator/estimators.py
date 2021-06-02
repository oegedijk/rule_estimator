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
from sklearn.metrics import accuracy_score, mean_squared_error

from igraph import Graph 

from .businessrule import BusinessRule 
from .nodes import BinaryNode
from .rules import CaseWhen, EmptyRule


class RuleEstimator(BusinessRule):
    def __init__(self, rules:Union[BusinessRule, List[BusinessRule]]=None,
                 final_estimator:BaseEstimator=None, fit_remaining_only:bool=True):
        super().__init__()
        if rules is None:
            self.rules = CaseWhen()
            self._stored_params['rules'] = self.rules
        
        self._reset_rule_ids()
        self.fitted = False
        
    def fit(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]):
        """If a final_estimator has been specified, then fit this estimator.

        Args:
            X (pd.DataFrame): Input data
            y (Union[pd.Series, np.ndarray]): labels

        Returns:
            RuleEstimator: the fitted object
        """
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
        """Predict labels based on input

        Args:
            X (pd.DataFrame): input data

        Returns:
            np.ndarray: predictions
        """
        assert self.final_estimator is None or self.fitted, \
            "If you have a final_estimator you need to .fit(X, y) first!"

        y = self.rules.predict(X)
         
        if self.final_estimator is not None:
            y[np.isnan(y)] = self.final_estimator.predict(X[np.isnan(y)])
        return y

    def __rulerepr__(self)->str:
        return "RuleEstimator"

    def _reset_rule_ids(self)->None:
        """recursively go through all the rules and assign them increasing
        self._rule_id identifiers"""
        self.rules.set_rule_id(0)

    def _get_max_rule_id(self)->int:
        """Returns the highest _rule_id of any rules currently defined. """
        self._reset_rule_ids()
        return self.rules.get_max_rule_id(max_rule_id=0)

    def _get_casewhens(self):
        self._reset_rule_ids()
        return self.rules._get_casewhens()
    
    def append_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        """Appends a rule to the rule with rule_id. If the rule is a CaseWhen
        rule then simply appends new_rule to that CaseWhen rule. Otherwise wraps
        the original rule and the new_rule in a new CaseWhen rule.

        Args:
            rule_id (int): rule_id of the rule that you want to appendn a rule to
            new_rule (BusinessRule): The new rul that you would like to append
        """
        def get_casewhen_id(casewhens, rule_id):
            for k, v in casewhens.items():
                if rule_id in v:
                    return k
        self._reset_rule_ids()

        rule = self.get_rule(rule_id)
        casewhen_id  = get_casewhen_id(self._get_casewhens(), rule_id)

        if isinstance(rule, CaseWhen):
            rule.append_rule(new_rule)
        elif isinstance(rule, EmptyRule):
            new_rule = rule.replace_rule(rule._rule_id, new_rule)
        elif casewhen_id is not None:
            casewhen_rule = self.get_rule(casewhen_id)
            casewhen_rule.append_rule(new_rule, rule_id)
        else:
            self.replace_rule(rule_id, CaseWhen(rules=[deepcopy(rule), new_rule]))
        self._reset_rule_ids()
        if isinstance(new_rule, BinaryNode):
            return new_rule.if_true._rule_id
        return new_rule._rule_id

    def replace_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        """Replace rule with rule_id with new_rule

        Args:
            rule_id (int): rule_id
            new_rule (BusinessRule): rule to replace the original rule
        """
        new_rule = self.rules.replace_rule(rule_id, new_rule)
        self._stored_params['rules'] = self.rules
        self._reset_rule_ids()
        if isinstance(new_rule, BinaryNode):
            return new_rule.if_true
        return new_rule
        
    def replace_binarynode_true_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        """Replace the if_true rule of a BinaryNode with new_rule

        Args:
            rule_id (int): rule_id of the BinaryNode
            new_rule (BusinessRule): New rule to replace the old rule

        Raises:
            ValueError: In case the rule with rule_id is not a BinaryNode
        """
        binary_node = self.get_rule(rule_id)
        if isinstance(binary_node, BinaryNode):
            binary_node.if_true = new_rule
            self.reset_rule_ids()
        else:
            raise ValueError(f"rule {rule_id} is not a BinaryNode!")
        
    def replace_binarynode_false_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        """Replace the if_false rule of a BinaryNode with new_rule

        Args:
            rule_id (int): rule_id of the BinaryNode
            new_rule (BusinessRule): New rule to replace the old rule

        Raises:
            ValueError: In case the rule with rule_id is not a BinaryNode
        """
        binary_node = self.get_rule(rule_id)
        if isinstance(binary_node, BinaryNode):
            binary_node.if_false = new_rule
            self.reset_rule_ids()
        else:
            raise ValueError(f"rule {rule_id} is not a BinaryNode!")
        
    def score_rule(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray])->pd.DataFrame:
        """dummy method for self.scores_rules(X, y). Use scores_rules(X, y) instead."""
        return self.score_rules(X, y)
    
    def score_rules(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray])->pd.DataFrame:
        """Go through each rule and keep track of the ingoing and outgoing data
        and the performance of that rule (accuracy for classifier, rmse for regressor).
        Returns a dataframe with the information for each rule.

        Args:
            X (pd.DataFrame): Input data
            y (Union[pd.Series, np.ndarray]): input labels

        Returns:
            pd.DataFrame: [description]
        """
        return self.rules.score_rule(X, y)
    
    def get_rule(self, rule_id:int)->BusinessRule:
        """Return the rule with a specific rule_id. Can get an overview of the
        rule_id's with estimator.describe()

        Args:
            rule_id (int): rule id

        Returns:
            BusinessRule: rule with rule_id equal to rule_id
        """
        return self.rules.get_rule(rule_id)
        
    def get_rule_input(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None,
            after:bool=False)->Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]:
        """Return the input data for rule with rule_id. Each rule only receives the input
        data for which no predictions have been generated by upstream rules. In order
        to evaluate the effectiveness of a rule and perhaps find a better one it is useful
        to find the input data to that specific rule. You can find the rule_id's by
        calling estimator.describe().

        If you only pass parameter X, you will only get X back. If you pass poth
        X and y, you will get both X and y back. 

        Args:
            rule_id (int): rule_id
            X (pd.DataFrame): Input data
            y (Union[pd.Series, np.ndarray], optional): Input labels. Defaults to None.
            after (bool, optional): return get_rule_leftover(X, y) instead. Defaults to False.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]: [description]
        """
        if after:
            return self.get_rule_leftover(rule_id, X, y)
        return self.rules.get_rule_input(rule_id, X, y)
        
    def get_rule_leftover(self, rule_id, X:pd.DataFrame, y:Union[pd.Series, np.ndarray]=None
            )->Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]:
        """Return the leftover data for rule with rule_id. Each rule only receives the input
        data for which no predictions have been generated by upstream rules, and then
        either outputs predictions or np.nans when the rule does not apply. In order
        to come up with a new rule that should come after this rule it is useful
        to find the leftover data of that specific rule. You can find the rule_id's by
        calling estimator.describe().

        If you only pass parameter X, you will only get X back. If you pass poth
        X and y, you will get both X and y back. 

        Args:
            rule_id (int): rule_id
            X (pd.DataFrame): Input data
            y (Union[pd.Series, np.ndarray], optional): Input labels. Defaults to None.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]]: [description]
        """
        return self.rules.get_rule_leftover(rule_id, X, y)
     
    
    def get_rule_params(self, rule_id:int)->dict:
        """Get the parameters of the rule with rule_id. You can get an overview
        of rule_id's with estimator.describe().

        Args:
            rule_id (int): rule_id

        Returns:
            dict: dictionary with all input parameters for the rule
        """
        return self.rules.get_rule_params(rule_id)
        
    def set_rule_params(self, rule_id:int, **params)->None:
        """Update the parameter for a specific rule with rule_id. Find the
        rule_id's with estimator.describe()

        Args:
            rule_id (int): rule_id
            **params: kwargs will be passed to the rule and updated.
        """
        self.rules.set_rule_params(rule_id, **params)
        
    def get_igraph(self)->Graph:
        """Return an igraph object with all the rules as vertices and the 
        connections between rules as edges. Can be used to plot the system of rules
        with estimator.plot()

        Returns:
            Graph: an igraph.Graph instances. vertex attributes are 'rule_id', 
                'description' and 'rule'.
        """
        self._reset_rule_ids()
        return self.rules.add_to_igraph()

    def describe(self)->str:
        """returns an indented high-level description of the system of rules
        used by the estimator. Including rule_id's. 

        Returns:
            str: [description]
        """
        self._reset_rule_ids()
        description = super().describe()
        if self.final_estimator is not None:
            description += f"final_estimator = {self.final_estimator}\n"
        return description
    
    def plot(self):
        """
        Returns a plotly Figure of the rules. Uses the Reingolf-Tilford algorithm
        to generate the tree layout. 

        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Failed to load plotly, the plotting backend. "
                              "You need to install it seperately with pip install plotly.")

        graph = self.get_igraph()
        layout = graph.layout_reingold_tilford(mode="in", root=0)
        nodes_x = [6*pos[0] for pos in layout]
        nodes_y = [pos[1] for pos in layout]
        #find the max Y in order to invert the graph:
        nodes_y = [2*max(nodes_y)-y for y in nodes_y]

        casewhen_x, casewhen_y = [], []
        for edge in graph.es.select(casewhen=True):
            casewhen_x += [nodes_x[edge.tuple[0]], nodes_x[edge.tuple[1]], None]
            casewhen_y += [nodes_y[edge.tuple[0]], nodes_y[edge.tuple[1]], None]

        node_true_x, node_true_y = [], []
        for edge in graph.es.select(binary_node="if_true"):
            node_true_x += [nodes_x[edge.tuple[0]], nodes_x[edge.tuple[1]], None]
            node_true_y += [nodes_y[edge.tuple[0]], nodes_y[edge.tuple[1]], None]

        node_false_x, node_false_y = [], []
        for edge in graph.es.select(binary_node="if_false"):
            node_false_x += [nodes_x[edge.tuple[0]], nodes_x[edge.tuple[1]], None]
            node_false_y += [nodes_y[edge.tuple[0]], nodes_y[edge.tuple[1]], None]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
                        x=casewhen_x,
                        y=casewhen_y,
                        mode='lines',
                        name='connections',
                        line=dict(color='rgb(210,210,210)', width=2, dash='dot'),
                        hoverinfo='none'
                        ))

        fig.add_trace(go.Scatter(
                        x=node_true_x,
                        y=node_true_y,
                        mode='lines',
                        name='connections',
                        text=[None, "if true", None] * int(len(node_true_x)/3),
                        line=dict(color='rgb(210,210,210)', width=2, dash='dash'),
                        hoverinfo='none'
                        ))

        for (start_x, end_x, none), (start_y, end_y, none) in zip(zip(*[iter(node_true_x)]*3), zip(*[iter(node_true_y)]*3)):
            fig.add_annotation(x=(end_x+start_x)/2, y=(end_y+start_y)/2, text="true", showarrow=False)


        fig.add_trace(go.Scatter(
                        x=node_false_x,
                        y=node_false_y,
                        mode='lines',
                        name='connections',
                        text=[None, "if false", None] * int(len(node_true_x)/3),
                        line=dict(color='rgb(210,210,210)', width=2, dash='dash'),
                        hoverinfo='none'
                        ))

        for (start_x, end_x, none), (start_y, end_y, none) in zip(zip(*[iter(node_false_x)]*3), zip(*[iter(node_false_y)]*3)):
            fig.add_annotation(x=(end_x+start_x)/2, y=(end_y+start_y)/2, text="false", showarrow=False)

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

        fig.update_layout(showlegend=False, margin=dict(b=0, t=0, l=0, r=0))
        fig.update_xaxes(visible=False, range=(min(nodes_x)-1, max(nodes_x)+1))
        fig.update_yaxes(visible=False)
        return fig

    def pie(self, X:pd.DataFrame, y:np.ndarray, rule_id:int=None, after=False,
                size=120, margin=0, showlegend=False):
        try:
            import plotly.express as px
        except ImportError:
            raise ImportError("Failed to load plotly, the plotting backend. "
                              "You need to install it seperately with pip install plotly.")

        color_discrete_map = {str(i): color for i, color in enumerate(px.colors.qualitative.Plotly)}

        if rule_id is not None:
            X, y = self.get_rule_input(rule_id, X, y, after)
        
        return (px.pie(names=y.value_counts().index.astype(str), 
                                   values=y.value_counts().values, 
                                   color=y.value_counts().index.astype(str),
                                   color_discrete_map = color_discrete_map)
                               .update_layout(showlegend=showlegend, width=size, height=size)
                               .update_layout(margin=dict(t=margin, b=margin, l=margin, r=margin))
                               .update_traces(textinfo='none'))

    def parallel_coordinates(self, X:pd.DataFrame, y:np.ndarray, rule_id:int=None, 
                                cols:List[str]=None, after=False,
                                ymin=None, ymax=None):
        """generate parallel coordinates plot for data X, y. If rule_id is specified
        then only use data that reaches the rule with rule_id. You can select
        a sublist of columns by passing a list of cols.

        Args:
            X (pd.DataFrame): input
            y (np.ndarray): labels
            rule_id (int, optional): find the rule_id's with estimator.describe(). Defaults to None.
            cols (List[str], optional): List of columns to display. Defaults to None.

        Raises:
            ImportError: If you don't have plotly installed, raises import error.

        Returns:
            plotly.graph_objs.Figure
        """
        try:
            import plotly.graph_objs as go
            import plotly.express as px
        except ImportError:
            raise ImportError("Failed to load plotly, the plotting backend. "
                              "You need to install it seperately with pip install plotly.")
        if rule_id is not None:
            X, y = self.get_rule_input(rule_id, X, y, after)
        
        if cols is not None:
            X = X[cols]

        ymax = ymax if ymax is not None else y.max()
        ymin = ymin if ymin is not None else y.min()

        colors = px.colors.qualitative.Plotly
        colorscale = []
        for a, b in enumerate(np.linspace(0.0, 1.0, int(ymax)+2, endpoint=True)):
            if b<0.01:
                colorscale.append((b, colors[a])) 
            elif b > 0.01 and b < 0.99:
                colorscale.append((b, colors[a-1]))
                colorscale.append((b, colors[a]))
            else:
                colorscale.append((b, colors[a-1]))

        dimensions = [dict(label=col, values=X[col]) for col in X.columns]
        dimensions.append(dict(label = "y", values=y))

        fig = go.Figure(data=
                    go.Parcoords(
                        line = dict(color = y,
                                    cmin=ymin, 
                                    cmax=ymax,
                                    colorscale=colorscale,
                                    colorbar={'tickvals': y.tolist(),
                                            'ticktext': y.tolist()},
                                    showscale=True),
                        dimensions = dimensions
                    )
                )
        return fig
          
class RuleClassifier(RuleEstimator):
    
    def score_rules(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray])->pd.DataFrame:
        """Go through each rule and keep track of the ingoing and outgoing data
        and the accuracy score of that rule. Returns a dataframe with the information for each rule.

        Args:
            X (pd.DataFrame): Input data
            y (Union[pd.Series, np.ndarray]): input labels

        Returns:
            pd.DataFrame: [description]
        """
        return self.rules.score_rule(X, y, is_classifier=True)
    
    def suggest_rule(self, rule_id:int, X:pd.DataFrame, y:Union[pd.Series, np.ndarray], 
                     kind='rule', after:bool=False)->str:
        """Suggests a new rule in the place of rule with rule_id. Uses the 
        X and y that flow into that rule to suggest a replacement rule. 
        
        Can either use 
            kind='prediction': suggest a PredictionRule with
                the most prevalent class at that point.
            kind='rule': suggest a LesserThan rule based on a single step 
                DecisionTreeClassifier
            kind='node': suggest a LesserThanNode rule based on a single step 
                DecisionTreeClassifier

        If you pass after=True, use the leftover data out of the rule instead
        to make a suggestion.

        Args:
            rule_id (int): [description]
            X (pd.DataFrame): [description]
            y (Union[pd.Series, np.ndarray]): [description]
            kind (str, {'prediction', 'rule', 'node'}, optional): The type of 
                rule to return. Either a PredictionRule, a LesserThan rule or 
                a LesserThanNode. Defaults to 'rule'.
            after (bool, optional): Use the leftover data that is not assigned
                by the rule instead of the data flowing into the rule. Defaults to False.

        Raises:
            ValueError: If len(X) == 0

        Returns:
            str: string description of the sugggested rule
        """
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
                     kind='rule', after:bool=False)->str:
        """Suggests a new rule in the place of rule with rule_id. Uses the 
        X and y that flow into that rule to suggest a replacement rule. 
        
        Can either use 
            kind='prediction': suggest a PredictionRule with
                the most prevalent class at that point.
            kind='rule': suggest a LesserThan rule based on a single step 
                DecisionTreeRegressor
            kind='node': suggest a LesserThanNode rule based on a single step 
                DecisionTreeRegressor

        If you pass after=True, use the leftover data out of the rule instead
        to make a suggestion.

        Args:
            rule_id (int): [description]
            X (pd.DataFrame): [description]
            y (Union[pd.Series, np.ndarray]): [description]
            kind (str, {'prediction', 'rule', 'node'}, optional): The type of 
                rule to return. Either a PredictionRule, a LesserThan rule or 
                a LesserThanNode. Defaults to 'rule'.
            after (bool, optional): Use the leftover data that is not assigned
                by the rule instead of the data flowing into the rule. Defaults to False.

        Raises:
            ValueError: If len(X) == 0

        Returns:
            str: string description of the sugggested rule
        """
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