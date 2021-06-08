__all__ = [
    'RuleEstimator',
    'RuleClassifier',
    'RuleRegressor'
]
from typing import Union, List, Dict, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from igraph import Graph 

from .businessrule import BusinessRule 
from .nodes import BinarySplit
from .rules import CaseWhen, EmptyRule


def describe_businessrule(obj, spaces:int=0, indent:int=0, prefix:str=None)->str:
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
    arrow = chr(int("21B3", 16))
    yes_arrow = arrow + 'y ' 
    no_arrow = arrow + 'n '
    if prefix is None: prefix = ""
    if isinstance(obj, BusinessRule):
        rule_id = f"{obj._rule_id}: " if obj._rule_id is not None else ""
        rulerepr += " " * (spaces) + prefix + rule_id + obj.__rulerepr__() # + "\n"
        if hasattr(obj, "default") and not np.isnan(obj.default):
            rulerepr += f" (default={obj.default})\n"
        else:
            rulerepr += "\n"

    if isinstance(obj, BinarySplit):
        rulerepr += ''*spaces + describe_businessrule(obj.if_true, spaces=spaces+indent, indent=3, prefix=yes_arrow)
        rulerepr += ''*spaces + describe_businessrule(obj.if_false, spaces=spaces+indent, indent=3, prefix=no_arrow)
    elif isinstance(obj, CaseWhen):
        for rule in obj.rules:
            rulerepr += describe_businessrule(rule, spaces=spaces+indent, indent=2, prefix=arrow+' ')   
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):
                rulerepr += describe_businessrule(v, spaces=spaces+2)
    if isinstance(obj, dict):
        for v in obj.values():
            rulerepr += describe_businessrule(v, spaces=spaces+2)
    elif isinstance(obj, list):
        for v in obj:
            rulerepr += describe_businessrule(v, spaces=spaces+1)
    return rulerepr

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

    def _get_binarynodes(self):
        self._reset_rule_ids()
        return self.rules._get_binarynodes()
    
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

        def get_binarynode_id(binarynodes, rule_id):
            for k, v in binarynodes.items():
                if rule_id == v['if_true']:
                    return k, True
                elif rule_id == v['if_false']:
                    return k, False
            return None, None

        self._reset_rule_ids()

        rule = self.get_rule(rule_id)
        casewhen_id  = get_casewhen_id(self._get_casewhens(), rule_id)
        binarynode_id, if_true = get_binarynode_id(self._get_binarynodes(), rule_id)

        if isinstance(rule, CaseWhen):
            rule.append_rule(new_rule)
        elif casewhen_id is not None:
            casewhen_rule = self.get_rule(casewhen_id)
            casewhen_rule.append_rule(new_rule, rule_id)
        elif isinstance(rule, EmptyRule):
            new_rule = rule.replace_rule(rule._rule_id, new_rule)
            binarynode_id, if_true = get_binarynode_id(self._get_binarynodes(), rule_id)
            if binarynode_id:
                node = self.get_rule(binarynode_id)
                if if_true:
                    node.if_true = new_rule
                    node._stored_params['if_true'] = new_rule
                else:
                    node.if_false = new_rule
                    node._stored_params['if_false'] = new_rule
        else:
            self.replace_rule(rule_id, CaseWhen(rules=[deepcopy(rule), new_rule]))
        self._reset_rule_ids()
        if isinstance(new_rule, BinarySplit):
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
        if isinstance(new_rule, BinarySplit):
            return new_rule.if_true
        return new_rule

    def remove_rule(self, rule_id:int):
        if rule_id == 0:
            self.rules = CaseWhen()
            self._stored_params['rules'] = self.rules
            return self.rules
        remove = self.rules.remove_rule(rule_id)
        self._reset_rule_ids()
        return remove
        
    def replace_binarynode_true_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        """Replace the if_true rule of a BinarySplit with new_rule

        Args:
            rule_id (int): rule_id of the BinarySplit
            new_rule (BusinessRule): New rule to replace the old rule

        Raises:
            ValueError: In case the rule with rule_id is not a BinarySplit
        """
        binary_node = self.get_rule(rule_id)
        if isinstance(binary_node, BinarySplit):
            binary_node.if_true = new_rule
            self.reset_rule_ids()
        else:
            raise ValueError(f"rule {rule_id} is not a BinarySplit!")
        
    def replace_binarynode_false_rule(self, rule_id:int, new_rule:BusinessRule)->None:
        """Replace the if_false rule of a BinarySplit with new_rule

        Args:
            rule_id (int): rule_id of the BinarySplit
            new_rule (BusinessRule): New rule to replace the old rule

        Raises:
            ValueError: In case the rule with rule_id is not a BinarySplit
        """
        binary_node = self.get_rule(rule_id)
        if isinstance(binary_node, BinarySplit):
            binary_node.if_false = new_rule
            self.reset_rule_ids()
        else:
            raise ValueError(f"rule {rule_id} is not a BinarySplit!")
        
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
        description = describe_businessrule(self)
        if self.final_estimator is not None:
            description += f"final_estimator = {self.final_estimator}\n"
        return description

    @staticmethod
    def _sort_cols_by_histogram_overlap(X:pd.DataFrame, y:pd.Series, cols:List[str]=None, *, bins:int=25, agg:str='minimum'):
        def histogram_overlap(x, y, bins=10, agg='minimum', numeric=True):
            n_classes = len(np.unique(y))
            hist_overlap = 0 if agg == 'average' else 1
            for label in np.unique(y):
                if numeric:
                    hist1, _ = np.histogram(x[y==label].values, bins, (x.min(), x.max()))
                    hist0, _ = np.histogram(x[y!=label].values, bins, (x.min(), x.max()))
                    tmp_hist_overlap = np.where(hist0<hist1, hist0, hist1).sum()/np.where(hist0>=hist1, hist0, hist1).sum()
                else:
                    vc1 = x[y==label].value_counts()/len(x[y==label])
                    vc0 = x[y!=label].value_counts()/len(x[y!=label])
                    tmp_hist_overlap = pd.DataFrame(dict(neg=vc0, pos=vc1)).min(axis=1).sum()/pd.DataFrame(dict(neg=vc0, pos=vc1)).max(axis=1).sum()
                if agg == 'average':
                    hist_overlap += tmp_hist_overlap/n_classes
                else:
                    hist_overlap = min(hist_overlap, tmp_hist_overlap)
            return hist_overlap 

        if cols is None:
            cols = X.columns

        col_scores = []
        for col in cols:
            col_scores.append((col, histogram_overlap(X[col], y.values, bins=bins, agg=agg, numeric=is_numeric_dtype(X[col]))))

        col_scores.sort(key=lambda x:x[1], reverse=True)
        return [col for col, score in col_scores]
    
    def plot(self, X:pd.DataFrame=None, y:pd.Series=None, color_scale:str='absolute', 
            highlight_id:int=None, scatter_text='name'):
        """
        Returns a plotly Figure of the rules. Uses the Reingolf-Tilford algorithm
        to generate the tree layout. 

        Args:
            X (pd.DataFrame, optional): input dataframe. If you pass both X and y
                then plot scatter markers will be colored by accuracy.
            y (pd.Series, optional): input labels. If you pass both X and y
                then plot scatter markers will be colored by accuracy.
            color_scale (str, {'absolute', 'relative'}): If 'absolute' scale
                the marker color from 0 to 1, which means that if all accuracies
                are relatively high, all the markers will look grean. If 'relative'
                markers color are scaled from lowest to highest.
            highlight_id (int, optional): node to highlight in the graph
            scatter_text (str, list, {'name', 'description'}): display the name
                (.e.g. 'PredictionRule') or the description (e.g. 'Always predict 1')
                next to the scatter markers. If you provide a list of str, with
                the same length as the nodes, these will be displayed instead. 

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

        if X is not None and y is not None:
            rule_scores_df = self.score_rules(X, y).drop_duplicates(subset=['rule_id'], keep='first')
            rule_accuracy = rule_scores_df['accuracy'].values
            cmin = 0 if color_scale == 'absolute' else rule_scores_df['accuracy'].dropna().min()
            cmax = 1 if color_scale == 'absolute' else rule_scores_df['accuracy'].dropna().max()
            if cmin == cmax: 
                cmin, cmax = 0, 1
            hovertext=[f"rule:{id}<br>"
                       f"{descr}<br>"
                       f"Prediction:{pred}<br>"
                       f"Accuracy:{acc:.2f}<br>"
                       f"Coverage:{cov:.2f}<br>"
                       f"n_inputs:{input}<br>"
                       f"n_outputs:{output}<br>"
                           for id, descr, pred, acc, cov, input, output in zip(
                                   graph.vs['rule_id'], 
                                   graph.vs['description'], 
                                   rule_scores_df['prediction'].values,
                                   rule_scores_df['accuracy'].values,
                                   rule_scores_df['coverage'].values,
                                   rule_scores_df['n_inputs'].values,
                                   rule_scores_df['n_outputs'].values)]
        else:
            rule_accuracy = np.full(len(layout), np.nan)
            cmin, cmax = 0, 1
            hovertext = [f"rule:{id}<br>"
                         f"{descr}<br>"
                            for id, descr in zip(
                                   graph.vs['rule_id'], 
                                   graph.vs['description'])]
            
    
        fig = go.Figure()

        if highlight_id is not None:
            fig.add_trace(go.Scatter(
                        x=[nodes_x[highlight_id]],
                        y=[nodes_y[highlight_id]],
                        mode='markers',
                        name='highlight',
                        hoverinfo='none',
                        marker=dict(
                            symbol='circle',
                            size=40,
                            color='purple',
                            opacity=0.5,
                            line=dict(width=3, color='violet'),
                        ),
            ))

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

        if isinstance(scatter_text, str) and scatter_text == 'name':
            scatter_text = graph.vs['name']
        elif isinstance(scatter_text, str) and scatter_text == 'description':
            scatter_text = graph.vs['description']
        elif isinstance(scatter_text, list) and len(scatter_text) == len(nodes_x):
            pass
        else:
            raise ValueError(f"ERROR: scatter_text should either be 'name', 'description' "
                f"or a list of str of the right length, but you passed {scatter_text}!")

        fig.add_trace(go.Scatter(
                        x=nodes_x,
                        y=nodes_y,
                        mode='markers+text',
                        name='nodes',
                        marker=dict(symbol='circle',
                                        size=18,
                                        color=rule_accuracy, #scores_df.accuracy.values,#'#6175c1',  
                                        colorscale="temps",
                                        reversescale=True,
                                        cmin=cmin,cmax=cmax,
                                        line=dict(color='rgb(50,50,50)', width=1),
                                    
                                        ),
                        text=[f"{id}: {desc}" for id, desc in zip(graph.vs['rule_id'], scatter_text)],
                        textposition="top right",
                        hovertemplate = "<b>%{hovertext}</b>",
                        hovertext=hovertext,
                        opacity=0.8
                        ))

        fig.update_layout(showlegend=False, dragmode='pan', margin=dict(b=0, t=0, l=0, r=0))
        fig.update_xaxes(visible=False, range=(min(nodes_x)-4, max(nodes_x)+4))
        fig.update_yaxes(visible=False)
        return fig

    def pie(self, X:pd.DataFrame, y:np.ndarray, rule_id:int=None, after=False,
                size=120, margin=0, showlegend=False):
        try:
            import plotly.express as px
            import plotly.graph_objs as go
        except ImportError:
            raise ImportError("Failed to load plotly, the plotting backend. "
                              "You need to install it seperately with pip install plotly.")

        if rule_id is not None:
            X, y = self.get_rule_input(rule_id, X, y, after)
        
        y_vc = y.value_counts().sort_index()
        return (go.Figure(
                    go.Pie(
                        labels=y_vc.index.astype(str), 
                        values=y_vc.values, 
                        marker=dict(colors=[px.colors.qualitative.Plotly[i] for i in y_vc.index]),
                        sort=False
                    )
                )
                .update_layout(showlegend=showlegend, width=size, height=size)
                .update_layout(margin=dict(t=margin, b=margin, l=margin, r=margin))
                .update_traces(textinfo='none', hoverinfo='none'))


    def parallel_coordinates(self, X:pd.DataFrame, y:np.ndarray, rule_id:int=None, 
                                cols:List[str]=None, labels=None, after=False,
                                ymin=None, ymax=None, sort_by_histogram_overlap=False):
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

        def encode_col(X, col):
            if is_numeric_dtype(X[col]):
                return dict(label=col, values=X[col])
            else:
                col_df = pd.DataFrame({col:reversed(y.groupby(X[col]).mean().index)})
                index_range = [0, len(col_df)-1] if len(col_df) > 1 else [0,1]
                return dict(range=index_range,
                    tickvals = col_df.index.tolist(), ticktext = col_df[col].tolist(),
                    label=col, values=X[col].replace(dict(zip(col_df[col], col_df.index))).values)

        if labels is None:
            labels = [str(i) for i in range(y.nunique())]

        if rule_id is not None:
            X, y = self.get_rule_input(rule_id, X, y, after)
        
        if cols is not None:
            X = X[cols]

        if sort_by_histogram_overlap:
            X = X[self._sort_cols_by_histogram_overlap(X, y)]

        if X.empty:
            fig = go.Figure()
            fig.update_layout(
                xaxis =  { "visible": False },
                yaxis = { "visible": False },
                annotations = [{   
                        "text": "No data!<br>Try setting after=False or selecting 'replace' instead of 'append'",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 14}
                    }])
            return fig

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

        dimensions = [encode_col(X, col) for col in X.columns]
        dimensions.append(dict(range=[0, len(labels)-1],
                    tickvals = list(range(len(labels))), ticktext = labels,
                    label="y", values=y))

        fig = go.Figure(data=
                    go.Parcoords(
                        line = dict(color=y,
                                    cmin=ymin, 
                                    cmax=ymax,
                                    colorscale=colorscale,
                                    colorbar=dict(tickvals = list(range(len(labels))), ticktext = labels),
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
            kind='node': suggest a LesserThanSplit rule based on a single step 
                DecisionTreeClassifier

        If you pass after=True, use the leftover data out of the rule instead
        to make a suggestion.

        Args:
            rule_id (int): [description]
            X (pd.DataFrame): [description]
            y (Union[pd.Series, np.ndarray]): [description]
            kind (str, {'prediction', 'rule', 'node'}, optional): The type of 
                rule to return. Either a PredictionRule, a LesserThan rule or 
                a LesserThanSplit. Defaults to 'rule'.
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
            return f"LesserThanSplit(col='{col}', cutoff={cutoff})"

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
            kind='node': suggest a LesserThanSplit rule based on a single step 
                DecisionTreeRegressor

        If you pass after=True, use the leftover data out of the rule instead
        to make a suggestion.

        Args:
            rule_id (int): [description]
            X (pd.DataFrame): [description]
            y (Union[pd.Series, np.ndarray]): [description]
            kind (str, {'prediction', 'rule', 'node'}, optional): The type of 
                rule to return. Either a PredictionRule, a LesserThan rule or 
                a LesserThanSplit. Defaults to 'rule'.
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
            return f"LesserThanSplit(col='{col}', cutoff={cutoff})"
    
    def __rulerepr__(self)->str:
        return "RuleRegressor"