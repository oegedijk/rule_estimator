__all__ = ['RuleClassifierDashboard']

from math import log10, floor
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from sklearn.model_selection import train_test_split

from .businessrule import *
from .splits import *
from .rules import *
from .estimators import RuleClassifier
from .plotting import *

instructions_markdown = """

This dashboard allows you to create a set of business rules that act as a classifier:
predicting an outcome label based on input features. This classifier can then be used
just like any other machine learning classifier from the python-based scikit-learn machine 
learning library.

### Adding new rules

New rules are either based on a **single feature** or **multiple features**. Single feature
rules are either based on a *cutoff* (for numerical features), or a *list of categories*
(for categorical features). Multi feature rules can be a combination of numerical
feature ranges and categorical feature lists. 

You can add three kinds of rules: "Split", "Predict Selected" and "Predict All".

A **Split** divides the data points in two branches: all input data for which the conditions holds
go left, and all other input data goes right. You can then add additional rules
both to the left branch and to the right branch. Once you have finished with one
branch you can select the other branch by clicking in the model graph.

**Predict Selected** applies a prediction label to all incoming data for which the 
condition holds, i.e. the data that is selected by this rule. Data for which the condition 
does not hold (i.e. unselected data) goes onlabeled (or rather: predicted nan).
Unlabeled data will be passed on the next rule.

**Predict All** unconditionally applies a prediction label to all incoming data.
After this rule there will be no remaining unlabeled data in this branch. You can
add a *Predict All* rule at the end of every branch to make sure that every input
receives a label.

The dashboard offers "suggest" buttons that helps you with selecting appropriate rules.
The feature suggest button will select a good feature for a new rule (based on maximum gini reduction, 
similar to DecisionTree algorithms). The cutoff suggest button will suggest a good
cutoff (again to maximize gini reduction). Once you have added a rule, automatically
the best next feature and cutoff for the remaining data will be selected. 

When you append a rule, the model automatically generates a `CaseWhen` block if needed. A CaseWhen
block evaluates a list of rules one by one, applying predictions if the rule condition holds
and otherwise passing the data to the next rule.

By default the dashboard will append a new rule to the currently selected rule (creating 
a new CaseWhen block if needed). Thus when you select a rule, the plots will display only the 
data points that have not been labeled after this rule. You can also select to replace the current rule. In that
case the dashboard will display all the data coming into the the current rule.

### Inspecting the model

The decision rules together form a "model" that takes input data and outputs predicted
labels. You can inspect the model in the "Model" section. You can view a Graphical
representation of the model or a textual Description.

Once you are happy with your model you can export a scikit-learn compatible model as a 
python pickle file from the navbar "save as..." menu. You can also export a `model.yaml` file 
that you can use to instantiate the model from python with `RuleClassifier.from_yaml("model.yaml")`. 
Finally you can also copy-paste the python code that will generate the model. Using the upload
button you can load `.pkl` or `.yaml` files that you have previously exported. 

You can delete individual rules or reset the entire model, but these actions cannot be undone, so be careful.

### Performance

The dashboard offers a few basic metrics for measuring the performance of the model.

A confusion matrix shows the absolute and relative number of True Negatives, True Positives, False Negatives
and False Positives. 

Accuracy (number of correctly predicted labels), 
precision (number of positively labeled predictions that were in fact positive), 
recall (number of positively labeled data points that were in fact labeled positive) 
as well as the f1-score (combination of predicion and recall) get calculated.

Coverage is defined as the fraction of input data that receive a label.

### Training data vs Validation data

It is good practice to split your data into a training set and a validation set. 
You construct your rules using the training set, and then evaluate them using the
validation set. There is a toggle in the navbar to switch between the two data sets.

Ideally you would keep a test set apart completely which you only use to evaluate
your final model.
"""

class RuleClassifierDashboard:
    
    def __init__(self, X, y, X_val=None, y_val=None, val_size=None, model=None, labels=None, port=8050):
        
        self.model = model if model is not None else RuleClassifier()
            
        if X_val is not None and y_val is not None:
            self.X, self.y = X, pd.Series(y)
            self.X_val, self.y_val = X_val, pd.Series(y_val)
        elif val_size is not None:
            self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=0.2, stratify=y)
            self.y, self.y_val = pd.Series(self.y), pd.Series(self.y_val)
        else:
            self.X, self.y = X, pd.Series(y)
            self.X_val, self.y_val = None, None

        if labels is None:
            self.labels = [str(i) for i in range(self.y.nunique())]
        else:
            self.labels = labels

        self.cats = [col for col in X.columns if not is_numeric_dtype(X[col])]
        self.non_cats = [col for col in X.columns if is_numeric_dtype(X[col])]
        self.initial_col = self.model._sort_cols_by_gini_reduction(X, y)[0]
            
        self.port = port
        
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "RuleClassifier"
        self.app.layout = self.layout()
        self.register_callbacks(self.app)

    @staticmethod
    def _process_constraintrange(ranges:List, ticktext:List=None, sig:int=4)->List:
        """helper function to format selections in a go.Parcoords plot. 
        For numerical features range bounds are rounded to four significant digits.
        
        For categorical features the ranges are converted to a list of categories.

        Args:
            ranges (List[List]) either a single list of two bounds or a list of lists of two bounds
            ticktext (List): the ticktext property of a go.Parcoords fig lists the order
                in which the categorical features are displayed in the plot. This is used
                to convert ranges to a list of categories.
            sig (int): number of significant digits to round to. So e.g 123553 is round to 123600
                and 0.000034512553 is round to 0.00003451
        """
        def round_sig(x, sig):
            return round(x, sig-int(floor(log10(abs(x))))-1)
        def round_range(range_list:List, sig)->List:
            return [round_sig(range_list[0], sig), round_sig(range_list[1], sig)]
        def range_to_cats(range_list, ticktext):
            return [tickval for i, tickval in enumerate(ticktext) if i >= range_list[0] and i <= range_list[1]]
        def process_range(range_list, sig, ticktext):
            if ticktext is not None:
                return range_to_cats(range_list, ticktext)
            else:
                return round_range(range_list, sig)

        if len(ranges) == 2 and not isinstance(ranges[0], list):
            return process_range(ranges, sig, ticktext)
        else:
            return [process_range(range_list, sig, ticktext) for range_list in ranges]

    @staticmethod
    def _get_stepsize(min, max, steps=500, sig=3):
        """returns step size range (min, max) into steps, rounding to sig significant digits"""
        return round((max-min)/steps, sig-int(floor(log10(abs((max-min)/steps))))-1) 

    @staticmethod
    def _round_cutoff(cutoff, min, max, steps=500, sig=3):
        """returns cutoff for dividing div into steps, rounding to sig significant digits"""
        return round(cutoff, sig-int(floor(log10(abs((max-min)/steps))))-1)

    
    def _get_range_dict_from_parallel_plot(self, fig):
        """
        Extracts a range_dict from a go.Parcoords() figure.
        a range_dict is of the format e.g.:
        range_dict = {
            'petal length (cm)': [[4.1, 4.7], [5.2, 7.5]],
            'petal width (cm)': [1.6, 2.6],
            'sex' : ['male']
        }

        """
        plot_data = fig['data'][0].get('dimensions', None)
        range_dict = {}
        for col_data in plot_data:
            if col_data['label'] != 'y' and 'constraintrange' in col_data:
                range_dict[col_data['label']] = self._process_constraintrange(
                    col_data['constraintrange'], col_data['ticktext'] if 'ticktext' in col_data else None)
        return range_dict
    
    @staticmethod
    def _cats_to_range(cats:List, cats_order:List):
        """converts a list of categories of a categorical feature to a list of ranges
        for a go.Parcoords plot. 

        Args:
            cats: list of categories to encode
            cats_order: list of cats in the order that they are displayed in the go.Parcoords plot
        """
        return [[max(0, cats_order.index(cat)-0.25),  min(len(cats_order)-1, cats_order.index(cat)+0.25)] for cat in cats]

    @staticmethod
    def _get_callback_trigger():
        """Returns the dash id of the component that triggered a callback
        Is used when there are multiple Input's and the Output depends on which
        Input triggered the callback.
        """
        return dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    def _get_model(self, json_model=None):
        """Returns a model instance from a json model definition. 
        If the json_model is still empty (=None) then returns self.model"""
        if json_model:
            return RuleClassifier.from_json(json_model)
        else:
            return RuleClassifier.from_json(self.model.to_json())

    def _get_X_y(self, train_or_val='train'):
        """Returns either the full training set (X,y ) or the full validation set (X, y)"""
        X, y = (self.X, self.y) if train_or_val == 'train' else (self.X_val, self.y_val)
        return X, y

    @staticmethod
    def _infer_after(append_or_replace):
        """infer whether to use data that has not been assigned after applying a 
        rule with rule_id (after=True) or to use all data that reaches a certain rule_id 
        (after=False) by checking the append_or_replace toggle"""
        return (append_or_replace == 'append')

    def _get_model_X_y(self, model=None, train_or_val='train', rule_id=0, after=False):
        """returns a (model, X, y) tuple

        Args:
            train_or_val: return 'train' data or 'val' data
            rule_id: return data for rule rule_id
            after: If True return data that has not been assigned
                a prediction after rule rule_id. Can also be a string
                in {'append', 'replace}, in which case it will get inferred.

        """
        if model is None or not isinstance(model, RuleClassifier):
            model = self._get_model(model)         

        X, y = self._get_X_y(train_or_val)

        if not isinstance(after, bool):
            if after in {'append', 'replace'}:
                after = self._infer_after(after)
            else:
                raise ValueError(f"After should either be a bool or in 'append', 'replace',"
                                f" but you passed {after}!")
        if rule_id == 0 and not after:
            return model, X, y
        X, y = model.get_rule_input(rule_id, X, y, after)
        return model, X, y

    def _change_model(self, model, rule_id, new_rule, append_or_replace='append'):
        """Update a model with a new rule.

        Args:
            model: model to be updated
            rule_id: rule to which new rule should be appended or replaced
            new_rule: instance of new rule
            append_or_replace ({'append', 'replace'})
        """
        if not isinstance(model, RuleClassifier):
            model = self._get_model(model)            
        if append_or_replace=='append':
            new_rule_id = model.append_rule(rule_id, new_rule)
        elif append_or_replace=='replace':
            new_rule = model.replace_rule(rule_id, new_rule)
            new_rule_id = new_rule._rule_id
        return new_rule_id, model
        
    def layout(self):
        """Returns the dash layout of the dashboard. """
        return dbc.Container([
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink(children=[html.Div("Instructions")], id="instructions-open-button", n_clicks=0)),
                    dbc.NavItem(dbc.NavLink(children=[html.Div("Upload")], id="upload-button", n_clicks=0)),
                    dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("as .yaml", id='download-yaml-button', n_clicks=None),
                            dcc.Download(id='download-pickle'),
                            dbc.DropdownMenuItem("as pickle", id='download-pickle-button', n_clicks=None),
                            dcc.Download(id='download-yaml'),
                        ],
                        nav=True,
                        in_navbar=True,
                        label="Save model",
                    ),
                    html.Div([
                        dbc.Select(
                            options=[{'label':'Training data', 'value':'train'},
                                    {'label':'Validation data', 'value':'val'}],
                            value='train',
                            id='train-or-val'
                        ),
                        dbc.Tooltip("Display values using the training data set or the "
                            "validation data set. Use the training set to build your rules, "
                            "and the validation to measure performance and find rules that "
                            "do not generalize well.", target='train-or-val'),
                    ], style=dict(display="none") if self.X_val is None else dict()),
                ],
                brand="RuleClassifierDashboard",
                brand_href="https://github.com/oegedijk/rule_estimator/",
                color="primary",
                dark=True,
            ),

            dbc.Modal(
                [
                    dbc.ModalHeader("Dashboard Intructions"),
                    dbc.ModalBody([dcc.Markdown(instructions_markdown)]),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="instructions-close-button", className="ml-auto", n_clicks=0)
                    ),

                ],
                id="instructions-modal",
                is_open=True,
                size="xl",
            ),

            # helper storages as a workaround for dash limitation that each output can
            # only be used in a single callback
            dcc.Store(id='updated-rule-id'),
            dcc.Store(id='parallel-updated-rule-id'),
            dcc.Store(id='density-num-updated-rule-id'), 
            dcc.Store(id='density-cats-updated-rule-id'), 
            dcc.Store(id='removerule-updated-rule-id'),
            dcc.Store(id='resetmodel-updated-rule-id'),

            dcc.Store(id='model-store'),
            dcc.Store(id='parallel-updated-model'),
            dcc.Store(id='density-num-updated-model'), 
            dcc.Store(id='density-cats-updated-model'),
            dcc.Store(id='removerule-updated-model'),
            dcc.Store(id='resetmodel-updated-model'),
            dcc.Store(id='uploaded-model'),

            dcc.Store(id='update-model-performance'),
            dcc.Store(id='update-model-graph'),
            dcc.Store(id='added-num-density-rule'),
            dcc.Store(id='added-cats-density-rule'),

            html.Div(id='upload-div', children=[
                dcc.Upload(
                    id='upload-model',
                    children=html.Div([
                        'Drag and drop a .yaml or .pkl model or ',
                        html.A('Select File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
            ], style=dict(display="none")),
               
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col([
                                    html.H3("Add New Rule", className='card-title'),
                                ], md=8),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Select(
                                            options=[{'label':'Append new rule after (show data out)', 'value':'append'},
                                                    {'label':'Replace rule (show data in)', 'value':'replace'}],
                                            value='append',
                                            id='append-or-replace'),
                                        dbc.Tooltip("When you select to 'append' a rule, the plots will display all the data "
                                                    "that is still unlabeled after the selected rule. When you add a rule it will appended"
                                                    " with a CaseWhen block. "   
                                                    "When you select 'replace' the plots will display all the data going *into* "
                                                    "the selected rule. When you add a rule, it will replace the existing rule.",
                                                    target='append-or-replace'),
                                    ]),
                                ], md=4),
                            ]),  
                        ]),
                        dbc.CardBody([
                            dcc.Tabs(id='rule-tabs', value='density-tab', 
                                children=[
                                    dcc.Tab(label="Single Feature Rules", id='density-tab', value='density-tab', children=[
                                        dbc.Card([
                                            dbc.CardBody([
                                                dbc.Row([
                                                    dbc.Col([
                                                        dbc.Label("Feature"),
                                                        dcc.Dropdown(id='density-col', 
                                                                    options=[{'label':col, 'value':col} for col in self.X.columns], 
                                                                    value=self.initial_col, 
                                                                    clearable=False),
                                                    ], md=9),
                                                    dbc.Col([
                                                            dbc.FormGroup([
                                                                html.Div([
                                                                    dbc.Button("Suggest", id='density-col-suggest-button', color="primary", size="sm",
                                                                        style={'position':'absolute', 'bottom':'25px'}),
                                                                    dbc.Tooltip("Select the feature with the largest gini reduction potential.", 
                                                                                target='density-col-suggest-button'),
                                                                ], style={'display': 'flex', 'flex-direction':'column'}),
                                                            ]),
                                                        ], md=1, ),
                                                    dbc.Col([
                                                        dbc.FormGroup([
                                                            dbc.Label("Sort features", html_for='parallel-sort'),
                                                            dbc.Select(id='density-sort',
                                                                options=[dict(label=s, value=s) for s in ['dataframe', 'alphabet', 'histogram overlap', 'gini reduction']],
                                                                value='gini reduction', 
                                                                style=dict(height=40, width=150, horizontalAlign = 'right'),
                                                                bs_size="sm",
                                                            ),
                                                            dbc.Tooltip("You can sort the features by their potential gini reduction (default), "
                                                                        "or alphabetically, by order in the dataframe, or by histogram overlap.", 
                                                                        target='density-sort'),
                                                        ]),
                                                    ], md=2),
                                                ]),
                                                html.Div([
                                                    dbc.Row([
                                                        dbc.Col([
                                                            dcc.Graph(id='density-num-plot', config=dict(modeBarButtons=[[]], displaylogo=False)),
                                                        ], md=10),
                                                        dbc.Col([
                                                            dbc.Label("All", id='density-num-pie-all-label'),
                                                            dcc.Graph(id='density-num-pie-all', config=dict(modeBarButtons=[[]], displaylogo=False), style=dict(marginBottom=20)),
                                                            dbc.Label("Selected", id='density-num-pie-selected-label'),
                                                            dcc.Graph(id='density-num-pie-selected', config=dict(modeBarButtons=[[]], displaylogo=False), style=dict(marginBottom=20)),
                                                            dbc.Label("Not Selected", id='density-num-pie-not-selected-label'),
                                                            dcc.Graph(id='density-num-pie-not-selected', config=dict(modeBarButtons=[[]], displaylogo=False)),
                                                        ], md=2),
                                                    ]),
                                                    dbc.Row([
                                                        dbc.Col([
                                                            html.Div([
                                                                dbc.FormGroup([
                                                                    dcc.RangeSlider(id='density-num-cutoff', 
                                                                        allowCross=False, min=0, max=10,
                                                                        tooltip=dict(always_visible=True)),
                                                                ]), 
                                                            ], style=dict(marginLeft=45)),                                                                    
                                                        ], md=9),
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                dbc.Select(
                                                                    options=[
                                                                        {'label':'Range', 'value':'range'},
                                                                        {'label':'Lesser than', 'value':'lesser_than'},
                                                                        {'label':'Greater than', 'value':'greater_than'}
                                                                    ],
                                                                    value='lesser_than',
                                                                    id='density-num-ruletype',
                                                                    bs_size="sm",
                                                                ),
                                                                dbc.Tooltip("LesserThan rules ignore the lower bound and GreaterThan "
                                                                            "rules ignore the upper bound. RangeRules respect both upper and lower bounds.", 
                                                                                target='density-num-ruletype'),
                                                            ]), 
                                                        ], md=2),
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                html.Div([
                                                                    dbc.Button("Suggest", id='density-num-suggest-button', color="primary", size="sm"),
                                                                    dbc.Tooltip("Select the best split point that minimizes the weighted average gini after "
                                                                                "the split, similar to a DecisionTree",  target='density-num-suggest-button'),
                                                                ], style={'vertical-align': 'bottom'})  
                                                            ]),
                                                        ], md=1),
                                                    ], form=True),
                                                    dbc.Row([
                                                        dbc.Col([
                                                            html.Hr()
                                                        ])
                                                    ]),
                                                    dbc.Row([
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                dbc.Button("Split", id='density-num-split-button', 
                                                                        color="primary", size="m", style=dict(width=150)),
                                                                dbc.Tooltip("Generate a Split rule where all observations where the condition holds go left "
                                                                            "and all other observations go right.", target='density-num-split-button'),
                                                            ]),
                                                        ], md=6),
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                html.Div([
                                                                    dbc.Select(id='density-num-prediction', options=[
                                                                            {'label':f"{y}. {label}", 'value':str(y)} for y, label in enumerate(self.labels)],
                                                                                value=str(len(self.labels)-1), bs_size="md"),
                                                                    dbc.Button("Predict Selected", id='density-num-predict-button', color="primary",
                                                                                size="m", style=dict(marginLeft=10, width=400)),
                                                                    dbc.Tooltip("Apply the prediction for all observation for which the condition holds. All "
                                                                                "other observations will either be covered by later rules, or predicted nan.", 
                                                                                target='density-num-predict-button'),
                                                                    dbc.Button(children="Predict All", id='density-num-predict-all-button', color="primary",
                                                                                size="m", style=dict(marginLeft=10, width=400)),
                                                                    dbc.Tooltip(children="Apply the prediction to all observations regardless whether the condition holds. Use this "
                                                                                "rule as the final rule in order to prevent any nan's showing up in your predictions.",
                                                                                target='density-num-predict-all-button'),
                                                                ], style={'width': '100%', 'display': 'flex', 'align-items': 'right', 'justify-content': 'right'}),
                                                            ], row=True),
                                                        ], md=6),
                                                    ], form=True),
                                                ], id='density-num-div', style=dict(display="none")),
                                                html.Div([
                                                    dbc.Row([
                                                        dbc.Col([
                                                            dcc.Graph(id='density-cats-plot', config=dict(modeBarButtons=[[]], displaylogo=False)),
                                                        ], md=10),
                                                        dbc.Col([
                                                            dbc.Label("All", id='density-cats-pie-all-label'),
                                                            dcc.Graph(id='density-cats-pie-all', config=dict(modeBarButtons=[[]], displaylogo=False), style=dict(marginBottom=20)),
                                                            dbc.Label("Selected", id='density-cats-pie-selected-label'),
                                                            dcc.Graph(id='density-cats-pie-selected', config=dict(modeBarButtons=[[]], displaylogo=False), style=dict(marginBottom=20)),
                                                            dbc.Label("Not selected", id='density-cats-pie-not-selected-label'),
                                                            dcc.Graph(id='density-cats-pie-not-selected', config=dict(modeBarButtons=[[]], displaylogo=False)),
                                                        ], md=2),
                                                    ]),
                                                    dbc.Row([
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                dcc.Dropdown(id='density-cats-cats', value=[], multi=True,
                                                                    style=dict(marginBottom=20)),
                                                            ]), 
                                                        ], md=8),
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                html.Div([
                                                                    dbc.Button("Invert", id='density-cats-invert-button', color="primary", size="sm"),
                                                                    dbc.Tooltip("Invert the category selection: all selected will be unselected and "
                                                                                    "all not selected will be selected", target='density-cats-invert-button'),
                                                                ], style={'vertical-align': 'bottom'}),
                                                            ]),
                                                        ], md=1),
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                html.Div([
                                                                    dbc.Button("Suggest", id='density-cats-suggest-button', color="primary", size="sm"),
                                                                    dbc.Tooltip("Suggest best single category to select to minimize weighted gini. Either the category "
                                                                                "or the inverse will be selected, whichever has the lowest gini", target='density-cats-suggest-button'),
                                                                ], style={'vertical-align': 'bottom'}) 
                                                            ]),
                                                        ], md=1),
                                                        dbc.Col([
                                                            html.Div([
                                                                dbc.FormGroup([
                                                                    dbc.Checklist(
                                                                        options=[{"label":  "Display Relative", "value": True}],
                                                                        value=[],
                                                                        id='density-cats-percentage',
                                                                        inline=True,
                                                                        switch=True,
                                                                    ),
                                                                    dbc.Tooltip("Display barcharts as percentages instead of counts.", target='density-cats-percentage'),
                                                                ]),
                                                            ], style={'display': 'flex', 'align-items': 'right', 'justify-content': 'flex-end'}),
                                                        ], md=2),
                                                    ], form=True),
                                                    dbc.Row([
                                                        dbc.Col([
                                                            html.Hr()
                                                        ])
                                                    ]),
                                                    dbc.Row([
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                dbc.Button("Split", id='density-cats-split-button', 
                                                                    color="primary", size="m", style=dict(width=150)),
                                                                dbc.Tooltip("Generate a Split rule where all observations where the condition holds go left "
                                                                            "and all other observations go right.", target='density-cats-split-button'),
                                                            ]),
                                                        ], md=6),
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                html.Div([
                                                                    dbc.Select(id='density-cats-prediction', options=[
                                                                            {'label':f"{y}. {label}", 'value':str(y)} for y, label in enumerate(self.labels)],
                                                                            value=str(len(self.labels)-1), #clearable=False, 
                                                                            bs_size="md"),
                                                                    dbc.Button("Predict Selected", id='density-cats-predict-button', color="primary",
                                                                                size="m", style=dict(marginLeft=10, width=400)),
                                                                    dbc.Tooltip("Apply the prediction for all observation for which the condition holds. All "
                                                                                "other observations will either be covered by later rules, or predicted nan.",
                                                                                target='density-cats-predict-button'),
                                                                    dbc.Button("Predict All", id='density-cats-predict-all-button', color="primary",
                                                                                size="m", style=dict(marginLeft=10, width=400)),
                                                                    dbc.Tooltip("Apply the prediction to all observations regardless whether the condition holds. Use this "
                                                                            "rule as the final rule in order to prevent any nan's showing up in your predictions.",
                                                                                target='density-cats-predict-all-button'),
                                                                ], style = {'width': '100%', 'display': 'flex', 'align-items': 'right', 'justify-content': 'right'})    
                                                            ], row=True),
                                                        ], md=6),
                                                    ], form=True),
                                                ], id='density-cats-div'),
                                            ]),
                                        ]),
                                    ]),
                                    dcc.Tab(label="Multi Feature Rules", id='parallel-tab', value='parallel-tab', children=[
                                        dbc.Card([
                                            dbc.CardBody([
                                                dbc.Row([
                                                    dbc.Col([
                                                        dbc.FormGroup([
                                                                dbc.Label("Display Features", html_for='parallel-cols', id='parallel-cols-label'),
                                                                dcc.Dropdown(id='parallel-cols', multi=True,
                                                                    options=[{'label':col, 'value':col} 
                                                                            for col in self.X.columns],
                                                                    value = self.X.columns.tolist()),
                                                                dbc.Tooltip("Select the features to be displayed in the Parallel Plot", 
                                                                        target='parallel-cols-label'),
                                                            ]),
                                                    ], md=10),
                                                    dbc.Col([
                                                        dbc.FormGroup([
                                                            dbc.Label("Sort features", html_for='parallel-sort', id='parallel-sort-label'),
                                                            dbc.Select(id='parallel-sort',
                                                                options=[dict(label=s, value=s) for s in ['dataframe', 'alphabet', 'histogram overlap', 'gini reduction']],
                                                                value='gini reduction', 
                                                                style=dict(height=40, width=150, horizontalAlign = 'right'),
                                                                bs_size="sm",
                                                            ),
                                                            dbc.Tooltip("Sort the features in the plot from least histogram overlap "
                                                                        "(feature distributions look the most different for different "
                                                                        "values of y) on the right, to highest histogram overlap on the left.", 
                                                                        target='parallel-sort-label'),
                                                        ])
                                                    ], md=2),
                                                ], form=True),
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Small('You can select multiple ranges in the parallel plot to define a multi feature rule.', className="text-muted",),
                                                        dcc.Graph(id='parallel-plot'),
                                                        html.Div(id='parallel-description'), 
                                                        
                                                    ])
                                                ]),
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Div([
                                                            dbc.Row(dbc.Col([
                                                                dbc.Label("All", id='parallel-pie-all-label', html_for='parallel-pie-all'),
                                                                dbc.Tooltip("Label distribution for all observations in the parallel plot above.", 
                                                                        target='parallel-pie-all-label'),
                                                                dcc.Graph(id='parallel-pie-all', config=dict(modeBarButtons=[[]], displaylogo=False)),   
                                                            ])),
                                                        ], style={'display':'flex', 'justify-content': 'center', 'align-items': 'center'}),
                                                    ]),
                                                    dbc.Col([
                                                        html.Div([
                                                            dbc.Row(dbc.Col([
                                                                dbc.Label("Selected", id='parallel-pie-selection-label'),
                                                                dbc.Tooltip("Label distribution for all the feature ranges selected above", 
                                                                        target='parallel-pie-selection-label'),
                                                                dcc.Graph(id='parallel-pie-selection', config=dict(modeBarButtons=[[]], displaylogo=False)), 
                                                            ])),  
                                                        ], style={'display':'flex', 'justify-content': 'center', 'align-items': 'center'}),
                                                    ]),
                                                    dbc.Col([
                                                        html.Div([
                                                            dbc.Row(dbc.Col([
                                                                dbc.Label("Not selected", id='parallel-pie-non-selection-label'),
                                                                dbc.Tooltip("Label distribution for all the feature ranges not selected above", 
                                                                        target='parallel-pie-non-selection-label'),
                                                                dcc.Graph(id='parallel-pie-non-selection', config=dict(modeBarButtons=[[]], displaylogo=False)),
                                                            ])),
                                                        ], style={'display':'flex', 'justify-content': 'center', 'align-items': 'center'}),
                                                    ]),
                                                ]),
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Hr()
                                                    ]),
                                                ]),
                                                dbc.Row([
                                                    dbc.Col([
                                                        dbc.FormGroup([
                                                            dbc.Button("Split", id='parallel-split-button', 
                                                                            color="primary", size="m", style=dict(width=150)),
                                                            dbc.Tooltip("Make a split using the selection in the Parallel Plot. "
                                                                    "Data in the selected ranges goes left (true), all other data "
                                                                    "goes right (false).", target='parallel-split-button'),
                                                        ]),
                                                    ], md=6),
                                                    dbc.Col([
                                                        dbc.FormGroup([
                                                            html.Div([
                                                                dbc.Select(id='parallel-prediction', options=[
                                                                    {'label':f"{y}. {label}", 'value':str(y)} for y, label in enumerate(self.labels)],
                                                                    value=str(len(self.labels)-1), 
                                                                    bs_size="md"),
                                                                dbc.Tooltip("The prediction to be applied. Either to all data ('Predict All'), "
                                                                            "or to the selected data ('Predict Selected'). Will get automatically "
                                                                            "Inferred from the selected data in the Parallel Plot.", target='parallel-prediction'),
                                                                dbc.Button("Predict Selected", id='parallel-predict-button', 
                                                                        color="primary", size="m", style=dict(marginLeft=10, width=400)),
                                                                dbc.Tooltip("Apply the prediction to all data within the ranges "
                                                                            "selected in the Parallel Plot.", target='parallel-predict-button'),
                                                                dbc.Button("Predict All", id='parallel-predict-all-button', 
                                                                            color="primary", size="m", style=dict(marginLeft=10, width=400)),
                                                                dbc.Tooltip("Add a PredictionRule: Apply a single uniform prediction to all the "
                                                                            "data without distinction.", target='parallel-predict-all-button'),
                                                            ], style = {'width': '100%', 'display': 'flex', 'align-items': 'right', 'justify-content': 'right'})     
                                                        ], row=True),
                                                    ], md=6),
                                                ], form=True),
                                            ]),
                                        ]),
                                    ]),
                            ], style=dict(marginTop=5)),
                        ]),
                    ]),
                ]),
            ], style=dict(marginBottom=20, marginTop=20)),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col([
                                    html.H3("Model", className='card-title'),
                                ], md=6),
                                dbc.Col([
                                    html.Div([
                                        dbc.FormGroup([
                                            dbc.Label("Selected rule:", id='selected-rule-id-label', 
                                                        html_for='selected-rule-id', className="mr-2"),
                                            dcc.Dropdown(id='selected-rule-id', options=[
                                                {'label':str(ruleid), 'value':int(ruleid)} 
                                                    for ruleid in range(self.model._get_max_rule_id()+1)],
                                                    value=0, clearable=False, style=dict(width=80)),
                                            dbc.Tooltip("You can either select a rule id here or by clicking in the model graph.", 
                                                            target='selected-rule-id-label'),
                                            dcc.ConfirmDialogProvider(
                                                children=html.Button("Remove Rule", id='remove-rule-button', className="btn btn-danger btn-sm",
                                                                        style=dict(marginLeft=5)),
                                                id='remove-rule-confirm',
                                                message='Warning! Once you have removed a rule there is no undo button or ctrl-z! Are you sure?'
                                            ),
                                            dbc.Tooltip("Remove the selected rule from the model. Warning! Cannot be undone!", 
                                                        target='remove-rule-button'),
                                            dcc.ConfirmDialogProvider(
                                                children=html.Button("Reset Model", id='reset-model-button', className="btn btn-danger btn-sm",
                                                                        style=dict(marginLeft=10)),
                                                id='reset-model-confirm',
                                                message='Warning! Once you have reset the model there is no undo button or ctrl-z! Are you sure?'
                                            ),
                                            dbc.Tooltip("Reset the model to the initial state. Warning! Cannot be undone!", 
                                                        target='reset-model-button'),
                                        ], row=True),
                                    ], style={'display': 'flex', 'align-items': 'right', 'justify-content': 'flex-end'}),
                                ], md=6), 
                            ], form=True, style={'marginLeft':8, 'marginTop':10}),
                        ]),
                        dbc.CardBody([
                            dcc.Tabs(id='model-tabs', value='model-graph-tab', children=[
                                dcc.Tab(id='model-graph-tab', value='model-graph-tab', label='Graph', 
                                        children=html.Div([
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        dbc.Label("Color scale: ", html_for='absolute-or-relative', className="mr-2"),
                                                        dbc.Select(id='model-graph-color-scale',
                                                            options=[{'label':'Absolute', 'value':'absolute'},
                                                                     {'label':'Relative', 'value':'relative'}],
                                                            value='absolute',
                                                            bs_size="sm", style=dict(width=100)),
                                                        dbc.Tooltip("Color the rules by either by absolute accuracy (0%=red, 100%=green), "
                                                                "or by relative accuracy (lowest accuracy=red, highest=green)", target='model-graph-color-scale'),
                                                    ], row=True), 
                                                ], md=3),
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        dbc.Label("Display: ", html_for='model-graph-scatter-text', className="mr-2"),
                                                        dbc.Select(id='model-graph-scatter-text',
                                                            options=[dict(label=o, value=o) for o in ['name', 'description', 'coverage', 'accuracy']],
                                                            value='description', 
                                                            bs_size="sm",style=dict(width=130)),
                                                        dbc.Tooltip("You can display rule description, accuracy or coverage next to the markers on the model graph.", 
                                                                        target='model-graph-scatter-text'),
                                                    ], row=True), 
                                                ], md=3),
                                            ], form=True, style=dict(marginTop=8, marginLeft=16, marginRight=16)),
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Graph(id='model-graph'),
                                                ]),
                                            ]),   
                                            dbc.Row([
                                                dbc.Col([
                                                    html.P("You can select a rule by clicking on it in the Graph"),
                                                ])
                                            ])
                                        ])
                                ),
                                dcc.Tab(id='model-description-tab', value='model-description-tab', label='Description', 
                                        children=html.Div([dbc.Row([dbc.Col([
                                                html.Div([
                                                    dcc.Clipboard(
                                                        target_id="model-description",
                                                        title="copy"),
                                                ], style={'display': 'flex', 'align-items': 'right', 'justify-content': 'flex-end'}),
                                                dcc.Markdown(id='model-description'),
                                                
                                            ])])])
                                ),
                                dcc.Tab(id='model-yaml-tab', value='model-yaml-tab', label='.yaml', 
                                        children=html.Div([dbc.Row([dbc.Col([
                                                html.Div("To instantiate a model from a .yaml file:", style=dict(marginBottom=10)),
                                                dcc.Markdown("```\nfrom rule_estimator import RuleClassifier\n"
                                                            "model = RuleClassifier.from_yaml('model.yaml')\n"
                                                            "model.predict(X_test)\n```"),
                                                html.B("model.yaml:"),
                                                html.Div([
                                                    dcc.Clipboard(
                                                        target_id="model-yaml",
                                                        title="copy"),
                                                ], style={'display': 'flex', 'align-items': 'right', 'justify-content': 'flex-end'}),
                                                dcc.Markdown(id='model-yaml'),
                                                
                                            ])])])
                                ),
                                dcc.Tab(id='model-code-tab', value='model-code-tab', label='Python Code', 
                                        children=[
                                            html.Div([
                                                    dcc.Clipboard(
                                                        target_id="model-code",
                                                        title="copy"),
                                                ], style={'marginTop':20, 'display': 'flex', 'align-items': 'right', 'justify-content': 'flex-end'}),
                                            dcc.Markdown(id='model-code'),
                                            
                                        ]
                                ),
                            ]),
                        ]),
                    ]),
                ]),
            ], style=dict(marginBottom=20, marginTop=20)),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H3("Performance"),
                        ]),
                        dbc.CardBody([
                            dcc.Tabs(id='performance-tabs', value='performance-overview-tab', children=[
                                dcc.Tab(id='performance-rules-tab2', value='performance-overview-tab', label="Model Performance", 
                                    children=html.Div([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Select(id='model-performance-select',
                                                    options=[
                                                        dict(label='Full model', value='model'),
                                                        dict(label='Single Rule', value='rule'),
                                                    ],
                                                    value='full_model',
                                                ),
                                                dbc.Tooltip("Display performance for the model as a whole or "
                                                            "only for the currently selected rule", target='model-performance-select'),
                                            ]),
                                        ]),
                                        dbc.Row([
                                            dbc.Col([
                                                dcc.Graph(id='model-performance-confmat'),
                                            ], md=6),
                                            dbc.Col([
                                                html.Div(id='model-performance-metrics'),
                                                html.Div(id='model-performance-coverage'),
                                            ]),
                                        ]),
                                    ])),
                                dcc.Tab(id='performance-rules-tab', value='performance-rules-tab', label="All rules",
                                    children=html.Div(id='model-performance')),
                            ]),
                        ]),
                    ]),  
                ]),
            ], style=dict(marginBottom=20, marginTop=20)),
        ])
                                
    def register_callbacks(self, app):

        @app.callback(
            Output("download-yaml", "data"),
            Input("download-yaml-button", "n_clicks"),
            State('model-store', 'data')
        )
        def download_yaml(n_clicks, model):
            if n_clicks is not None:
                model = self._get_model(model)
                return dict(content=model.to_yaml(), filename="model.yaml")
            raise PreventUpdate
        
        @app.callback(
            Output("download-pickle", "data"),
            Input("download-pickle-button", "n_clicks"),
            State('model-store', 'data')
        )
        def download_pickle(n_clicks, model):
            if n_clicks is not None:
                model = self._get_model(model)
                return dcc.send_bytes(model.pickle().read(), "model.pkl")
            raise PreventUpdate

        @app.callback(
            Output('updated-rule-id', 'data'),
            Input('parallel-updated-rule-id', 'data'),
            Input('density-num-updated-rule-id', 'data'),
            Input('density-cats-updated-rule-id', 'data'),
            Input('removerule-updated-rule-id', 'data'),
            Input('resetmodel-updated-rule-id', 'data'),
            Input('model-graph', 'clickData'),
            Input('uploaded-model', 'data')
        )
        def update_model(parallel_rule_id, num_rule_id, cats_rule_id, 
                        removerule_rule_id, resetmodel_rule_id, clickdata,
                        uploaded_model):
            trigger = self._get_callback_trigger()
            if trigger == 'uploaded-model':
                model = self._get_model(uploaded_model)
                if model.get_max_rule_id() > 0:
                    return 1
                return 0
            if trigger == 'model-graph':
                if (clickdata is not None and clickdata['points'][0] is not None and 
                    'hovertext' in clickdata['points'][0]):
                    rule = clickdata['points'][0]['hovertext'].split('rule:')[1].split('<br>')[0]
                    if rule is not None:
                        return int(rule)
            if trigger == 'parallel-updated-rule-id':
                return parallel_rule_id
            if trigger == 'density-num-updated-rule-id':
                return num_rule_id
            if trigger == 'density-cats-updated-rule-id':
                return cats_rule_id
            elif trigger == 'removerule-updated-rule-id':
                return removerule_rule_id
            elif trigger == 'resetmodel-updated-rule-id':
                return resetmodel_rule_id
            raise PreventUpdate

        @app.callback(
            Output('model-store', 'data'),
            Input('parallel-updated-model', 'data'),
            Input('density-num-updated-model', 'data'),
            Input('density-cats-updated-model', 'data'),
            Input('removerule-updated-model', 'data'),
            Input('resetmodel-updated-model', 'data'),
            Input('uploaded-model', 'data'),
            State('model-store', 'data')
        )
        def store_model(parallel_update, num_update, cats_update, 
                        removerule_update, resetmodel_update, uploaded_model, 
                        model):
            trigger = self._get_callback_trigger()
            if trigger == 'parallel-updated-model':
                if parallel_update is not None: return parallel_update
            elif trigger == 'density-num-updated-model':
                if num_update is not None: return num_update
            elif trigger == 'density-cats-updated-model':
                if cats_update is not None: return cats_update
            elif trigger == 'removerule-updated-model':
                if removerule_update is not None: return removerule_update
            elif trigger == 'resetmodel-updated-model':
                if resetmodel_update is not None: return resetmodel_update
            elif trigger == 'uploaded-model':
                if uploaded_model is not None: return uploaded_model
            if model is None:
                return self.model.to_json()
            raise PreventUpdate

        @app.callback(
            Output('update-model-graph', 'data'),
            Output('model-description', 'children'),
            Output('model-yaml', 'children'),
            Output('model-code', 'children'),
            Output('update-model-performance', 'data'),
            Output('selected-rule-id', 'options'),
            Output('selected-rule-id', 'value'),
            Input('updated-rule-id', 'data'),
            Input('model-store', 'data')
        )
        def update_model(rule_id, model):
            model = self._get_model(model)
            rule_id_options = [{'label':str(ruleid), 'value':int(ruleid)} 
                    for ruleid in range(model._get_max_rule_id()+1)]
            
            rule_id = dash.no_update if rule_id is None else int(rule_id)
            return ("update_graph", 
                    f"```\n{model.describe()}```",
                    f"```yaml\n{model.to_yaml()}\n```",
                    f"```python\nfrom rule_estimator import *\n\nmodel = {model.to_code()[1:]}\n```",
                    "update_performance",
                    rule_id_options, rule_id)
        

        #########                                               DENSITY NUM RULE CALLBACK
        @app.callback(
            Output('density-num-updated-rule-id', 'data'),
            Output('density-num-updated-model', 'data'),
            Output('added-num-density-rule', 'data'),
            Input('density-num-split-button', 'n_clicks'),
            Input('density-num-predict-button', 'n_clicks'),
            Input('density-num-predict-all-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('density-num-ruletype', 'value'),
            State('density-col', 'value'),
            State('density-num-cutoff', 'value'),
            State('density-num-prediction', 'value'),
            State('model-store', 'data'),
        )
        def update_model_rule(split_clicks, predict_clicks, all_clicks, rule_id, append_or_replace, rule_type, 
                                col, cutoff, prediction, model):
            new_rule = None
            trigger = self._get_callback_trigger()
            if trigger == 'density-num-predict-button':
                if rule_type == 'lesser_than':
                    new_rule = LesserThan(col=col, cutoff=cutoff[1], prediction=int(prediction))
                elif rule_type == 'greater_than':
                    new_rule = GreaterThan(col=col, cutoff=cutoff[0], prediction=int(prediction))
                elif rule_type == 'range':
                    new_rule = RangeRule(col=col, min=cutoff[0], max=cutoff[1], prediction=int(prediction))
            elif trigger == 'density-num-split-button':
                if rule_type == 'lesser_than':
                    new_rule = LesserThanSplit(col=col, cutoff=cutoff[1])
                elif rule_type == 'greater_than':
                    new_rule = GreaterThanSplit(col=col, cutoff=cutoff[0])
                elif rule_type == 'range':
                    new_rule = RangeSplit(col=col, min=cutoff[0], max=cutoff[1])
            elif trigger == 'density-num-predict-all-button':
                new_rule = PredictionRule(prediction=int(prediction))
            
            if new_rule is not None:
                rule_id, model = self._change_model(model, rule_id, new_rule, append_or_replace)
                return rule_id, model.to_json(), "trigger"
            raise PreventUpdate

        #########                                               DENSITY CATS RULE CALLBACK
        @app.callback(
            Output('density-cats-updated-rule-id', 'data'),
            Output('density-cats-updated-model', 'data'),
            Output('added-cats-density-rule', 'data'),
            Input('density-cats-split-button', 'n_clicks'),
            Input('density-cats-predict-button', 'n_clicks'),
            Input('density-cats-predict-all-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('density-col', 'value'),
            State('density-cats-cats', 'value'),
            State('density-cats-prediction', 'value'),
            State('model-store', 'data'),
        )
        def update_model_rule(split_clicks, predict_clicks, all_clicks, rule_id, append_or_replace, col, cats, prediction, model):
            new_rule = None
            trigger = self._get_callback_trigger()
            if trigger == 'density-cats-split-button':
                new_rule = IsInSplit(col=col, cats=cats)
            elif trigger == 'density-cats-predict-button':
                new_rule = IsInRule(col=col, cats=cats, prediction=int(prediction))
            elif trigger == 'density-cats-predict-all-button':
                new_rule = PredictionRule(prediction=int(prediction))

            if new_rule is not None:
                rule_id, model = self._change_model(model, rule_id, new_rule, append_or_replace)
                return rule_id, model.to_json(), "trigger"
            raise PreventUpdate
        
        #########                                               PARALLEL RULE CALLBACK
        @app.callback(
            Output('parallel-updated-rule-id', 'data'),
            Output('parallel-updated-model', 'data'),
            Input('parallel-split-button', 'n_clicks'),
            Input('parallel-predict-button', 'n_clicks'),
            Input('parallel-predict-all-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('parallel-prediction', 'value'),
            State('parallel-plot', 'figure'),
            State('model-store', 'data'),
        )
        def update_model_parallel(split_clicks, predict_clicks, predict_all_clicks, 
                        rule_id, append_or_replace, prediction, fig, model):
            new_rule = None
            trigger = self._get_callback_trigger()
            if fig is not None:
                model = self._get_model(model)
                plot_data = fig['data'][0].get('dimensions', None)
                range_dict = {}
                for col_data in plot_data:
                    if col_data['label'] != 'y' and 'constraintrange' in col_data:
                        range_dict[col_data['label']] = self._process_constraintrange(
                            col_data['constraintrange'], col_data['ticktext'] if 'ticktext' in col_data else None)
                
                if trigger == 'parallel-split-button':
                    new_rule = MultiRangeSplit(range_dict)
                elif trigger == 'parallel-predict-button':
                    new_rule = MultiRange(range_dict, prediction=int(prediction))
                elif trigger == 'parallel-predict-all-button':
                    new_rule = PredictionRule(prediction=int(prediction))
                else:
                    raise PreventUpdate

                rule_id, model = self._change_model(model, rule_id, new_rule, append_or_replace)
                return rule_id, model.to_json()

            raise PreventUpdate

        @app.callback(
            Output('removerule-updated-rule-id', 'data'),
            Output('removerule-updated-model', 'data'),
            Input('remove-rule-confirm', 'submit_n_clicks'),
            State('selected-rule-id', 'value'),
            State('model-store', 'data'),
        )
        def remove_rule(n_clicks, rule_id, model):
            if n_clicks is not None:
                model = self._get_model(model)
                model.remove_rule(rule_id)
                return min(rule_id, model._get_max_rule_id()), model.to_json()
            raise PreventUpdate

        @app.callback(
            Output('resetmodel-updated-rule-id', 'data'),
            Output('resetmodel-updated-model', 'data'),
            Input('reset-model-confirm', 'submit_n_clicks'),
            State('selected-rule-id', 'value'),
        )
        def reset_model(n_clicks, rule_id):
            if n_clicks is not None:
                return 0, self.model.to_json()
            raise PreventUpdate
 
        @app.callback(
            Output('density-num-div', 'style'),
            Output('density-cats-div', 'style'),
            Input('density-col', 'value'),
        )
        def update_density_hidden_divs(col):
            if col is not None:
                if col in self.cats:
                    return dict(display="none"), {}
                else:
                    return {}, dict(display="none")
            else:
                if self.X.columns[0] in self.cats:
                    return dict(display="none"), {}
                else:
                    return {}, dict(display="none")

        @app.callback(
            Output('density-col', 'value'),
            Output('rule-tabs', 'value'),
            Input('selected-rule-id', 'value'),
            Input('append-or-replace', 'value'),
            Input('density-col-suggest-button', 'n_clicks'),
            State('train-or-val', 'value'),
            State('model-store', 'data'),
        )
        def update_model_node(rule_id, append_or_replace, n_clicks, train_or_val, model):
            trigger = self._get_callback_trigger()
            if trigger == 'density-col-suggest-button':
                model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace) 
                return model._sort_cols_by_gini_reduction(X, y)[0], dash.no_update
            if append_or_replace == 'replace' and rule_id is not None:
                model = self._get_model(model)
                rule = model.get_rule(rule_id)
                if isinstance(rule, IsInRule):
                    return rule.col, "density-tab"
                elif isinstance(rule, IsInSplit):
                    return rule.col, "density-tab"
                elif isinstance(rule, LesserThan):
                    return rule.col, "density-tab"
                elif isinstance(rule, GreaterThan):
                    return rule.col, "density-tab"
                elif isinstance(rule, LesserThanSplit):
                    return rule.col, "density-tab"
                elif isinstance(rule, GreaterThanSplit):
                    return rule.col, "density-tab"
                elif isinstance(rule, MultiRange):
                    return dash.no_update, "density-tab"
                elif isinstance(rule, RangeRule):
                    return dash.no_update, "density-tab"
                elif isinstance(rule, RangeRule):
                    return dash.no_update, "density-tab"
                elif isinstance(rule, MultiRangeSplit):
                    return dash.no_update, "parallel-tab"
            raise PreventUpdate

        @app.callback(
            Output('density-num-prediction', 'value'),
            Output('density-num-pie-all-label', 'children'),
            Output('density-num-pie-all', 'figure'),
            Output('density-num-pie-selected-label', 'children'),
            Output('density-num-pie-selected', 'figure'),
            Output('density-num-pie-not-selected-label', 'children'),
            Output('density-num-pie-not-selected', 'figure'),
            Input('density-num-cutoff', 'value'),
            Input('selected-rule-id', 'value'),
            Input('density-num-ruletype', 'value'),
            Input('train-or-val', 'value'),
            Input('append-or-replace', 'value'),
            State('density-col', 'value'),
            State('model-store', 'data'),   
        )
        def update_density_num_pies(cutoff, rule_id, rule_type, train_or_val, append_or_replace, col, model):
            if col is not None and col in self.non_cats and cutoff is not None:
                model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace) 
                pie_size = 80

                if rule_type == 'lesser_than':
                    rule = LesserThanSplit(col, cutoff[1])
                elif rule_type == 'greater_than':
                    rule = GreaterThanSplit(col, cutoff[0])
                elif rule_type == 'range':
                    rule = RangeSplit(col, cutoff[0], cutoff[1])
                else:
                    raise ValueError("rule_type should be either lesser_than or greater_than!")

                X_rule, y_rule = X[rule.__rule__(X)], y[rule.__rule__(X)]

                pie_all =  plot_label_pie(model, X, y, size=pie_size)
                pie_selection = plot_label_pie(model, X_rule, y_rule, size=pie_size)
                pie_non_selection = plot_label_pie(model, X[~rule.__rule__(X)], y[~rule.__rule__(X)], size=pie_size)
                
                trigger = self._get_callback_trigger()
                prediction = None
                if append_or_replace=='replace' and trigger in ['selected-rule-id', 'append-or-replace']:
                    rule = model.get_rule(rule_id)
                    if (isinstance(rule, LesserThan) or isinstance(rule, LesserThanSplit) or 
                        isinstance(rule, GreaterThan) or isinstance(rule, GreaterThanSplit) or
                        isinstance(rule, RangeRule) or isinstance(rule, RangeSplit)):
                        prediction = rule.prediction

                if prediction is None and not X_rule.empty:
                    prediction = y_rule.value_counts().index[0]
                elif prediction is None and not X.empty:
                    prediction = y.value_counts().index[0]
                else:
                    prediction = 0       
             
                return (str(prediction),
                        f"All ({len(X)})", pie_all, 
                        f"Selected ({len(X_rule)})", pie_selection, 
                        f"Not Selected ({len(X)-len(X_rule)})", pie_non_selection
                    )
            raise PreventUpdate

        @app.callback(
            Output('density-cats-prediction', 'value'),
            Output('density-cats-pie-all-label', 'children'),
            Output('density-cats-pie-all', 'figure'),
            Output('density-cats-pie-selected-label', 'children'),
            Output('density-cats-pie-selected', 'figure'),
            Output('density-cats-pie-not-selected-label', 'children'),
            Output('density-cats-pie-not-selected', 'figure'),
            Input('density-cats-cats', 'value'),
            Input('selected-rule-id', 'value'),
            Input('train-or-val', 'value'),
            Input('append-or-replace', 'value'),
            State('density-col', 'value'),
            State('model-store', 'data'),   
        )
        def update_density_cats_pies(cats, rule_id,  train_or_val, append_or_replace, col, model):
            if col is not None:
                model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace) 
                pie_size = 80

                rule = IsInSplit(col, cats)
                X_rule, y_rule = X[rule.__rule__(X)], y[rule.__rule__(X)]

                pie_all =  plot_label_pie(model, X, y, size=pie_size)
                pie_selection = plot_label_pie(model, X_rule, y_rule, size=pie_size)
                pie_non_selection = plot_label_pie(model, X[~rule.__rule__(X)], y[~rule.__rule__(X)], size=pie_size)

                trigger = self._get_callback_trigger()
                prediction = None
                if append_or_replace=='replace' and trigger in ['selected-rule-id', 'append-or-replace']:
                    rule = model.get_rule(rule_id)
                    if isinstance(rule, IsInRule):
                        prediction = rule.prediction

                if prediction is None and not X_rule.empty:
                    prediction = y_rule.value_counts().index[0]
                elif prediction is None and not X.empty:
                    prediction = y.value_counts().index[0]
                else:
                    prediction = 0

                return (str(prediction),
                        f"All ({len(X)})", pie_all, 
                        f"Selected ({len(X_rule)})", pie_selection, 
                        f"Not Selected ({len(X)-len(X_rule)})", pie_non_selection
                    )
            raise PreventUpdate

        @app.callback(
            Output('density-cats-plot', 'figure'),
            Output('density-num-plot', 'figure'),
            Input('density-col', 'value'),
            Input('selected-rule-id', 'value'),
            Input('train-or-val', 'value'),
            Input('append-or-replace', 'value'),
            Input('density-num-cutoff', 'value'),
            Input('density-cats-cats', 'value'),
            Input('density-cats-percentage', 'value'),
            State('model-store', 'data'),
            
        )
        def update_density_plot(col, rule_id, train_or_val, append_or_replace, cutoff, cats, percentage, model):
            if col is not None:
                model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace) 
                after = self._infer_after(append_or_replace)
                if col in self.cats:
                    #percentage = (percentage=='percentage')
                    fig = plot_cats_density(model, X, y, col, rule_id=rule_id, after=after, 
                                labels=self.labels, percentage=bool(percentage), highlights=cats)
                    return fig, dash.no_update

                elif col in self.non_cats:
                    fig = plot_density(model, X, y, col, rule_id=rule_id, after=after, 
                                labels=self.labels, cutoff=cutoff)
                    return dash.no_update, fig
            raise PreventUpdate

        @app.callback(
            Output('density-col-suggest-button', 'n_clicks'),
            Input('added-num-density-rule', 'data'),
            Input('added-cats-density-rule', 'data'),
            State('density-col-suggest-button', 'n_clicks'),
        )
        def trigger_new_suggested_col(num_trigger, cats_trigger, old_clicks):
            if old_clicks is not None:
                return old_clicks+1
            return 1

        @app.callback(
            Output('density-col', 'options'),
            Input('density-sort', 'value'),
            Input('selected-rule-id', 'value'),
            Input('train-or-val', 'value'),
            Input('append-or-replace', 'value'),
            State('model-store', 'data'),
            State('density-col', 'options'),
        )
        def update_density_col(sort, rule_id, train_or_val, append_or_replace, model, old_options):
            if sort=='dataframe':
                return [dict(label=col, value=col) for col in self.X.columns]
            elif sort == 'alphabet':
                return [dict(label=col, value=col) for col in sorted(self.X.columns.tolist())]
            elif sort == 'histogram overlap':
                model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace)
                return [dict(label=col, value=col) for col in model._sort_cols_by_histogram_overlap(X, y)]
            elif sort == 'gini reduction':
                model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace)
                return [dict(label=col, value=col) for col in model._sort_cols_by_gini_reduction(X, y)]
            else:
                raise ValueError(f"Wrong sort value: {sort}!")
            raise PreventUpdate

        @app.callback(
            Output('density-cats-cats', 'value'),
            Output('density-num-cutoff', 'value'),
            Output('density-num-ruletype', 'value'),
            Input('density-cats-plot', 'clickData'),
            Input('density-col', 'value'),
            Input('selected-rule-id', 'value'),
            Input('train-or-val', 'value'),
            Input('append-or-replace', 'value'),
            Input('density-num-suggest-button', 'n_clicks'),
            Input('density-cats-suggest-button', 'n_clicks'),
            Input('density-cats-invert-button', 'n_clicks'),
            Input('density-num-ruletype', 'value'), 
            Input('density-num-cutoff', 'value'),
            State('model-store', 'data'),
            State('density-cats-cats', 'value'),
            State('density-num-cutoff', 'min'),
            State('density-num-cutoff', 'max'),
            State('density-cats-cats', 'options'),
        )
        def check_cats_clicks(clickdata, col, rule_id, train_or_val, append_or_replace,
                                num_suggest_n_clicks, cats_suggest_n_clicks, invert_n_clicks, num_ruletype,
                                old_cutoff, model, old_cats,  cutoff_min, cutoff_max, cats_options):
            trigger = self._get_callback_trigger()
            
            if trigger == 'density-cats-invert-button':
                new_cats = [cat['value'] for cat in cats_options if cat['value'] not in old_cats]
                return new_cats, dash.no_update, dash.no_update

            if append_or_replace=='replace' and trigger in ['selected-rule-id', 'append-or-replace']:
                model = self._get_model(model)
                rule = model.get_rule(rule_id)
                if isinstance(rule, IsInRule) or isinstance(rule, IsInSplit):
                    return rule.cats, dash.no_update, dash.no_update
                if isinstance(rule, GreaterThan) or isinstance(rule, GreaterThanSplit):
                    return dash.no_update, [rule.cutoff, cutoff_max], "greater_than"
                if isinstance(rule, LesserThan) or isinstance(rule, LesserThanSplit):
                    return dash.no_update, [cutoff_min, rule.cutoff], "lesser_than"
                if isinstance(rule, RangeRule) or isinstance(rule, RangeSplit):
                    return dash.no_update, [rule.min, rule.max], "range"
            if  trigger == 'density-cats-plot':
                clicked_cat = clickdata['points'][0]['x']
                if old_cats is None:
                    return [clicked_cat], dash.no_update, dash.no_update
                elif clicked_cat in old_cats:
                    return [cat for cat in old_cats if cat != clicked_cat], dash.no_update, dash.no_update
                else:
                    old_cats.append(clicked_cat)
                    return old_cats, dash.no_update, dash.no_update
            
            model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace) 
            
            if trigger == 'density-num-ruletype':
                if num_ruletype == 'greater_than':
                    if old_cutoff[0] == X[col].min():
                        return dash.no_update, [old_cutoff[1], X[col].max()], dash.no_update
                    return dash.no_update, [old_cutoff[0], X[col].max()], dash.no_update
                if num_ruletype == 'lesser_than':
                    if old_cutoff[1] == X[col].max():
                        return dash.no_update, [X[col].min(), old_cutoff[0]], dash.no_update
                    return dash.no_update, [X[col].min(), old_cutoff[1]], dash.no_update
                return dash.no_update, old_cutoff, dash.no_update
            
            if trigger == 'density-num-cutoff':
                if num_ruletype == 'lesser_than' and old_cutoff[0] != X[col].min():
                    return dash.no_update, [X[col].min(), old_cutoff[1]], dash.no_update
                if num_ruletype == 'greater_than' and old_cutoff[1] != X[col].max():
                    return dash.no_update, [old_cutoff[0], X[col].max()], dash.no_update
                raise PreventUpdate

            if not X.empty:
                if col in self.cats:
                    cat, gini, single_cat = model.suggest_split(X, y, col)
                    if single_cat:
                        return [cat], dash.no_update, dash.no_update
                    elif cats_options:
                        return [cat_col['value'] for cat_col in cats_options if cat_col['value'] != cat], dash.no_update, dash.no_update
                    else:
                        [cat_col for cat_col in X[col].unique() if cat_col != cat], dash.no_update, dash.no_update
                elif col in self.non_cats:
                    cutoff, gini, lesser_than = model.suggest_split(X, y, col)
                    cutoff = self._round_cutoff(cutoff, X[col].min(), X[col].max())
                    if lesser_than:
                        return dash.no_update, [X[col].min(), cutoff], "lesser_than"
                    else:
                        return dash.no_update, [cutoff, X[col].max()], "greater_than"
            raise PreventUpdate

        @app.callback(
            Output('density-num-suggest-button', 'n_clicks'),
            Output('density-cats-suggest-button', 'n_clicks'),
            Input('density-col', 'value'),
            State('density-num-suggest-button', 'n_clicks'),
            State('density-cats-suggest-button', 'n_clicks'),
        )
        def trigger_suggest_buttons_on_col(col, num_clicks, cats_clicks):
            if col in self.cats:
                return dash.no_update, cats_clicks+1 if cats_clicks else 1
            elif col in self.non_cats:
                return num_clicks+1 if num_clicks else 1, dash.no_update
            raise PreventUpdate

        @app.callback(
            Output('density-num-cutoff', 'min'),
            Output('density-num-cutoff', 'max'),
            Output('density-num-cutoff', 'step'),
            Output('density-cats-cats', 'options'),
            Input('density-col', 'value'),
            Input('selected-rule-id', 'value'),
            Input('train-or-val', 'value'),
            Input('append-or-replace', 'value'),
            State('model-store', 'data'),
        )
        def update_density_plot(col, rule_id, train_or_val, append_or_replace, model): 
            if col is not None:
                model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace)
                if col in self.cats:
                    cats_options = [dict(label=cat, value=cat) for cat in X[col].unique()]
                    return dash.no_update, dash.no_update, dash.no_update, cats_options
                elif col in self.non_cats:
                    min_val, max_val = X[col].min(), X[col].max()
                    return min_val, max_val, self._get_stepsize(min_val, max_val), dash.no_update
            raise PreventUpdate

        @app.callback(
            Output('parallel-plot', 'figure'),
            Input('selected-rule-id', 'value'),
            Input('parallel-cols', 'value'),
            Input('append-or-replace', 'value'),
            Input('parallel-sort', 'value'),
            Input('train-or-val', 'value'),
            State('parallel-plot', 'figure'),
            State('model-store', 'data'),
        )
        def return_parallel_plot(rule_id, cols, append_or_replace, sort, train_or_val, old_fig, model):
            model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace)
            after = self._infer_after(append_or_replace)

            if sort=='dataframe':
                cols = [col for col in self.X.columns if col in cols]
            elif sort == 'alphabet':
                cols = sorted(cols)
            elif sort == 'histogram overlap':
                cols = model._sort_cols_by_histogram_overlap(X, y, cols, reverse=True)
            elif sort == 'gini reduction':
                cols = model._sort_cols_by_gini_reduction(X, y, cols, reverse=True)
            else:
                raise ValueError(f"Wrong sort value: {sort}!")
            
            fig = plot_parallel_coordinates(model, X, y, rule_id, cols=cols, labels=self.labels, after=after, 
                                                ymin=self.y.min(), ymax=self.y.max())
            fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
            
            if fig['data'] and 'dimensions' in fig['data'][0]:
                trigger = self._get_callback_trigger()
                if append_or_replace=='replace' and trigger in ['selected-rule-id', 'append-or-replace']:
                    rule = model.get_rule(rule_id)

                    if (isinstance(rule, MultiRange) or 
                        isinstance(rule, MultiRangeSplit)):
                        plot_data = fig['data'][0]['dimensions']
                        for col, ranges in rule.range_dict.items():
                            for dimension in fig['data'][0]['dimensions']:
                                if dimension['label'] == col:
                                    if isinstance(ranges[0], str) and 'ticktext' in dimension:
                                        dimension['constraintrange'] = self._cats_to_range(ranges, dimension['ticktext'])
                                    else:
                                        dimension['constraintrange'] = ranges
                if trigger == 'train-or-val' and old_fig['data'] and 'dimensions' in old_fig['data'][0]:
                    fig['data'][0]['dimensions'] = old_fig['data'][0]['dimensions']
            return fig
        
        @app.callback(
            Output('parallel-prediction', 'value'),
            Output('parallel-pie-all-label', 'children'),
            Output('parallel-pie-all', 'figure'),
            Output('parallel-pie-selection-label', 'children'),
            Output('parallel-pie-selection', 'figure'),
            Output('parallel-pie-non-selection-label', 'children'),
            Output('parallel-pie-non-selection', 'figure'),
            Input('parallel-plot', 'restyleData'),
            Input('selected-rule-id', 'value'),
            Input('append-or-replace', 'value'),
            Input('train-or-val', 'value'),
            Input('parallel-plot', 'figure'),  
            State('model-store', 'data'),
        )
        def update_parallel_prediction(restyle, rule_id, append_or_replace, train_or_val, fig, model):
            if fig is not None and fig['data']:
                model, X, y = self._get_model_X_y(model, train_or_val, rule_id, append_or_replace)
                range_dict = self._get_range_dict_from_parallel_plot(fig)
                rule = MultiRangeSplit(range_dict)
                
                after = self._infer_after(append_or_replace)
                pie_size = 50

                pie_all = plot_label_pie(model, self.X, self.y, rule_id=rule_id, after=after, size=pie_size)

                X_rule, y_rule = X[rule.__rule__(X)], y[rule.__rule__(X)]
                pie_selection = plot_label_pie(model, X_rule, y_rule, size=pie_size)
                pie_non_selection = plot_label_pie(model, X[~rule.__rule__(X)], y[~rule.__rule__(X)], size=pie_size)

                if not X_rule.empty:
                    prediction = y_rule.value_counts().index[0]
                elif not X.empty:
                    prediction = y.value_counts().index[0]
                else:
                    prediction = 0

                trigger = self._get_callback_trigger()
                if append_or_replace=='replace' and trigger in ['selected-rule-id','append-or-replace']:
                    rule = model.get_rule(rule_id)
                    if isinstance(rule, PredictionRule):
                        prediction = rule.prediction
                    
                return (str(prediction), 
                        f"All ({len(X)})", pie_all, 
                        f"Selected ({len(X_rule)})", pie_selection, 
                        f"Not selected ({len(X)-len(X_rule)})", pie_non_selection
                     )          
            raise PreventUpdate

        @app.callback(
            Output('model-performance', 'children'),
            Input('update-model-performance', 'data'),
            Input('train-or-val', 'value'), 
            State('model-store', 'data'), 
        )
        def update_performance_metrics(update, train_or_val, model):
            if update:
                model = self._get_model(model)
                X, y = self._get_X_y(train_or_val) 
                return dbc.Table.from_dataframe(
                                model.score_rules(X, y)
                                    .assign(coverage = lambda df:df['coverage'].apply(lambda x: f"{100*x:.2f}%"))
                                    .assign(accuracy = lambda df:df['accuracy'].apply(lambda x: f"{100*x:.2f}%"))
                )
            raise PreventUpdate

        @app.callback(
            Output('model-performance-confmat', 'figure'),
            Input('update-model-performance', 'data'),
            Input('train-or-val', 'value'),  
            Input('selected-rule-id', 'value'),
            Input('model-performance-select', 'value'),
            State('model-store', 'data'), 
        )
        def update_performance_confmat(update, train_or_val, rule_id, model_or_rule, model):
            if update:
                if model_or_rule == 'rule':
                    model, X, y = self._get_model_X_y(model, train_or_val, rule_id=rule_id)
                    return plot_confusion_matrix(model, X, y, labels=self.labels, rule_id=rule_id, rule_only=True)
                else:
                    model, X, y = self._get_model_X_y(model, train_or_val)
                    return plot_confusion_matrix(model, X, y, labels=self.labels)

            raise PreventUpdate

        @app.callback(
            Output('model-performance-metrics', 'children'),
            Input('update-model-performance', 'data'),
            Input('train-or-val', 'value'),  
            Input('selected-rule-id', 'value'),
            Input('model-performance-select', 'value'),
            State('model-store', 'data'), 
        )
        def update_performance_metrics(update, train_or_val, rule_id, model_or_rule, model):
            if update:
                if model_or_rule == 'rule':
                    model, X, y = self._get_model_X_y(model, train_or_val, rule_id=rule_id)
                    return dbc.Table.from_dataframe(get_metrics_df(model, X, y, rule_id=rule_id, rule_only=True))
                else:
                    model, X, y = self._get_model_X_y(model, train_or_val)
                    return dbc.Table.from_dataframe(get_metrics_df(model, X, y))


            raise PreventUpdate

        @app.callback(
            Output('model-performance-coverage', 'children'),
            Input('update-model-performance', 'data'),
            Input('train-or-val', 'value'), 
            Input('selected-rule-id', 'value'),
            Input('model-performance-select', 'value'),
            State('model-store', 'data'), 
        )
        def update_performance_coverage(update, train_or_val, rule_id, model_or_rule, model):
            if update:
                if model_or_rule == 'rule':
                    model, X, y = self._get_model_X_y(model, train_or_val, rule_id=rule_id)
                    return dbc.Table.from_dataframe(get_coverage_df(model, X, y, rule_id=rule_id, rule_only=True))
                else:
                    model, X, y = self._get_model_X_y(model, train_or_val)
                    return dbc.Table.from_dataframe(get_coverage_df(model, X, y))
            raise PreventUpdate

        @app.callback(
            Output('model-graph', 'figure'),
            Input('update-model-graph', 'data'),
            Input('train-or-val', 'value'),
            Input('model-graph-color-scale', 'value'),
            Input('selected-rule-id', 'value'),
            Input('model-graph-scatter-text', 'value'),
            State('model-store', 'data'), 
        )
        def update_model_graph(update, train_or_val, color_scale, highlight_id, scatter_text, model):
            if update:
                model = self._get_model(model)
                X, y = self._get_X_y(train_or_val) 
                if scatter_text=='coverage':
                    def format_cov(row):
                        return f"coverage={100*row.coverage:.2f}% ({row.n_outputs}/{row.n_inputs})"

                    scatter_text = model.score_rules(X, y).drop_duplicates(subset=['rule_id'])[['coverage', 'n_inputs', 'n_outputs']].apply(
                                lambda row: format_cov(row), axis=1).tolist()
                elif scatter_text=='accuracy':
                    scatter_text = model.score_rules(X, y).drop_duplicates(subset=['rule_id'])['accuracy'].apply(lambda x: f"accuracy: {100*x:.2f}%").tolist()
                return plot_model_graph(model, X, y, color_scale=color_scale, highlight_id=highlight_id, scatter_text=scatter_text)
            raise PreventUpdate

        @app.callback(
            Output('uploaded-model', 'data'),
            Output('upload-div', 'style'),
            Input('upload-model', 'contents'),
            Input('upload-button', 'n_clicks'),
            State('upload-model', 'filename'),
            State('upload-div', 'style'),
        )
        def update_output(contents, n_clicks, filename, style):
            trigger = self._get_callback_trigger()
            if trigger == 'upload-button':
                if not style:
                    return dash.no_update, dict(display="none")
                return dash.no_update, {}

            if contents is not None:
                import base64
                import io
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)

                if filename.endswith(".yaml"):
                    try:
                        model = RuleClassifier.from_yaml(config=decoded.decode('utf-8'))
                        return model.to_json(), dict(display="none")
                    except:
                        pass

                elif filename.endswith(".pkl"):
                    import pickle
                    try:
                        model = pickle.loads(decoded)
                        if isinstance(model, RuleClassifier):
                            return model.to_json(), dict(display="none")
                    except:
                        pass
            raise PreventUpdate

        @app.callback(
            Output('instructions-modal', 'is_open'),
            Input('instructions-open-button', 'n_clicks'),
            Input('instructions-close-button', 'n_clicks')
        )
        def toggle_modal(open_clicks, close_clicks):
            trigger = self._get_callback_trigger()
            if trigger == 'instructions-open-button':
                return True
            elif trigger == 'instructions-close-button':
                return False
            raise PreventUpdate

    def run(self, debug=False):
        self.app.run_server(port=self.port)#, use_reloader=False, debug=debug)

                    
        
        