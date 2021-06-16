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
            
        self.port = port
        
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
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
        if json_model is not None:
            return RuleClassifier.from_json(json_model)
        else:
            return self.model

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

    def _get_model_X_y(self, model, train_or_val='train', rule_id=0, after=False):
        """returns a (model, X, y) tuple

        Args:
            train_or_val: return 'train' data or 'val' data
            rule_id: return data for rule rule_id
            after: If True return data that has not been assigned
                a prediction after rule rule_id. Can also be a string
                in {'append', 'replace}, in which case it will get inferred.

        """
        if not isinstance(model, RuleClassifier):
            model = self._get_model(model)            

        X, y = self._get_X_y(train_or_val)
        if not isinstance(after, bool) and after in {'append', 'replace'}:
            after = self._infer_after(after)
        else:
            raise ValueError(f"After should either be a bool or in 'append', 'replace',"
                             f" but you passed {after}!")
        X, y = model.get_rule_input(rule_id, self.X, self.y, after)
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
                ],
                brand="RuleClassifierDashboard",
                brand_href="https://github.com/oegedijk/rule_estimator/",
                color="primary",
                dark=True,
            ),

            # helper storages as a workaround dash limitation that output can
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

            dcc.Store(id='update-model-performance'),
            dcc.Store(id='update-model-graph'),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Label("Selected rule:", id='selected-rule-id-label', 
                                                    html_for='selected-rule-id', className="mr-2"),
                                        dcc.Dropdown(id='selected-rule-id', options=[
                                            {'label':str(ruleid), 'value':int(ruleid)} 
                                                for ruleid in range(self.model._get_max_rule_id()+1)],
                                                value=0, clearable=False, style=dict(width=200)),
                                        dbc.Tooltip("You can either select a rule id here or by clicking in the model graph. "
                                                    "The parallel plot will show either all the unlabeled data after this rule "
                                                    "(when you select 'append'), or all the unlabeled data coming into this rule "
                                                    "(when you select 'replace')", target='selected-rule-id-label'),
                                    ], row=True, className="mr-3"),
                                ]),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Select(
                                            options=[{'label':'Append rule after (show data out)', 'value':'append'},
                                                    {'label':'Replace rule (show data in)', 'value':'replace'}],
                                            value='append',
                                            id='append-or-replace'),
                                        dbc.Tooltip("When you select to 'append' a rule, the Parallel Plot will show "
                                                    "all the data that is still unlabeled after the selected rule. "
                                                    "When you add a rule it will appended to a CaseWhen block. "   
                                                    "When you select 'replace' the Parallel Plot will display the data going into "
                                                    "this rule. When you add a rule, it will replace the existing rule.",
                                                        target='append-or-replace'),
                                    ], className="mr-3"),
                                ]),
                                html.Div([
                                    dbc.Col([
                                        dbc.FormGroup([
                                            dbc.Select(
                                                    options=[{'label':'Train data', 'value':'train'},
                                                            {'label':'Validation data', 'value':'val'}],
                                                    value='train',
                                                    id='train-or-val'),
                                            ], className="mr-3"),
                                    ]),
                                ], style=dict(display="none") if self.X_val is None else dict()),     
                            ], form=True),
                        ]),
                    ], style=dict(marginBottom=10, marginTop=10)),
                    dcc.Tabs(id='rule-tabs', value='parallel-tab', 
                        children=[
                            dcc.Tab(label="Multi Feature Rules", id='parallel-tab', value='parallel-tab', children=[
                                dbc.Card([
                                    dbc.CardHeader([
                                        html.H3("Parallel Feature Plot", className="card-title"),
                                        html.H6("Select (multiple) feature ranges to generate a split or add a prediction rule", className="card-subtitle"),   
                                    ]),
                                    dbc.CardBody([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.FormGroup([
                                                        dbc.Label("Features", html_for='parallel-cols', id='parallel-cols-label'),
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
                                                    dbc.Label("Sort features", html_for='sort-by-overlap'),
                                                    dbc.Checklist(
                                                        options=[{"label":  "sort", "value": True}],
                                                        value=[True],
                                                        id='sort-by-overlap',
                                                        inline=True,
                                                        switch=True,
                                                        style=dict(height=40, width=100, horizontalAlign = 'right'),
                                                    ),
                                                    dbc.Tooltip("Sort the features in the plot from least histogram overlap "
                                                                "(feature distributions look the most different for different "
                                                                "values of y) on the right, to highest histogram overlap on the left.", 
                                                                target='sort-by-overlap'),
                                                ])
                                            ], md=2),
                                        ], form=True),
                                        dbc.Row([
                                            dbc.Col([
                                                dcc.Graph(id='parallel-plot'),
                                                html.Div(id='parallel-description'), 
                                            ])
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
                                                                    color="primary", size="m", style=dict(width=400)),
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
                                                            bs_size="sm"),
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
                                    dbc.CardFooter([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("All data", id='parallel-pie-all-label'),
                                                dbc.Tooltip("Label distribution for all observations", target='parallel-pie-all-label'),
                                                dcc.Graph(id='parallel-pie-all', config=dict(modeBarButtons=[[]], displaylogo=False)),  
                                            ]),
                                            dbc.Col([
                                                dbc.Label("This data", id='parallel-pie-data-label'),
                                                dbc.Tooltip("Label distribution for all observations in the parallel plot above.", 
                                                        target='parallel-pie-data-label'),
                                                dcc.Graph(id='parallel-pie-data', config=dict(modeBarButtons=[[]], displaylogo=False)),   
                                            ]),
                                            dbc.Col([
                                                dbc.Label("Selected", id='parallel-pie-selection-label'),
                                                dbc.Tooltip("Label distribution for all the feature ranges selected above", 
                                                        target='pie-parallel-selection-label'),
                                                dcc.Graph(id='parallel-pie-selection', config=dict(modeBarButtons=[[]], displaylogo=False)),   
                                            ]),
                                            dbc.Col([
                                                dbc.Label("Not selected", id='parallel-pie-non-selection-label'),
                                                dbc.Tooltip("Label distribution for all the feature ranges not selected above", 
                                                        target='parallel-pie-non-selection-label'),
                                                dcc.Graph(id='parallel-pie-non-selection', config=dict(modeBarButtons=[[]], displaylogo=False)),
                                            ]),
                                        ]),
                                    ]),
                                ]),

                            ]),
                            dcc.Tab(label="Single Feature Rules", id='density-tab', value='density-tab', children=[
                                dbc.Card([
                                    dbc.CardHeader([
                                        html.H3("Density Plot", className='card-title')
                                    ]),
                                    dbc.CardBody([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Column"),
                                                dcc.Dropdown(id='density-col', 
                                                            options=[{'label':col, 'value':col} for col in self.X.columns], 
                                                            value=self.X.columns.tolist()[0],
                                                            clearable=False),
                                            ]),
                                        ]),
                                        html.Div([
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Graph(id='density-num-plot'),
                                                    
                                                    
                                                ], md=10),
                                                dbc.Col([
                                                    dbc.Label("All"),
                                                    dcc.Graph(id='density-num-pie'),
                                                    dbc.Label("Selected"),
                                                    dcc.Graph(id='density-num-pie-include'),
                                                    dbc.Label("Not Selected"),
                                                    dcc.Graph(id='density-num-pie-exclude'),
                                                ], md=2),
                                            ]),
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Slider(id='density-num-cutoff', min=0, max=10,
                                                        tooltip=dict(always_visible=True)),
                                                ], md=10),
                                                dbc.Col([
                                                    dbc.Select(
                                                            options=[{'label':'Select lesser than', 'value':'lesser_than'},
                                                                        {'label':'Select greater than', 'value':'greater_than'}],
                                                            value='lesser_than',
                                                            id='density-num-ruletype',
                                                            #style=dict(height='20px'),
                                                            bs_size="sm",
                                                            ),
                                                ])
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
                                                                color="primary", size="m", style=dict(width=400)),
                                                    ]),
                                                ], md=6),
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        html.Div([
                                                            dbc.Select(id='density-num-prediction', options=[
                                                                    {'label':f"{y}. {label}", 'value':str(y)} for y, label in enumerate(self.labels)],
                                                                        value=str(len(self.labels)-1), bs_size="sm"),
                                                            dbc.Button("Predict Selected", id='density-num-predict-button', color="primary",
                                                                        size="m", style=dict(marginLeft=10, width=400)),
                                                            dbc.Button("Predict All", id='density-num-predict-all-button', color="primary",
                                                                        size="m", style=dict(marginLeft=10, width=400)),
                                                        ], style={'width': '100%', 'display': 'flex', 'align-items': 'right', 'justify-content': 'right'}),
                                                    ], row=True),
                                                ], md=6),
                                            ], form=True),
                                        ], id='density-num-div'),
                                        html.Div([
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Graph(id='density-cats-plot'),
                                                    dbc.Row([
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                dbc.Label("Categories to include:", html_for='density-cats-cats'),
                                                                dcc.Dropdown(id='density-cats-cats', value=[], multi=True,
                                                                    style=dict(marginBottom=20)),
                                                            ]), 
                                                        ], md=10),
                                                        dbc.Col([
                                                            dbc.FormGroup([
                                                                dbc.Label("percentage:", html_for='density-cats-cats'),
                                                                dbc.Select('density-cats-percentage', 
                                                                        options=[dict(label='absolute', value='absolute'),
                                                                                dict(label='percentage', value='percentage')],
                                                                                value='absolute'),
                                                                
                                                            ]), 
                                                        ], md=2)
                                                    ], form=True), 
                                                ], md=10),
                                                dbc.Col([
                                                    dbc.Label("All"),
                                                    dcc.Graph(id='density-cats-pie'),
                                                    dbc.Label("Selected"),
                                                    dcc.Graph(id='density-cats-pie-include'),
                                                    dbc.Label("Not selected"),
                                                    dcc.Graph(id='density-cats-pie-exclude'),
                                                ], md=2),
                                            ]),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Hr()
                                                ])
                                            ]),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        dbc.Button("Split", id='density-cats-split-button', 
                                                            color="primary", size="m", style=dict(width=100)),
                                                    ]),
                                                ], md=6),
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        html.Div([
                                                            dbc.Select(id='density-cats-prediction', options=[
                                                                    {'label':f"{y}. {label}", 'value':str(y)} for y, label in enumerate(self.labels)],
                                                                    value=str(len(self.labels)-1), #clearable=False, 
                                                                    bs_size="sm"),
                                                            dbc.Button("Predict", id='density-cats-predict-button', color="primary",
                                                                        size="m", style=dict(marginLeft=10, width=100)),
                                                            dbc.Button("All", id='density-cats-predict-all-button', color="primary",
                                                                        size="m", style=dict(marginLeft=10, width=100)),
                                                        ], style = {'width': '100%', 'display': 'flex', 'align-items': 'right', 'justify-content': 'right'})    
                                                    ], row=True),
                                                ], md=6),
                                            ], form=True),
                                        ], id='density-cats-div'),
                                    ]),
                                ]),
                            ]),
                        ]),
                    
                ]),
            ], style=dict(marginBottom=30)),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H3("Model", className='card-title'),
                            html.H6("You can select a rule by clicking on it in the Graph", className='card-subtitle'),
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
                                                    ], row=True), 
                                                ], md=3),
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        dbc.Label("Display: ", html_for='model-graph-scatter-text', className="mr-2"),
                                                        dbc.Select(id='model-graph-scatter-text',
                                                            options=[dict(label=o, value=o) for o in ['name', 'description', 'coverage', 'accuracy']],
                                                            value='description', 
                                                            bs_size="sm",style=dict(width=130)),
                                                    ], row=True), 
                                                ], md=3),
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        html.Div([
                                                            dcc.ConfirmDialogProvider(
                                                                children=html.Button("Remove Rule", id='remove-rule-button', className="btn btn-danger btn-sm",
                                                                                        style=dict(marginLeft=50)),
                                                                id='remove-rule-confirm',
                                                                message='Warning! Once you have removed a rule there is undo button or ctrl-z! Are you sure?'
                                                            ),
                                                            dbc.Tooltip("Remove the selected rule from the model. Warning! Cannot be undone!", 
                                                                        target='remove-rule-button'),
                                                            dcc.ConfirmDialogProvider(
                                                                children=html.Button("Reset Model", id='reset-model-button', className="btn btn-danger btn-sm",
                                                                                        style=dict(marginLeft=10)),
                                                                id='reset-model-confirm',
                                                                message='Warning! Once you have reset the model there is undo button or ctrl-z! Are you sure?'
                                                            ),
                                                            dbc.Tooltip("Reset the model to the initial state. Warning! Cannot be undone!", 
                                                                        target='reset-model-button'),
                                                        ], style = {'width': '100%', 'display': 'flex', 'align-items': 'right', 'justify-content': 'right'}), 
                                                    ], row=True),
                                                ], md=6),
                                            ], form=True, style=dict(marginTop=5)),
                                            dbc.Row([
                                                dbc.Col([
                                                    dcc.Graph(id='model-graph'),
                                                ]),
                                            ]),   
                                        ])
                                ),
                                dcc.Tab(id='model-description-tab', value='model-description-tab', label='Description', 
                                        children=html.Div([dbc.Row([dbc.Col([
                                                dcc.Markdown(id='model-description'),
                                            ])])])
                                ),
                                dcc.Tab(id='model-yaml-tab', value='model-yaml-tab', label='.yaml', 
                                        children=html.Div([dbc.Row([dbc.Col([
                                                html.Div("To instantiate a model from a .yaml file:"),
                                                dcc.Markdown("```\nfrom rule_estimator import *\n"
                                                            "model = RuleClassifier.from_yaml('model.yaml')\n"
                                                            "model.predict(X_test)\n```"),
                                                dbc.Label("model.yaml:"),
                                                dcc.Markdown(id='model-yaml'),
                                            ])])])
                                ),
                                dcc.Tab(id='model-code-tab', value='model-code-tab', label='Code', 
                                        children=dcc.Markdown(id='model-code')
                                ),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H3("Performance"),
                        ]),
                        dbc.CardBody([
                            html.Div(id='model-performance')
                        ])
                    ]),  
                ]),
            ]),
        ])
                                
    def register_callbacks(self, app):

        @app.callback(
            Output('updated-rule-id', 'data'),
            Input('parallel-updated-rule-id', 'data'),
            Input('density-num-updated-rule-id', 'data'),
            Input('density-cats-updated-rule-id', 'data'),
            Input('removerule-updated-rule-id', 'data'),
            Input('resetmodel-updated-rule-id', 'data'),
            Input('model-graph', 'clickData'),
        )
        def update_model(parallel_rule_id, num_rule_id, cats_rule_id, 
                        removerule_rule_id, resetmodel_rule_id, clickdata):
            trigger = self._get_callback_trigger()
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
            State('model-store', 'data')
        )
        def store_model(parallel_update, num_update, cats_update, 
                        removerule_update, resetmodel_update, model):
            if model is None:
                return self.model.to_json()
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
                    f"```python\nmodel = {model.to_code()[1:]}\n```",
                    "update_performance",
                    rule_id_options, rule_id)
        

        #########                                               DENSITY NUM RULE CALLBACK
        @app.callback(
            Output('density-num-updated-rule-id', 'data'),
            Output('density-num-updated-model', 'data'),
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
                    new_rule = LesserThan(col=col, cutoff=cutoff, prediction=int(prediction))
                elif rule_type == 'greater_than':
                    new_rule = GreaterThan(col=col, cutoff=cutoff, prediction=int(prediction))
            elif trigger == 'density-num-split-button':
                if rule_type == 'lesser_than':
                    new_rule = LesserThanSplit(col=col, cutoff=cutoff)
                elif rule_type == 'greater_than':
                    new_rule = GreaterThanSplit(col=col, cutoff=cutoff)
            elif trigger == 'density-num-predict-all-button':
                new_rule = PredictionRule(prediction=int(prediction))
            
            if new_rule is not None:
                rule_id, model = self._change_model(model, rule_id, new_rule, append_or_replace)
                return rule_id, model.to_json()
            raise PreventUpdate

        #########                                               DENSITY CATS RULE CALLBACK
        @app.callback(
            Output('density-cats-updated-rule-id', 'data'),
            Output('density-cats-updated-model', 'data'),
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
                return rule_id, model.to_json()
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
            print(trigger, flush=True)
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
                print(new_rule, rule_id, model, flush=True)
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
            Output('density-num-div', 'style'),
            Output('density-cats-div', 'style'),
            Input('density-col', 'value'),
        )
        def update_density_hidden_divs(col):
            if col is not None:
                if is_numeric_dtype(self.X[col]):
                    return {}, dict(display="none")
                else:
                    return dict(display="none"), {}

        @app.callback(
            Output('density-col', 'value'),
            Output('rule-tabs', 'value'),
            Input('selected-rule-id', 'value'),
            Input('append-or-replace', 'value'),
            State('model-store', 'data'),
        )
        def update_model_node(rule_id, append_or_replace, model):
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
                    return dash.no_update, "parallel-tab"
                elif isinstance(rule, MultiRangeSplit):
                    return dash.no_update, "parallel-tab"
            raise PreventUpdate

        @app.callback(
            Output('density-num-pie', 'figure'),
            Output('density-num-pie-include', 'figure'),
            Output('density-num-pie-exclude', 'figure'),
            Output('density-num-prediction', 'value'),
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
                    rule = LesserThanSplit(col, cutoff)
                elif rule_type == 'greater_than':
                    rule = GreaterThanSplit(col, cutoff)
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
                        isinstance(rule, GreaterThan) or isinstance(rule, GreaterThanSplit)):
                        prediction = rule.prediction

                if prediction is None and not X_rule.empty:
                    prediction = y_rule.value_counts().index[0]
                elif prediction is None and not X.empty:
                    prediction = y.value_counts().index[0]
                else:
                    prediction = 0       
             
                return pie_all, pie_selection, pie_non_selection, str(prediction)
            raise PreventUpdate

        @app.callback(
            Output('density-cats-pie', 'figure'),
            Output('density-cats-pie-include', 'figure'),
            Output('density-cats-pie-exclude', 'figure'),
            Output('density-cats-prediction', 'value'),
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

                return pie_all, pie_selection, pie_non_selection, str(prediction)
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
                    percentage = (percentage=='percentage')
                    fig = plot_cats_density(model, X, y, col, rule_id=rule_id, after=after, 
                                labels=self.labels, percentage=percentage, highlights=cats)
                    return fig, dash.no_update

                elif col in self.non_cats:
                    fig = plot_density(model, X, y, col, rule_id=rule_id, after=after, 
                                labels=self.labels, cutoff=cutoff) 
                    return dash.no_update, fig
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
            State('model-store', 'data'),
            State('density-cats-cats', 'value'),
            State('density-num-cutoff', 'value'),
        )
        def check_cats_clicks(clickdata, col, rule_id, train_or_val, append_or_replace, model, old_cats, old_cutoff):
            trigger = self._get_callback_trigger()
            if append_or_replace=='replace' and trigger in ['selected-rule-id', 'append-or-replace']:
                model = self._get_model(model)
                rule = model.get_rule(rule_id)
                if isinstance(rule, IsInRule) or isinstance(rule, IsInSplit):
                    return rule.cats, dash.no_update, dash.no_update
                if isinstance(rule, GreaterThan) or isinstance(rule, GreaterThanSplit):
                    return dash.no_update, rule.cutoff, "greater_than"
                if isinstance(rule, LesserThan) or isinstance(rule, LesserThanSplit):
                    return dash.no_update, rule.cutoff, "lesser_than"
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
            if col in self.cats:
                return [cat for cat in old_cats if cat in X[col].unique()], dash.no_update, dash.no_update
            elif col in self.non_cats:
                return dash.no_update, X[col].median(), dash.no_update

            raise PreventUpdate

        @app.callback(
            Output('density-num-cutoff', 'min'),
            Output('density-num-cutoff', 'max'),
            Output('density-cats-cats', 'options'),
            Input('density-col', 'value'),
            Input('selected-rule-id', 'value'),
            Input('train-or-val', 'value'),
            Input('append-or-replace', 'value'),
            State('model-store', 'data'),
        )
        def update_density_plot(col, rule_id, train_or_val, append_or_replace, model): 
            if col is not None:
                model = self._get_model(model) 
                X, y = self._get_X_y(train_or_val)
                if col in self.cats:
                    cats_options = [dict(label=cat, value=cat) for cat in X[col].unique()]
                    return dash.no_update, dash.no_update, cats_options
                elif col in self.non_cats:
                    return X[col].min(), X[col].max(), dash.no_update
            raise PreventUpdate

        @app.callback(
            Output('parallel-plot', 'figure'),
            Input('selected-rule-id', 'value'),
            Input('parallel-cols', 'value'),
            Input('append-or-replace', 'value'),
            Input('sort-by-overlap', 'value'),
            Input('train-or-val', 'value'),
            State('parallel-plot', 'figure'),
            State('model-store', 'data'),
        )
        def return_parallel_plot(rule_id, cols, append_or_replace, sort_by_overlap, train_or_val, old_fig, model):
            model = self._get_model(model)
            X, y = self._get_X_y(train_or_val)
            after = self._infer_after(append_or_replace)

            margin = 50
            fig = (plot_parallel_coordinates(model, X, y, rule_id, cols=cols, labels=self.labels, after=after, 
                                                ymin=self.y.min(), ymax=self.y.max(), 
                                                sort_by_histogram_overlap=bool(sort_by_overlap))
                        .update_layout(margin=dict(t=margin, b=margin, l=margin, r=margin)))
            
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
            Output('parallel-pie-all', 'figure'),
            Output('parallel-pie-data', 'figure'),
            Output('parallel-pie-selection', 'figure'),
            Output('parallel-pie-non-selection', 'figure'),
            Output('parallel-pie-all-label', 'children'),
            Output('parallel-pie-data-label', 'children'),
            Output('parallel-pie-selection-label', 'children'),
            Output('parellel-pie-non-selection-label', 'children'),
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

                pie_all = plot_label_pie(model, self.X, self.y, rule_id=0, after=False, size=pie_size)
                pie_data = plot_label_pie(model, self.X, self.y, rule_id=rule_id, after=after, size=pie_size)

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
                    
                return (str(prediction), pie_all, pie_data, pie_selection, pie_non_selection,
                            f"All data ({len(self.X)})", f"This data ({len(X)})", 
                            f"Selected ({len(X_rule)})", f"Not selected ({len(X)-len(X_rule)})")          
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
                    scatter_text = model.score_rules(X, y).drop_duplicates(subset=['rule_id'])['coverage'].apply(lambda x: f"coverage: {100*x:.2f}%").tolist()
                elif scatter_text=='accuracy':
                    scatter_text = model.score_rules(X, y).drop_duplicates(subset=['rule_id'])['accuracy'].apply(lambda x: f"accuracy: {100*x:.2f}%").tolist()
                return plot_model_graph(model, X, y, color_scale=color_scale, highlight_id=highlight_id, scatter_text=scatter_text)
            raise PreventUpdate

    def run(self, debug=False):
        self.app.run_server(port=self.port, use_reloader=False, debug=debug)

                    
        
        