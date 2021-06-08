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
from .nodes import *
from .rules import *
from .estimators import RuleClassifier


class RuleClassifierDashboard:

    _binary_rules = [
        'GreaterThan', 
        'GreaterEqualThan', 
        'LesserThan', 
        'LesserEqualThan',
    ]
    
    _binary_nodes = [
        'GreaterThanSplit', 
        'GreaterEqualThanSplit', 
        'LesserThanSplit', 
        'LesserEqualThanSplit', 
    ]
    
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
    def _process_constraintrange(ranges:List, ticktext=None, sig=4)->List:
        """helper function to round the range inputs of parallel plots to five
        significant digits"""
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
    def _cats_to_range(cats, cats_order):
        return [[max(0, cats_order.index(cat)-0.25),  min(len(cats_order)-1, cats_order.index(cat)+0.25)] for cat in cats]

    def _get_model(self, json_model):
        if json_model is not None:
            return RuleClassifier.from_json(json_model)
        else:
            return self.model

        
    def layout(self):
        return dbc.Container([
            dcc.Store(id='updated-rule-id'),
            dcc.Store(id='casewhen-updated-rule-id'), 
            dcc.Store(id='empty-updated-rule-id'),
            dcc.Store(id='parallel-updated-rule-id'),
            dcc.Store(id='rule-updated-rule-id'), 
            dcc.Store(id='node-updated-rule-id'),
            dcc.Store(id='prediction-updated-rule-id'),
            dcc.Store(id='isinrule-updated-rule-id'),
            dcc.Store(id='isinnode-updated-rule-id'),
            dcc.Store(id='removerule-updated-rule-id'),
            dcc.Store(id='resetmodel-updated-rule-id'),

            dcc.Store(id='model-store'),
            dcc.Store(id='casewhen-updated-model'), 
            dcc.Store(id='empty-updated-model'),
            dcc.Store(id='parallel-updated-model'),
            dcc.Store(id='rule-updated-model'), 
            dcc.Store(id='node-updated-model'),
            dcc.Store(id='prediction-updated-model'),
            dcc.Store(id='isinrule-updated-model'),
            dcc.Store(id='isinnode-updated-model'),
            dcc.Store(id='removerule-updated-model'),
            dcc.Store(id='resetmodel-updated-model'),

            dcc.Store(id='update-model-performance'),
            dcc.Store(id='update-model-graph'),
            
            dbc.Row([
                dbc.Col([
                    html.H1("RuleClassifierDashboard"), 
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Label("Selected rule:", html_for='selected-rule-id', className="mr-2"),
                                        dcc.Dropdown(id='selected-rule-id', options=[
                                            {'label':str(ruleid), 'value':int(ruleid)} 
                                                for ruleid in range(self.model._get_max_rule_id()+1)],
                                                value=0, clearable=False, style=dict(width=200)),
                                        dbc.Tooltip("You can either select a rule id here or by clicking in the model graph. "
                                                    "The parallel plot will show either all the unlabeled data after this rule "
                                                    "(when you select 'append'), or all the unlabeled data coming into this rule "
                                                    "(when you select 'replace')", target='selected-rule-id'),
                                    ], row=True, className="mr-3"),
                                ]),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.RadioItems(
                                            options=[{'label':'Append rule after (show data out)', 'value':'append'},
                                                    {'label':'Replace rule (show data in)', 'value':'replace'}],
                                            value='append',
                                            id='append-or-replace', 
                                            inline=True),
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
                                            dbc.RadioItems(
                                                    options=[{'label':'Train data', 'value':'train'},
                                                            {'label':'Validation data', 'value':'val'}],
                                                    value='train',
                                                    id='train-or-val', 
                                                    inline=True),
                                            ], className="mr-3"),
                                    ]),
                                ], style=dict(display="none") if self.X_val is None else dict()),     
                            ], form=True),
                        ]),
                    ], style=dict(marginBottom=10)),
                    dbc.Card([
                        dbc.CardHeader([
                            html.H3("Parallel Feature Plot", className="card-title"),
                            html.H6("Select (multiple) feature ranges to generate a split or add a prediction rule", className="card-subtitle"),   
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.FormGroup([
                                            dcc.Dropdown(id='parallel-cols', multi=True,
                                                options=[{'label':col, 'value':col} 
                                                        for col in self.X.columns],
                                                value = self.X.columns.tolist()),
                                            dbc.Tooltip("Select the features to be displayed in the Parallel Plot", 
                                                    target='parallel-cols'),
                                        ]),
                                ], md=10),
                                dbc.Col([
                                    dbc.FormGroup([
                                        #dbc.Label("Sort by histogram overlap", html_for='sort-by-overlap'),
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
                                    dbc.FormGroup([
                                        html.Div([
                                            dbc.Button("Split on Selection", id='add-parallel-node-button', 
                                                            color="primary", size="m"),
                                            dbc.Tooltip("Make a split using the selection in the Parallel Plot. "
                                                    "Data in the selected ranges goes left (true), all other data "
                                                    "goes right (false).", target='add-parallel-node-button'),
                                        ], style=dict(horizontalAlign='center'))  
                                    ]),
                                ], md=4),
                                dbc.Col([
                                    dbc.FormGroup([
                                            dbc.Label("Prediction: ", #size="", 
                                            html_for='parallel-prediction',
                                                style=dict(verticalAlignment='text-bottom')),
                                            dcc.Dropdown(id='parallel-prediction', options=[
                                                {'label':f"{y}. {label}", 'value':y} for y, label in enumerate(self.labels)],
                                                value=len(self.labels)-1, clearable=False, 
                                                style={'height': '20px', 'width':'150px'}),
                                            dbc.Tooltip("The prediction to be applied. Either to all data ('Predict All'), "
                                                        "or to the selected data ('Predict Selected'). Will get automatically "
                                                        "Inferred from the selected data in the Parallel Plot.", target='parallel-prediction'),
                                        ], row=True),
                                ], md=3),
                                dbc.Col([
                                    dbc.FormGroup([
                                            dbc.Button("Predict Selected Only", id='add-parallel-rule-button', 
                                                    color="primary", size="s", style=dict(marginRight=10)),
                                            dbc.Tooltip("Apply the prediction to all data within the ranges "
                                                        "selected in the Parallel Plot.", target='add-parallel-rule-button'),
                                            dbc.Button("Predict All", id='add-prediction-button', 
                                                        color="primary", size="m"),
                                            dbc.Tooltip("Add a PredictionRule: Apply a single uniform prediction to all the "
                                                        "data without distinction.", target='add-prediction-button'),
                                    ], row=True),
                                ], md=5),
                            ], form=True),
                        ]),
                        dbc.CardFooter([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("All data", id='pie-all-label'),
                                    dbc.Tooltip("Label distribution for all observations", target='pie-all-label'),
                                    dcc.Graph(id='pie-all', config=dict(modeBarButtons=[[]], displaylogo=False)),  
                                ]),
                                dbc.Col([
                                    dbc.Label("This data", id='pie-rule-id-label'),
                                    dbc.Tooltip("Label distribution for all observations in the parallel plot above.", 
                                            target='pie-rule-id-label'),
                                    dcc.Graph(id='pie-rule-id', config=dict(modeBarButtons=[[]], displaylogo=False)),   
                                ]),
                                dbc.Col([
                                    dbc.Label("Selected", id='pie-parallel-selection-label'),
                                    dbc.Tooltip("Label distribution for all the feature ranges selected above", 
                                            target='pie-parallel-selection-label'),
                                    dcc.Graph(id='pie-parallel-selection', config=dict(modeBarButtons=[[]], displaylogo=False)),   
                                ]),
                                dbc.Col([
                                    dbc.Label("Not selected", id='pie-parallel-non-selection-label'),
                                    dbc.Tooltip("Label distribution for all the feature ranges not selected above", 
                                            target='pie-parallel-non-selection-label'),
                                    dcc.Graph(id='pie-parallel-non-selection', config=dict(modeBarButtons=[[]], displaylogo=False)),
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
                                            dbc.Form([
                                                dbc.FormGroup([
                                                        dbc.Label("Color scale: ", html_for='absolute-or-relative', className="mr-2"),
                                                        dbc.RadioItems(
                                                            options=[{'label':'Absolute', 'value':'absolute'},
                                                                        {'label':'Relative', 'value':'relative'}],
                                                            value='absolute',
                                                            id='model-graph-color-scale', 
                                                            inline=True),
                                                    ], row=True, className="mr-3"), 
                                                dbc.FormGroup([
                                                        dbc.Label("Display: ", html_for='model-graph-scatter-text', className="mr-2"),
                                                        dcc.Dropdown(id='model-graph-scatter-text',
                                                            options=[dict(label=o, value=o) for o in ['name', 'description', 'coverage', 'accuracy']],
                                                            value='description', clearable=False,
                                                            style=dict(width=130)),
                                                    ], row=True, className="mr-3"), 
                                            ], inline=True),
                                            dcc.Graph(id='model-graph'),
                                        ])),
                                dcc.Tab(id='model-description-tab', value='model-description-tab', label='Description', 
                                        children=html.Div([dbc.Row([dbc.Col([
                                                dcc.Markdown(id='model-description'),
                                            ])])])),
                                dcc.Tab(id='model-yaml-tab', value='model-yaml-tab', label='.yaml', 
                                        children=html.Div([dbc.Row([dbc.Col([
                                                html.Div("To instantiate a model from a .yaml file:"),
                                                dcc.Markdown("```\nfrom rule_estimator import *\n"
                                                            "model = RuleClassifier.from_yaml('model.yaml')\n"
                                                            "model.predict(X_test)\n```"),
                                                dbc.Label("model.yaml:"),
                                                dcc.Markdown(id='model-yaml'),
                                            ])])])),
                                dcc.Tab(id='model-code-tab', value='model-code-tab', label='Code', 
                                        children=dcc.Markdown(id='model-code')),
                                dcc.Tab(id='model-download-tab', value='model-daownload-tab', label="Download",
                                        children=dbc.Card([
                                                    dbc.CardHeader([
                                                        html.H3("Save model")
                                                    ]),
                                                    dbc.CardBody([
                                                        dbc.Button("Download pickle", id='download-pickle-button', 
                                                                    color="primary", size="m"),
                                                        dcc.Download(id='download-pickle'),
                                                        html.Div([html.P()]),
                                                        dbc.Button("Download .yaml", id='download-yaml-button', color="primary", size="m"),
                                                        dcc.Download(id='download-yaml'),
                                                    ])
                                                ])),
                            ]),
                        ]),
                    ]),
                ], md=9),
                dbc.Col([                
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Other Rules"),
                        ]),
                        dbc.CardBody([
                            html.Div([html.P()]),
                            dbc.Button("CaseWhen", id='add-casewhen-button', color="primary", size="m"),
                            dbc.Tooltip("Add a new CaseWhen rule. You can append multiple rules to a CaseWhen wrapper "
                                        "that will be evaluated one-by-one.", target='add-casewhen-button'),
                            html.Div([html.P()]),
                            dbc.Button("EmptyRule", id='add-empty-button', color="primary", size="m"),
                            dbc.Tooltip("Add or replace with an EmptyRule that does not provide any prediction, "
                                        "but simply acts as a placeholder for other rules.", target='add-casewhen-button'),
                            html.Div([html.P()]),
                            dbc.Button(
                                "Cutoff Predict", id="collapse-binaryrule-button",
                                className="mb-3", size="m"),
                            dbc.Tooltip("Add a rule with a single cutoff for a single numerical feature. Rule can either "
                                        "be of type >, >=, < or <= ", target='collapse-binaryrule-button'),
                            dbc.Collapse(
                                dbc.Card(dbc.CardBody(
                                    html.Div([
                                        dbc.Label("Rule"),
                                        dcc.Dropdown(id='rule-rule', 
                                                    options=[{'label':col, 'value':col} 
                                                                for col in self._binary_rules], 
                                                    value = self._binary_rules[0],
                                                    clearable=False),
                                        dbc.Label("Col"),
                                        dcc.Dropdown(id='rule-col', 
                                                    options=[{'label':col, 'value':col} 
                                                                for col in self.non_cats], 
                                                    clearable=False),
                                        dbc.Label("Cutoff"),
                                        dbc.Input(id='rule-cutoff', type="number"),
                                        dbc.Label("Prediction"),
                                        dcc.Dropdown(id='rule-prediction', options=[
                                            {'label':f"{y}. {label}", 'value':y} for y, label in enumerate(self.labels)],
                                            value=1, clearable=False),
                                        html.P(),
                                        dbc.Button("Add Rule", id='add-binaryrule-button', color="primary", size="m"),       
                                    ]))),
                                id="collapse-binaryrule",
                            ),
                            dbc.Button(
                                "Cutoff Split",
                                id="collapse-binarynode-button",
                                className="mb-3",
                                size="m",
                            ),
                            dbc.Tooltip("Add a split with a single cutoff for a single numerical feature. Split can either "
                                        "be of type >, >=, < or <= ", target='collapse-binarynode-button'),
                            dbc.Collapse(
                                dbc.Card(dbc.CardBody(
                                    html.Div([
                                        dbc.Label("Split"),
                                        dcc.Dropdown(id='node-rule', 
                                                    options=[{'label':col, 'value':col} 
                                                                for col in self._binary_nodes],
                                                    value = self._binary_nodes[0],
                                                    clearable=False),
                                        dbc.Label("Col"),
                                        dcc.Dropdown(id='node-col', 
                                                    options=[{'label':col, 'value':col} 
                                                                for col in self.non_cats],
                                                    clearable=False),
                                        dbc.Label("Cutoff"),
                                        dbc.Input(id='node-cutoff', type="number"),
                                        html.P(),
                                        dbc.Button("Add Split", id='add-binarynode-button', color="primary", size="sm"),    
                                    ]))),
                                id="collapse-binarynode",
                            ),
                            dbc.Button(
                                "Category Predict",
                                id="collapse-isinrule-button",
                                className="mb-3",
                                size="m"
                            ),
                            dbc.Tooltip("Add a prediction based on a single categorical feature.", 
                                            target='collapse-isinrule-button'),
                            dbc.Collapse(
                                dbc.Card(dbc.CardBody(
                                    html.Div([
                                        dbc.Label("Col"),
                                        dcc.Dropdown(id='isinrule-col', 
                                                    options=[{'label':col, 'value':col} 
                                                                for col in self.cats], 
                                                    clearable=False),
                                        dbc.Label("Categories"),
                                        dcc.Dropdown(id='isinrule-cats', multi=True),
                                        dbc.Label("Prediction"),
                                        dcc.Dropdown(id='isinrule-prediction', options=[
                                            {'label':f"{y}. {label}", 'value':y} for y, label in enumerate(self.labels)],
                                            value=1, clearable=False),
                                        html.P(),
                                        dbc.Button("Add Rule", id='add-isinrule-button', color="primary", size="m"),       
                                    ]))),
                                id="collapse-isinrule",
                            ),
                            dbc.Button(
                                "Category Split",
                                id="collapse-isinnode-button",
                                className="mb-3",
                                size="m",
                            ),
                            dbc.Tooltip("Add a split based on a single categorical feature.", 
                                            target='collapse-isinnode-button'),
                            dbc.Collapse(
                                dbc.Card(dbc.CardBody(
                                    html.Div([
                                        dbc.Label("Col"),
                                        dcc.Dropdown(id='isinnode-col', 
                                                    options=[{'label':col, 'value':col} 
                                                                for col in self.cats],
                                                    clearable=False),
                                        dbc.Label("Categories"),
                                        dcc.Dropdown(id='isinnode-cats', multi=True),

                                        html.P(),
                                        dbc.Button("Add Split", id='add-isinnode-button', color="primary", size="m"),    
                                    ]))),
                                id="collapse-isinnode",
                            ),
                        ]),
                    ]), 
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                dcc.ConfirmDialogProvider(
                                    children=html.Button("Remove Rule", id='remove-rule-button', className="btn btn-danger"),
                                    id='remove-rule-confirm',
                                    message='Warning! Once you have removed a rule there is undo button or ctrl-z! Are you sure?'
                                ),
                                dbc.Tooltip("Remove the selected rule from the model. Warning! Cannot be undone!", 
                                            target='remove-rule-button'),
                            ]),
                            html.Div([
                                dcc.ConfirmDialogProvider(
                                    children=html.Button("Reset Model", id='reset-model-button', className="btn btn-danger"),
                                    id='reset-model-confirm',
                                    message='Warning! Once you have reset the model there is undo button or ctrl-z! Are you sure?'
                                ),
                                dbc.Tooltip("Reset the model to the initial state. Warning! Cannot be undone!", 
                                            target='reset-model-button'),
                            ], style=dict(marginTop=10))   
                        ]),
                    ], style=dict(marginTop=20)),
                ], md=3),
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
            Output('model-store', 'data'),
            Input('casewhen-updated-model', 'data'),
            Input('empty-updated-model', 'data'),
            Input('parallel-updated-model', 'data'),
            Input('rule-updated-model', 'data'),
            Input('node-updated-model', 'data'),
            Input('prediction-updated-model', 'data'),
            Input('isinrule-updated-model', 'data'),
            Input('isinnode-updated-model', 'data'),
            Input('removerule-updated-model', 'data'),
            Input('resetmodel-updated-model', 'data'),
            Input('model-store', 'modified_timestamp'),
            State('model-store', 'data')
        )
        def store_model(casewhen_update, empty_update, parallel_update, 
            rule_update, node_update, prediction_update, isinrule_update, isinnode_update, 
            removerule_update, resetmodel_update, ts, model):
            if model is None:
                return self.model.to_json()
            
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'casewhen-updated-model':
                if casewhen_update is not None: return casewhen_update
            elif trigger == 'empty-updated-model':
                if empty_update is not None: return empty_update
            elif trigger == 'parallel-updated-model':
                if parallel_update is not None: return parallel_update
            elif trigger == 'rule-updated-model':
                if rule_update is not None: return rule_update
            elif trigger == 'node-updated-model':
                if node_update is not None: return node_update
            elif trigger == 'prediction-updated-model':
                if prediction_update is not None: return prediction_update
            elif trigger == 'isinrule-updated-model':
                if isinrule_update is not None: return isinrule_update
            elif trigger == 'isinnode-updated-model':
                if isinnode_update is not None: return isinnode_update
            elif trigger == 'removerule-updated-model':
                if removerule_update is not None: return removerule_update
            elif trigger == 'resetmodel-updated-model':
                if resetmodel_update is not None: return resetmodel_update
            raise PreventUpdate

        @app.callback(
            Output('updated-rule-id', 'data'),
            Input('casewhen-updated-rule-id', 'data'),
            Input('empty-updated-rule-id', 'data'),
            Input('parallel-updated-rule-id', 'data'),
            Input('rule-updated-rule-id', 'data'),
            Input('node-updated-rule-id', 'data'),
            Input('prediction-updated-rule-id', 'data'),
            Input('isinrule-updated-rule-id', 'data'),
            Input('isinnode-updated-rule-id', 'data'),
            Input('removerule-updated-rule-id', 'data'),
            Input('resetmodel-updated-rule-id', 'data'),
            Input('model-graph', 'clickData'),
        )
        def update_model(casewhen_rule_id, empty_rule_id, parallel_rule_id, 
                         rule_rule_id, node_rule_id, prediction_rule_id, 
                         isinrule_rule_id, isinnode_rule_id, removerule_rule_id, 
                         resetmodel_rule_id, clickdata):
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'model-graph':
                if (clickdata is not None and clickdata['points'][0] is not None and 
                    'hovertext' in clickdata['points'][0]):
                    rule = clickdata['points'][0]['hovertext'].split('rule:')[1].split('<br>')[0]
                    if rule is not None:
                        return int(rule)
            if trigger == 'casewhen-updated-rule-id':
                return casewhen_rule_id
            if trigger == 'empty-updated-rule-id':
                return empty_rule_id
            if trigger == 'parallel-updated-rule-id':
                return parallel_rule_id
            elif trigger == 'rule-updated-rule-id':
                return rule_rule_id
            elif trigger == 'node-updated-rule-id':
                return node_rule_id
            elif trigger == 'prediction-updated-rule-id':
                return prediction_rule_id
            elif trigger == 'isinrule-updated-rule-id':
                return isinrule_rule_id
            elif trigger == 'isinnode-updated-rule-id':
                return isinnode_rule_id
            elif trigger == 'removerule-updated-rule-id':
                return removerule_rule_id
            elif trigger == 'resetmodel-updated-rule-id':
                return resetmodel_rule_id
            raise PreventUpdate

        @app.callback(
            Output("collapse-binaryrule", "is_open"),
            [Input("collapse-binaryrule-button", "n_clicks")],
            [State("collapse-binaryrule", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        @app.callback(
            Output("collapse-isinrule", "is_open"),
            [Input("collapse-isinrule-button", "n_clicks")],
            [State("collapse-isinrule", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        @app.callback(
            Output("collapse-binarynode", "is_open"),
            [Input("collapse-binarynode-button", "n_clicks")],
            [State("collapse-binarynode", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        @app.callback(
            Output("collapse-isinnode", "is_open"),
            [Input("collapse-isinnode-button", "n_clicks")],
            [State("collapse-isinnode", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open
                        
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
        
        @app.callback(
            Output('casewhen-updated-rule-id', 'data'),
            Output('casewhen-updated-model', 'data'),
            Input('add-casewhen-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('model-store', 'data'),
        )
        def update_model_rule(n_clicks, rule_id, append_or_replace, model):
            if n_clicks is not None:
                model = self._get_model(model)
                new_rule_id = None
                if append_or_replace=='append':
                    model.append_rule(rule_id, CaseWhen())
                elif append_or_replace=='replace':
                    new_rule = model.replace_rule(rule_id, CaseWhen())
                    new_rule_id = new_rule._rule_id
                return new_rule_id, model.to_json()
            raise PreventUpdate
        
        @app.callback(
            Output('empty-updated-rule-id', 'data'),
            Output('empty-updated-model', 'data'),
            Input('add-empty-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('model-store', 'data'),
        )
        def update_model_rule(n_clicks, rule_id, append_or_replace, model):
            if n_clicks is not None:
                model = self._get_model(model)
                new_rule_id = None
                if append_or_replace=='append':
                    model.append_rule(rule_id, EmptyRule())
                elif append_or_replace=='replace':
                    new_rule = model.replace_rule(rule_id, EmptyRule())
                    new_rule_id = new_rule._rule_id
                return new_rule_id, model.to_json()
            raise PreventUpdate
        
        @app.callback(
            Output('rule-updated-rule-id', 'data'),
            Output('rule-updated-model', 'data'),
            Input('add-binaryrule-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('rule-rule', 'value'),
            State('rule-col', 'value'),
            State('rule-cutoff', 'value'),
            State('rule-prediction', 'value'),
            State('model-store', 'data'),
        )
        def update_model_rule(n_clicks, rule_id, append_or_replace, new_rule, col, cutoff, prediction, model):
            if n_clicks is not None:
                model = self._get_model(model)
                new_rule_id = None
                if append_or_replace=='append':
                    new_rule = globals()[new_rule]
                    new_rule_id = model.append_rule(rule_id, new_rule(col=col, cutoff=cutoff, prediction=prediction))
                elif append_or_replace=='replace':
                    new_rule = globals()[new_rule]
                    new_rule = model.replace_rule(rule_id, new_rule(col=col, cutoff=cutoff, prediction=prediction))
                    new_rule_id = new_rule._rule_id
                return new_rule_id, model.to_json()
            raise PreventUpdate

        @app.callback(
            Output('isinrule-cats', 'options'),
            Output('isinrule-cats', 'value'),
            Input('isinrule-col', 'value'),
            Input('selected-rule-id', 'value'),
            State('train-or-val', 'value'),
            State('isinrule-cats', 'value'),
            State('model-store', 'data'),
        )
        def update_isinrule_cats(col, rule_id, train_or_val, old_cats, model):
            if col is not None and rule_id is not None:
                model = self._get_model(model)
                X, y = (self.X, self.y) if train_or_val == 'train' else (self.X_val, self.y_val)
                X, y = model.get_rule_input(rule_id, X, y)
                cats = X[col].unique().tolist()
                options = [dict(label=cat, value=cat) for cat in cats]
                if old_cats is None:
                    old_cats = []
                value = [cat for cat in old_cats if cat in cats]
                return options, value
            raise PreventUpdate

        @app.callback(
            Output('isinnode-cats', 'options'),
            Output('isinnode-cats', 'value'),
            Input('isinnode-col', 'value'),
            Input('selected-rule-id', 'value'),
            State('train-or-val', 'value'),
            State('isinnode-cats', 'value'),
            State('model-store', 'data'),
        )
        def update_isinrule_cats(col, rule_id, train_or_val, old_cats, model):
            if col is not None and rule_id is not None:
                model = self._get_model(model)
                X, y = (self.X, self.y) if train_or_val == 'train' else (self.X_val, self.y_val)
                X, y = model.get_rule_input(rule_id, X, y)
                cats = X[col].unique().tolist()
                options = [dict(label=cat, value=cat) for cat in cats]
                if old_cats is None:
                    old_cats = []
                value = [cat for cat in old_cats if cat in cats]
                return options, value
            raise PreventUpdate

        @app.callback(
            Output('isinrule-updated-rule-id', 'data'),
            Output('isinrule-updated-model', 'data'),
            Input('add-isinrule-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('isinrule-col', 'value'),
            State('isinrule-cats', 'value'),
            State('rule-prediction', 'value'),
            State('model-store', 'data'),
        )
        def update_model_rule(n_clicks, rule_id, append_or_replace, col, cats, prediction, model):
            if n_clicks is not None:
                model = self._get_model(model)
                new_rule_id = None
                if append_or_replace=='append':
                    new_rule_id = model.append_rule(rule_id, IsInRule(col=col, cats=cats, prediction=prediction))
                elif append_or_replace=='replace':
                    new_rule = model.replace_rule(rule_id, IsInRule(col=col, cats=cats, prediction=prediction))
                    new_rule_id = new_rule._rule_id
                return new_rule_id, model.to_json()
            raise PreventUpdate

        @app.callback(
            Output('isinnode-updated-rule-id', 'data'),
            Output('isinnode-updated-model', 'data'),
            Input('add-isinnode-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('isinnode-col', 'value'),
            State('isinnode-cats', 'value'),
            State('model-store', 'data'),
        )
        def update_model_rule(n_clicks, rule_id, append_or_replace, col, cats, model):
            if n_clicks is not None:
                model = self._get_model(model)
                new_rule_id = None
                if append_or_replace=='append':
                    new_rule_id = model.append_rule(rule_id, IsInSplit(col=col, cats=cats))
                elif append_or_replace=='replace':
                    new_rule = model.replace_rule(rule_id, IsInSplit(col=col, cats=cats))
                    new_rule_id = new_rule._rule_id
                return new_rule_id, model.to_json()
            raise PreventUpdate

        @app.callback(
            Output('rule-rule', 'value'),
            Output('rule-col', 'value'),
            Output('rule-cutoff', 'value'),
            Output('rule-prediction', 'value'),
            Input('selected-rule-id', 'value'),
            Input('append-or-replace', 'value'),
            State('model-store', 'data'),
        )
        def update_model_node(rule_id, append_or_replace, model):
            if append_or_replace == 'replace' and rule_id is not None:
                model = self._get_model(model)
                rule = model.get_rule(rule_id)
                if any([isinstance(rule, binnode) for binnode in [
                            LesserThan, LesserEqualThan, 
                            GreaterThan, GreaterEqualThan]]):
                    return rule.__class__.__name__, rule.col, rule.cutoff, rule.prediction
            raise PreventUpdate
                    
        @app.callback(
            Output('node-updated-rule-id', 'data'),
            Output('node-updated-model', 'data'),
            Input('add-binarynode-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('node-rule', 'value'),
            State('node-col', 'value'),
            State('node-cutoff', 'value'),
            State('model-store', 'data'),
        )
        def update_model_node(n_clicks, rule_id, append_or_replace, new_rule, col, cutoff, model):
            if n_clicks is not None:
                model = self._get_model(model)
                new_rule_id = None
                if append_or_replace=='append':
                        new_rule = globals()[new_rule]
                        new_rule_id = model.append_rule(rule_id, new_rule(col, cutoff))
                elif append_or_replace=='replace':
                    new_rule = globals()[new_rule]
                    new_rule = model.replace_rule(rule_id, new_rule(col, cutoff))
                    new_rule_id = new_rule._rule_id
                return new_rule_id, model.to_json()
            raise PreventUpdate

        @app.callback(
            Output('node-rule', 'value'),
            Output('node-col', 'value'),
            Output('node-cutoff', 'value'),
            Input('selected-rule-id', 'value'),
            Input('append-or-replace', 'value'),
            State('model-store', 'data'),
        )
        def update_model_node(rule_id, append_or_replace, model):
            if append_or_replace == 'replace' and rule_id is not None:
                model = self._get_model(model)
                rule = model.get_rule(rule_id)
                if any([isinstance(rule, binnode) for binnode in [LesserThanSplit, LesserEqualThanSplit, GreaterThanSplit, GreaterEqualThanSplit]]):
                    return rule.__class__.__name__, rule.col, rule.cutoff
            raise PreventUpdate
        
        @app.callback(
            Output('prediction-updated-rule-id', 'data'),
            Output('prediction-updated-model', 'data'),
            Input('add-prediction-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('parallel-prediction', 'value'),
            State('model-store', 'data'),
        )
        def update_model_prediction(n_clicks, rule_id, append_or_replace, prediction, model):
            if n_clicks is not None:
                model = self._get_model(model)
                new_rule_id = None
                if append_or_replace=='append':
                    new_rule_id = model.append_rule(rule_id, PredictionRule(prediction=prediction))
                elif append_or_replace=='replace':
                    new_rule = model.replace_rule(rule_id, PredictionRule(prediction=prediction))
                    new_rule_id = new_rule._rule_id
                return new_rule_id, model.to_json()
            raise PreventUpdate
        
        @app.callback(
            Output('parallel-updated-rule-id', 'data'),
            Output('parallel-updated-model', 'data'),
            Input('add-parallel-rule-button', 'n_clicks'),
            Input('add-parallel-node-button', 'n_clicks'),
            State('selected-rule-id', 'value'),
            State('append-or-replace', 'value'),
            State('parallel-prediction', 'value'),
            State('parallel-plot', 'figure'),
            State('model-store', 'data'),
        )
        def update_model_parallel(rule_clicks, node_clicks, rule_id, 
                                  append_or_replace, prediction, fig, model):
            if (rule_clicks is not None or node_clicks is not None) and fig is not None:
                model = self._get_model(model)
                plot_data = fig['data'][0].get('dimensions', None)
                range_dict = {}
                for col_data in plot_data:
                    if col_data['label'] != 'y' and 'constraintrange' in col_data:
                        range_dict[col_data['label']] = self._process_constraintrange(
                            col_data['constraintrange'], col_data['ticktext'] if 'ticktext' in col_data else None)
                
                ctx = dash.callback_context
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                if trigger == 'add-parallel-rule-button':
                    rule = MultiRange(range_dict, prediction)
                elif trigger == 'add-parallel-node-button':
                    rule = MultiRangeSplit(range_dict)
                else:
                    raise PreventUpdate
                    
                new_rule_id = None
                if append_or_replace=='append':
                    new_rule_id = model.append_rule(rule_id, rule)
                elif append_or_replace=='replace':
                    new_rule = model.replace_rule(rule_id, rule)
                    new_rule_id = new_rule._rule_id
                return new_rule_id, model.to_json()
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
            margin = 50
            after = (append_or_replace == 'append')
            X, y = (self.X, self.y) if train_or_val == 'train' else (self.X_val, self.y_val)
                
            fig = (model.parallel_coordinates(X, y, rule_id, cols=cols, labels=self.labels, after=after, 
                                                ymin=self.y.min(), ymax=self.y.max(), 
                                                sort_by_histogram_overlap=bool(sort_by_overlap))
                        .update_layout(margin=dict(t=margin, b=margin, l=margin, r=margin)))
            if fig['data'] and 'dimensions' in fig['data'][0]:
                ctx = dash.callback_context
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                if ((trigger == 'selected-rule-id' or trigger == 'append-or-replace') and 
                    append_or_replace=='replace'):
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
            Output('pie-all', 'figure'),
            Output('pie-rule-id', 'figure'),
            Output('pie-parallel-selection', 'figure'),
            Output('pie-parallel-non-selection', 'figure'),
            Output('pie-all-label', 'children'),
            Output('pie-rule-id-label', 'children'),
            Output('pie-parallel-selection-label', 'children'),
            Output('pie-parallel-non-selection-label', 'children'),
            Input('parallel-plot', 'restyleData'),
            Input('selected-rule-id', 'value'),
            Input('append-or-replace', 'value'),
            Input('parallel-plot', 'figure'),  
            State('model-store', 'data'),
        )
        def update_parallel_prediction(restyle, rule_id, append_or_replace, fig, model):
            if fig is not None and fig['data']:
                model = self._get_model(model)
                plot_data = fig['data'][0].get('dimensions', None)
                range_dict = {}
                for col_data in plot_data:
                    if col_data['label'] != 'y' and 'constraintrange' in col_data:
                        range_dict[col_data['label']] = self._process_constraintrange(
                            col_data['constraintrange'], col_data['ticktext'] if 'ticktext' in col_data else None)
                rule = MultiRangeSplit(range_dict)
                
                after = (append_or_replace == 'append')
                pie_size = 50
                pie_all = model.pie(self.X, self.y, rule_id=0, after=False, size=pie_size)
                pie_rule_id = model.pie(self.X, self.y, rule_id=rule_id, after=after, size=pie_size)
                
                X, y = model.get_rule_input(rule_id, self.X, self.y, after)
                if X.empty:
                    raise PreventUpdate
                X_rule, y_rule = X[rule.__rule__(X)], y[rule.__rule__(X)]
                if not X_rule.empty:
                    prediction = y_rule.value_counts().index[0]
                    pie_selection = model.pie(X_rule, y_rule, size=pie_size)
                else:
                    prediction = y.value_counts().index[0]
                    pie_selection = pie_rule_id

                if not len(X) == len(X_rule):
                    pie_non_selection = model.pie(X[~rule.__rule__(X)], y[~rule.__rule__(X)], size=pie_size)
                else:
                    pie_non_selection = pie_rule_id

                ctx = dash.callback_context
                trigger = ctx.triggered[0]['prop_id'].split('.')[0]
                if trigger == 'selected-rule-id' and append_or_replace=='replace':
                    rule = model.get_rule(rule_id)
                    if isinstance(rule, PredictionRule):
                        prediction = rule.prediction
                    
                return (prediction, pie_all, pie_rule_id, pie_selection, pie_non_selection,
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
                X, y = (self.X, self.y) if train_or_val == 'train' else (self.X_val, self.y_val)
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
                X, y = (self.X, self.y) if train_or_val=='train' else (self.X_val, self.y_val)
                if scatter_text=='coverage':
                    scatter_text = model.score_rules(X, y).drop_duplicates(subset=['rule_id'])['coverage'].apply(lambda x: f"coverage: {100*x:.2f}%").tolist()
                elif scatter_text=='accuracy':
                    scatter_text = model.score_rules(X, y).drop_duplicates(subset=['rule_id'])['accuracy'].apply(lambda x: f"accuracy: {100*x:.2f}%").tolist()
                return model.plot(X, y, color_scale=color_scale, highlight_id=highlight_id, scatter_text=scatter_text)
            raise PreventUpdate

             
    def run(self, debug=False):
        self.app.run_server(port=self.port, use_reloader=False, debug=debug)

                    
        
        