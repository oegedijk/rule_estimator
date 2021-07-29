__all__ = [
    'plot_model_graph',
    'plot_label_pie',
    'plot_parallel_coordinates',
    'plot_density',
    'plot_cats_density',
    'plot_confusion_matrix',
    'get_metrics_df',
    'get_coverage_df',
]

from typing import List, Tuple, Union
import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

empty_fig = go.Figure()
empty_fig.update_layout(
    xaxis =  { "visible": False },
    yaxis = { "visible": False },
    annotations = [{   
            "text": "No data!<br>Try setting after=False or selecting 'replace' instead of 'append'",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 14}
        }])

def plot_model_graph(model, X:pd.DataFrame=None, y:pd.Series=None, 
        color_scale:str='absolute', highlight_id:int=None, scatter_text='name'):
    """
    Returns a plotly Figure of the rules. Uses the Reingolf-Tilford algorithm
    to generate the tree layout. 

    Args:
        model
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

    graph = model.get_igraph()
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
        rule_scores_df = model.score_rules(X, y).drop_duplicates(subset=['rule_id'], keep='first')
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


def plot_label_pie(model, X:pd.DataFrame, y:np.ndarray, rule_id:int=None, after=False,
            size=120, margin=0, showlegend=False):
    if rule_id is not None:
        X, y = model.get_rule_input(rule_id, X, y, after)
    
    if X.empty:
        fig = go.Figure(go.Pie(values=[1.0], showlegend=False, marker=dict(colors=['grey'])))
    else:
        y_vc = y.value_counts().sort_index()

        labels = [str(round(100*lab/len(y), 1))+'%' if lab==y_vc.max() else " " for lab in y_vc]
        fig = go.Figure(
                    go.Pie(
                        labels=labels,
                        values=y_vc.values, 
                        marker=dict(colors=[px.colors.qualitative.Plotly[i] for i in y_vc.index]),
                        sort=False,
                        insidetextorientation='horizontal',
                    ))
    fig.update_layout(showlegend=showlegend, width=size, height=size)
    fig.update_layout(margin=dict(t=margin, b=margin, l=margin, r=margin))
    fig.update_layout(uniformtext_minsize=6, uniformtext_mode='hide')
    fig.update_traces(textinfo='none', hoverinfo='percent', textposition='inside')
    return fig
        

def plot_parallel_coordinates(model, X:pd.DataFrame, y:np.ndarray, rule_id:int=None, 
                            cols:List[str]=None, labels=None, after=False,
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
        X, y = model.get_rule_input(rule_id, X, y, after)
    
    if cols is None:
        cols = X.columns.tolist()

    if X.empty:
        return empty_fig

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

    dimensions = [encode_col(X, col) for col in cols]
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


def plot_density(model, X, y, col, rule_id=0, after=False, labels=None, cutoff=None):

    if labels is None:
        labels = [str(i) for i in range(y.nunique())]

    if rule_id is not None:
        X, y = model.get_rule_input(rule_id, X, y, after)

    if X.empty:
        return empty_fig

    hist_data = [X[y==label][col] for label in y.unique()]
    labels = [labels[label] for label in y.unique()]
    colors = [px.colors.qualitative.Plotly[label] for label in y.unique()]
    
    show_curve = True if len(X) > 10 else False

    try:
        fig = ff.create_distplot(hist_data, labels, show_rug=False, colors=colors, show_curve=show_curve)
    except:
        fig = ff.create_distplot(hist_data, labels, show_rug=False, colors=colors, show_curve=False)

    fig.update_layout(title_text=col, legend=dict(orientation="h"))   
    if isinstance(cutoff, list):
        fig.add_vrect(
            x0=cutoff[0], x1=cutoff[1],
            fillcolor="LightSkyBlue", opacity=0.8,
            layer="below", line_width=0,
        )
    elif cutoff is not None:
        fig.add_vline(cutoff)
    return fig


def plot_cats_density(model, X:pd.DataFrame, y:pd.Series, col:str, 
            rule_id:int=0, after:bool=False, labels:List[str]=None, 
            percentage:bool=False, highlights:List=None)->go.Figure:
    
    if labels is None:
        labels = [str(i) for i in range(y.nunique())]

    if rule_id is not None:
        X, y = model.get_rule_input(rule_id, X, y, after)

    if X.empty:
        return empty_fig

    assert not is_numeric_dtype(X[col])

    fig = go.Figure()
    cats = y.groupby(X[col]).mean().index.tolist()
    if highlights is None:
        highlights = []
    line_widths = [4 if cat in highlights else 0 for cat in cats]
    
    for label in y.unique():
        if percentage:
            y_vals = [len(y[(X[col]==cat) & (y==label)])/len(y[(X[col]==cat)]) for cat in cats]
        else:
            y_vals = [len(y[(X[col]==cat) & (y==label)]) for cat in cats]
        fig.add_trace(go.Bar(
            x=cats, 
            y=y_vals,
            name=labels[label],
            marker_color=px.colors.qualitative.Plotly[label]),
            )
    
        
    fig.update_layout(title=col, barmode='stack', legend=dict(orientation="h"))
    
    for bar in fig.data:
        bar.marker.line.color = 'darkmagenta'
        bar.marker.line.width = line_widths
    return fig


def plot_confusion_matrix(model, X:pd.DataFrame, y:pd.Series,  rule_id:int=0, after:bool=False,  
            rule_only=False, labels=None, percentage=True, normalize='all'):

    if rule_id is not None:
        X, y = model.get_rule_input(rule_id, X, y, after)

    if rule_only:
        rule = model.get_rule(rule_id)
        y_pred = rule.predict(X)
    else:
        y_pred = model.predict(X)
        
    if (~np.isnan(y_pred)).sum() == 0:
        cm = np.array([[0, 0], [0,0]])
    else:
        cm = confusion_matrix(y[~np.isnan(y_pred)], y_pred[~np.isnan(y_pred)])
    
    if normalize not in ['observed', 'pred', 'all']:
        raise ValueError("Error! parameters normalize must be one of {'observed', 'pred', 'all'} !")

    with np.errstate(all='ignore'):
        if normalize == 'all':
            cm_normalized = np.round(100*cm / cm.sum(), 1)
        elif normalize == 'observed':
            cm_normalized = np.round(100*cm / cm.sum(axis=1, keepdims=True), 1)
        elif normalize == 'pred':
            cm_normalized = np.round(100*cm / cm.sum(axis=0, keepdims=True), 1)
         
        cm_normalized = np.nan_to_num(cm_normalized)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])] 

    zmax = 130 # to keep the text readable at 100% accuracy
        
    data=[go.Heatmap(
        z=cm_normalized,
        x=[f" {lab}" for lab in labels],
        y=[f" {lab}" for lab in labels],
        hoverinfo="skip",
        zmin=0, zmax=zmax, colorscale='Blues',
        showscale=False,
    )]
   
    layout = go.Layout(
            title="Confusion Matrix",
            xaxis=dict(title='predicted',
                       constrain="domain",
                       tickmode = 'array',
                       showgrid = False,
                       tickvals = [f" {lab}" for lab in labels],
                       ticktext = [f" {lab}" for lab in labels]),
            yaxis=dict(title=dict(text='observed',standoff=20),
                       autorange="reversed", 
                       side='left',
                       scaleanchor='x', 
                       scaleratio=1,
                       showgrid = False,
                       tickmode = 'array',
                       tickvals = [f" {lab}" for lab in labels],
                       ticktext = [f" {lab}" for lab in labels]),
            plot_bgcolor = '#fff',
        )
    fig = go.Figure(data, layout)
    annotations = []
    for x in range(cm.shape[0]):
        for y in range(cm.shape[1]):
            top_text = f"{cm_normalized[x, y]}%" if percentage else f"{cm[x, y]}"
            bottom_text = f"{cm_normalized[x, y]}%" if not percentage else f"{cm[x, y]}" 
            annotations.extend([
                go.layout.Annotation(
                    x=fig.data[0].x[y], 
                    y=fig.data[0].y[x], 
                    text=top_text, 
                    showarrow=False,
                    font=dict(size=20)
                ),
                go.layout.Annotation(
                    x=fig.data[0].x[y], 
                    y=fig.data[0].y[x], 
                    text=f" <br> <br> <br>({bottom_text})", 
                    showarrow=False,
                    font=dict(size=12)
                )]
            )
    longest_label = max([len(label) for label in labels])    
    fig.update_layout(annotations=annotations)
    fig.update_layout(margin=dict(t=40, b=40, l=longest_label*7, r=40))
    return fig


def get_coverage_df(model, X:pd.DataFrame, y:pd.Series,  rule_id:int=0, after:bool=False,  
            rule_only=False, labels=None, percentage=True, normalize='all'):

    if rule_id is not None:
        X, y = model.get_rule_input(rule_id, X, y, after)

    if rule_only:
        rule = model.get_rule(rule_id)
        y_pred = rule.predict(X)
    else:
        y_pred = model.predict(X)
       
    y_true, y_pred = y[~np.isnan(y_pred)], y_pred[~np.isnan(y_pred)]
    
    coverage_dict = dict(
        n_input = len(X),
        predicted = len(y_pred),
        predicted_nan = len(X)-len(y_pred),
        coverage = round(len(y_pred)/len(X), 3),
    )
    coverage_df = (pd.DataFrame(coverage_dict, index=["count"])
                              .T.rename_axis(index="coverage").reset_index())
    return coverage_df


def get_metrics_df(model, X:pd.DataFrame, y:pd.Series,  rule_id:int=0, after:bool=False,  
            rule_only=False, labels=None, percentage=True, normalize='all'):

    if rule_id is not None:
        X, y = model.get_rule_input(rule_id, X, y, after)

    if rule_only:
        rule = model.get_rule(rule_id)
        y_pred = rule.predict(X)
    else:
        y_pred = model.predict(X)
       
    y_true, y_pred = y[~np.isnan(y_pred)], y_pred[~np.isnan(y_pred)]
    n_input = len(X)
    predicted = len(y_pred)
    not_predicted = len(X)-predicted
    
    
    if len(y_true) > 0:
        metrics_dict = {
            'accuracy' : accuracy_score(y_true, y_pred),
            'precision' : precision_score(y_true, y_pred, zero_division=0),
            'recall' : recall_score(y_true, y_pred),
            'f1' : f1_score(y_true, y_pred),
        }
    else:
        metrics_dict = dict(accuracy=np.nan, precision=np.nan, recall=np.nan, f1=np.nan)
    metrics_df = (pd.DataFrame(metrics_dict, index=["score"])
                              .T.rename_axis(index="metric").reset_index()
                              .round(3))
    
    return metrics_df