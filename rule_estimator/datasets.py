__all__ = ['iris_X_y', 'iris_labels', 'titanic_X_y', 'titanic_labels']

from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

iris_labels = load_iris()['target_names'].tolist()
titanic_labels = ['Not Survived', 'Survived']

def iris_X_y():
    """returns a X, y for the iris dataset"""
    return load_iris(return_X_y=True, as_frame=True)

def titanic_X_y():
    df = pd.read_csv(Path(__file__).resolve().parent /"data"/"titanic.csv")
    return df.drop('survived', axis=1), df['survived']