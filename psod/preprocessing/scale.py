import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Union


def minmax_scaling(df: pd.DataFrame, cols: Union[List[str], List[int], List[float], None] = None) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled = scaler.transform(df)

    if isinstance(cols, list):
        pass
    else:
        cols = df.columns

    scaled = pd.DataFrame(scaled, columns=cols)
    return scaled
