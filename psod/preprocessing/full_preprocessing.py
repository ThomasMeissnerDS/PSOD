import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from psod.preprocessing.scale import minmax_scaling
from psod.preprocessing.reduce_memory_footprint import reduce_mem_usage
import pandas as pd
from typing import List, Union


def auto_preprocess(df, cols: Union[List[str], List[int], List[float], None] = None) -> pd.DataFrame:
    df = reduce_mem_usage(df)
    df = minmax_scaling(df, cols)
    return df
