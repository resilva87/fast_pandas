"""

"""
import multiprocessing as mp
import numpy as np
import pandas as pd


def apply(series, function_obj, **kwargs):
    """
    bla
    """
    workers = mp.cpu_count() - 1
    if 'workers' in kwargs:
        workers = kwargs.pop('workers')
    pool = mp.Pool(processes=workers)
    partitions = np.array_split(series, workers)
    results = pool.map(_apply_function,
                       [(partition, function_obj, kwargs)
                        for partition in partitions])
    result = pd.concat(results)
    pool.close()
    return result


def _apply_function(args):
    """
    bla
    """
    series, function_obj, kwargs = args
    return series.apply(function_obj, kwargs)
