import pandas as pd
import numpy as np
from functools import wraps

def fill_empty_fields_with_value(df, value=np.nan):
    """Replace empty fields with `value`.

    Args:
        df (:class:`pandas.DataFrame`): DataFrame containing all data.
        value (optional, any): Value with which to replace empty data.

    Returns:
        :class:`pandas.DataFrame`: DataFrame with empty fields filled with `value`.
    """
    return df.replace(r'^\s*$', value, regex=True)

def make_numeric(df, errors='coerce'):
    """Make DataFrame `df` numeric.

    Args:
        df (:class:`pandas.DataFrame`): DataFrame you want to make numeric.
        errors (optional, str): How to handle errors. See: pd.to_numeric kwargs.

    Returns:
        :class:`pandas.DataFrame`: Numeric DataFrame.
    """
    return df.apply(pd.to_numeric, errors=errors)

def df_from_records(records, index=None, levels=None, flat=False):
    """Creates DataFrame from a nested dict or list of dicts.

    Args:
        records (list[dict]): List of nested dicts from which you want to make a DataFrame.
        flat (optional, bool): Whether columns should be flat or nested (a MultiIndex)

    Returns:
        :class:`pandas.DataFrame`: DataFrame containing all data.
    """
    if isinstance(records, dict):
        records = records.values()
    # create flat DataFrame from dict
    df = pd.io.json.json_normalize(records)
    if index is not None:
        df = df.set_index(index)
    if not flat:
        # make it hierarchical 
        df.columns = pd.MultiIndex.from_tuples([tuple(c.split('.')) for c in df.columns])
        if levels is not None:
            df.columns = df.columns.rename(levels)
    return df

def handle_transpose(multiindex='columns'):
    """Handles operations on a MultiIndexed DataFrame regardless of orientation (i.e. transposed or not).

    Args:
        multiindex (optional, str ('columns' | 'index')): Orientation of the DataFrame
            assumed in the wrapped func, i.e. columns as MultiIndex or index as MultiIndex.
    """
    multiindex = multiindex.lower()
    def handle_transpose_decorator(func):
        """Decorator to make functions able to take df with either index or columns as MultiIndex.

        Args:
            func (callable): Function to be wrapped. Must take DataFrame as first argument!

        Returns:
            wrapper (callable): Wrapper which returns modified DataFrame.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            df = args.pop(0)
            transposed = not isinstance(getattr(df, multiindex), pd.MultiIndex)
            if transposed:
                df = df.T
            df = func(df, *args, **kwargs)
            if transposed:
                df = df.T
            df = cleanup_indices(df)
            return df
        return wrapper
    return handle_transpose_decorator

def cleanup_indices(df):
    """Merge duplicate indices.

    Args:
        df (:class:`pandas.DataFrame`): Our DataFrame.

    Returns:
        :class:`pandas.DataFrame`: DataFrame with duplicate indices merged.
    """
    transposed = isinstance(df.columns, pd.MultiIndex)
    if transposed:
        df = df.T
    if isinstance(df.index, pd.MultiIndex):
        for i, level in enumerate(df.index.levels):
            df = df.reindex(index=level, level=i)
    return df.T if transposed else df

@handle_transpose(multiindex='columns')
def filter(df, condition):
    """Returns entries in `df` where `condition` (a boolean DataFrame) is True.

    Args:
        df (:class:`pandas.DataFrame`): Our DataFrame.
        condition (:class:`pandas.DataFrame`): Boolean DataFrame used to filter `df`.

    Returns:
        :class:`pandas.DataFrame`: Entries in `df` where `condition` is True.
    """
    return df.loc[condition[condition].index]

@handle_transpose(multiindex='index')
def swap_levels(df, i=-2, j=-1, outer_indices=None):
    """Swaps levels in a MultiIndex

    Args:
        df (:class:`pandas.DataFrame`): Our DataFrame
        outer_indices (str | sequence[str]): List of outermost indices for which to swap inner-most indices.
        i (optional, int | str): Levels to swap. Default: innermost two levels.
        j (optional, int | str): Levels to swap. Default: innermost two levels.

    Returns:
        :class:`pandas.DataFrame`: DataFrame with innermost indices swapped and duplicate indices merged.
    """
    if outer_indices is not None:
        if not isinstance(outer_indices, (list, tuple)):
            outer_indices = [outer_indices]
        for key in outer_indices:
            inner = df.xs(key).swaplevel(i=i, j=j, axis=0)
            df = pd.concat([df.drop(key), pd.concat([inner], keys=[key])])
    else:
        df = df.swaplevel(i=i, j=j, axis=0)
    return df

@handle_transpose(multiindex='columns')
def cross_section(df, **kwargs):
    """Extracts a cross-section of a possibly-MultiIndexed DataFrame.

    Args:
        df (:class:`pandas.DataFrame`): DataFrame from which you want to extract a cross-section.
        kwargs (dict): dict of the form dict(level=value)

    Returns:
        :class:`pandas.DataFrame`: Cross-section of DataFrame with all-nan levels dropped.
    """
    drop_na = kwargs.pop('drop_na', True)
    drop_level = kwargs.pop('drop_level', True)
    if any(isinstance(value, (tuple, list)) and len(value) > 1 for value in kwargs.values()):
        drop_level = False
    levels = df.columns.names
    this_df = df.copy()
    for level, values in reversed(sorted(kwargs.items(), key=lambda x: levels.index(x[0]))):
        if not isinstance(values, (tuple, list)):
            values = [values]
        new_df = this_df.xs(values[0], level=level, axis=1, drop_level=drop_level)
        for value in values[1:]:
            new_df = new_df.join(this_df.xs(value, level=level, axis=1, drop_level=drop_level))
        this_df = new_df
    return drop_nan_levels(new_df) if drop_na else new_df

@handle_transpose(multiindex='columns')
def drop_nan_levels(df):
    """Drops all-nan levels in MultiIndexed DataFrame.

    Args:
        df (:class:`pandas.DataFrame`): DataFrame from which to drop all-nan levels.
    Returns:
        :class:`pandas.DataFrame`: DataFrame with all-nan levels dropped.
    """
    try: # df.columns is a MultiIndex
        for name, code in zip(df.columns.names, df.columns.codes):
            if all(c == -1 for c in code): # code == -1 means index is NaN
                df = df.droplevel(name, axis=1)
    except AttributeError: #df.columns is an Index
        pass
    return df

@handle_transpose(multiindex='columns')
def average_over_levels(df, levels_to_average_over):
    """Average numerical data over levels in `levels_to_average_over`.

    Args:
        df (:class:`pandas.DataFrame`): Our DataFrame.
        levels_to_average_over (optional, str | sequence[str]): Level or list of levels to average over.
    Returns:
        :class:`pandas.DataFrame`: DataFrame with numeric data averaged over requested levels.
    """
    if not isinstance(levels_to_average_over, (tuple, list)):
        levels_to_average_over = [levels_to_average_over]
    levels = list(set(df.columns.names) - set(levels_to_average_over))
    df = make_numeric(df)
    return df._get_numeric_data().mean(axis=1, level=levels)

@handle_transpose(multiindex='columns')
def flatten(df, sep='.'):
    """Flattens a MultiIndexed DataFrame such that the indices are strings separated by periods.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [sep.join([c for c in col if str(c) != 'nan']).strip() for col in df.columns.values]
    return df

def df_to_sqlite(df, table_name, conn=None, path_to_db=None, if_exists='replace', flat=True):
    """Writes MultiIndexed DataFrame `df` to a table in an sqlite database specified by `conn` or
    `path_to_db`.

    Args:
        df (:class:`pandas.DataFrame`): DataFrame you wish to write to sqlite database.
        table_name (str): Name of the resulting table in the database.
        conn (optional, sqlite connection): Open connection to sqlite database.
        path_to_df (optional, str): Path to sqlite database, used to open new connection
            if `conn` is None.
        if_exists (optional, str): What to do if `table_name` already exists in this database.
            Options: ('replace', 'fail', 'append'). Default: 'replace'.
    """
    import sqlite3
    if conn is None:
        if path_to_db is None:
            raise ValueError('Must provide either "conn" or "path_to_db".')
        conn = sqlite3.connect(path_to_db)
    if isinstance(df.index, pd.MultiIndex):
        df = df.T
    if flat:
        df = flatten(df)
    with conn:
        df.to_sql(table_name, conn, index_label=df.index.names, if_exists=if_exists)
        
