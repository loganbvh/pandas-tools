# pandas-tools

Simple functions for manipulating [DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), especially those with a [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html).

Many of these functions rely on the decorator `handle_transpose`, which modifies a function of the form `func(df, *args, **kwargs) -> new_df`, where `df` is a `pandas.DataFrame` with either `df.columns` or `df.index` being a `pandas.MultiIndex`. `handle_transpose` allows you to write such a `func` for only one of these cases (`df.columns` is a `pandas.MultiIndex` or `df.index` is a `pandas.MultiIndex`), while ensuring that both `func(df, *args, **kwargs) -> new_df` and `func(df.T, *args, **kwargs) -> new_df.T` operate as expected.

```python
def handle_transpose(multiindex='columns'):

    """Handles operations on a MultiIndexed DataFrame regardless of orientation (i.e. transposed or not).
    
    Args:
        multiindex (optional, str ('columns' | 'index')): Orientation of the DataFrame
          assumed in the wrapped func, i.e. columns as MultiIndex or index as MultiIndex.
    """
    ...
```
