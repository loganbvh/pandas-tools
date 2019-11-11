# pandas-tools

Using [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) with datasets that are nested/not table-like requires the use of [`pandas.MultiIndex`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html), which is a powerful and (to me) somewhat unintuitive construct.

This is a collection of functions I have found useful for creating and manipulating `MultiIndexed` `DataFrames`. The particular use case for these tools is datasets composed of many observations ("records"), where the data from each observation has some nested structure. See [demo.ipynb](demo.ipynb) for an example. 

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
