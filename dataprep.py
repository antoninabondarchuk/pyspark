import pyspark.sql.functions as f


def get_mins(dataframe):
    """
    Counts minimal values of all columns of DataFrame by aggregating.
    Args:
        dataframe (DataFrame): Spark DataFrame.

    Returns:
        DataFrame.
    """
    return dataframe.agg({column: 'min' for column in dataframe.columns})


def get_maxs(dataframe):
    """
    Counts maximal values of all columns of DataFrame by aggregating.
    Args:
        dataframe (DataFrame): Spark DataFrame.

    Returns:
        DataFrame.
    """
    return dataframe.agg({column: 'max' for column in dataframe.columns})


def hypercube(dataframe):
    """
    Codes given values in the whole DataFrame on interval [-1; 1].
    Using formula: xi = 2 * (( x - min(xi)) / (max(xi) - min(xi))) - 1
    Args:
        dataframe (DataFrame): Spark DataFrame with coded values of given columns.

    Returns:
        DataFrame.
    """
    mins_df = get_mins(dataframe)
    maxs_df = get_maxs(dataframe)
    full_df = dataframe.crossJoin(mins_df).crossJoin(maxs_df)
    for column in dataframe.columns:
        full_df = full_df.withColumn('xi_' + column,
                                     2 * ((f.col(column) - f.col(f'min({column})'))
                                          / (f.col(f'max({column})') - f.col(f'min({column})'))) - 1)
    xi_cols = [col for col in full_df.columns if col.startswith('xi_')]
    return full_df.select(*xi_cols)


def get_means(dataframe):
    """
    Calculates mean for every column in DataFrame.
    Args:
        dataframe (DataFrame): Spark DataFrame.

    Returns:
        Aggregated DataFrame.
    """
    return dataframe.agg({column: 'mean' for column in dataframe.columns})


def center(dataframe):
    """
    Centers data according to the mean of each column.
    Using formula: xk = x - mean(x)
    Args:
        dataframe: Spark DataFrame with centered values.

    Returns:
        DataFrame.
    """
    means_df = get_means(dataframe)
    full_df = dataframe.crossJoin(means_df)
    for column in dataframe.columns:
        full_df = full_df.withColumn('xk_' + column,
                                     f.when(f.col(column).isNotNull(),
                                            f.col(column) - f.col(f'avg({column})'))
                                     .otherwise(0.))
    xk_columns = [column for column in full_df.columns if column.startswith('xk_')]
    return full_df.select(*xk_columns)


def dataprep(dataframe):
    intervaled_df = hypercube(dataframe)
    centered_df = center(intervaled_df)
    return centered_df

