from functools import reduce
from random import uniform
import pyspark.sql.types as t
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.functions import pandas_udf, PandasUDFType

FIRST_W_FLAG = 1


def normalize(dataframe):
    """
    Normalize each column of DataFrame using the following formula:
    wn = w / ||w||, where ||w|| = sqrt(sum(w**2)).
    Read more: https://en.wikipedia.org/wiki/Norm_(mathematics)
    Args:
        dataframe: Spark DataFrame with columns of numerical values.

    Returns:
        DataFrame.
    """
    aggregations = [((f.sum(pow(f.col(column), 2)))**0.5).alias('norm_' + column)
                    for column in dataframe.columns]
    norms_df = dataframe.agg(*aggregations)
    full_df = dataframe.crossJoin(norms_df)
    for column in dataframe.columns:
        full_df = (full_df.withColumn(column,
                                      f.col(column) / f.col('norm_' + column)))
    normalized_columns = [column for column in full_df.columns if not column.startswith('norm_')]
    return full_df.select(*normalized_columns)


def get_w0(spark_session, dataframe):
    """
    Counts first row of c coefficients randomly, where c = len(dataframe.columns),
    converts to vector and applies norm on it.
    Args:
        dataframe: Spark DataFrame for calculating c.

    Returns:
        One-columned Dataframe.
    """
    schema = t.StructType((t.StructField('w0', t.DoubleType(), False), ))
    data = [(uniform(-1., 1.), ) for _ in range(len(dataframe.columns))]
    w0_raw_df = spark_session.createDataFrame(data, schema=schema)
    w0_normalized = normalize(w0_raw_df)
    return w0_normalized


def get_new(row, dataframe, rows_num):
    dataframe = dataframe.withColumn('y', f.when(f.col('id') == row,
                                                 sum([f.col(dataframe.columns[i]) * f.col('w').getItem(i)
                                                      for i in range(len(dataframe.columns[:-3]))]))
                                     .when(f.col('id') < row, f.col('y'))
                                     .otherwise(None))

    w_list = [f.col('w').getItem(i) for i in range(len(dataframe.columns[:-3]))]
    dataframe = dataframe.withColumn('w',
                                     f.array([(w_list[i]
                                               + (f.col('y')/rows_num)
                                               * (f.col(dataframe.columns[:-3][i]) - f.col('y') * w_list[i]))
                                              for i in range(len(w_list))]))

    return dataframe


def apply_oja(spark_session, dataframe):
    w0_df = get_w0(spark_session, dataframe)
    w_list = w0_df.select(f.collect_list('w0').alias('w'))
    dataframe_w = dataframe.crossJoin(w_list)
    window = Window.orderBy('w')
    ready_df = (dataframe_w.withColumn('id', f.row_number().over(window))
                .withColumn('y', f.lit(0.)))

    @pandas_udf(ready_df.schema, PandasUDFType.GROUPED_MAP)
    def count_wy(pdf):
        rows_num = len(pdf)
        for i in range(0, rows_num - 1):
            if i == 0:
                pdf.at[i, 'w'] = list(pdf.at[i, 'w'])

            pdf.at[i, 'y'] = sum([pdf.at[i, 'w'][wi] * pdf.at[i, pdf.columns[wi]]
                                  for wi in range(len(pdf.at[i, 'w']))])

            w_norm = pow(sum([pdf.at[i, 'w'][wi] + (pdf.at[i, 'y'] / rows_num)
                              * (pdf.at[i, pdf.columns[wi]]
                                 - pdf.at[i, 'y'] * pdf.at[i, 'w'][wi])
                              for wi in range(len(pdf.at[i, 'w']))]), 0.5)

            pdf.at[i + 1, 'w'] = [pdf.at[i, 'w'][wi] / w_norm for wi in range(len(pdf.at[i, 'w']))]

        pdf.at[rows_num - 1, 'y'] = sum([pdf.at[rows_num - 1, 'w'][wi] * pdf.at[rows_num - 1, pdf.columns[wi]]
                                         for wi in range(len(pdf.at[rows_num - 1, 'w']))])
        return pdf

    wys_df = ready_df.groupby().apply(count_wy)

    @pandas_udf(wys_df.schema, PandasUDFType.GROUPED_MAP)
    def count_xij(pdf):
        rows_num = len(pdf)
        for i in range(rows_num):
            for xi in range(pdf.at[rows_num - 1, 'w'].size):
                pdf.at[i, pdf.columns[xi]] = pdf.at[i, pdf.columns[xi]] \
                                             - (pdf.at[i, 'y']
                                                * pdf.at[rows_num - 1, 'w'].item(xi))
        return pdf

    result_df = wys_df.groupby().apply(count_xij)

    return result_df
