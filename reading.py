from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.types as t
import pyspark.sql.functions as f

INPUT_FEATURES_NAME = 'features'


def read_to_df(spark_session, file_path, names=[], exclude=[], null_value=None):
    """
    Reading from csv file to DataFrame without titles and making string schema for it.
    Args:
        exclude (iter): list of columns' names to exclude.
        names (list): whole list of columns' names.
        null_value (str): char for missing values
        file_path (str): path to csv file with data.

    Returns:
        DataFrame.
    """
    schema = t.StructType([t.StructField(column, t.DoubleType(), True) for column in names])
    return (spark_session.read.csv(file_path,
                                   header=False,
                                   schema=schema,
                                   mode="PERMISSIVE",
                                   nullValue=null_value).drop(*exclude))


def to_vectors(dataframe):
    """
    Converts all columns of input DataFrame to one-columned DataFrame.
    Args:
        dataframe (DataFrame): Spark DataFrame with all numerical columns.
    Returns:
        DataFrame.
    """
    assembler = VectorAssembler(inputCols=dataframe.columns,
                                outputCol=INPUT_FEATURES_NAME)
    output = assembler.transform(dataframe)
    return output.select(INPUT_FEATURES_NAME)
