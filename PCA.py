from pyspark import SparkConf
from pyspark.ml.feature import PCA


INPUT_FEATURES_NAME = 'features'
OUTPUT_FEATURES_NAME = 'Outfeatures'


def apply_pca(dataframe, components_num=3):
    pca = PCA(k=components_num, inputCol=INPUT_FEATURES_NAME, outputCol=OUTPUT_FEATURES_NAME)
    model = pca.fit(dataframe)  # learning
    result = model.transform(dataframe).select(OUTPUT_FEATURES_NAME)

    return result
