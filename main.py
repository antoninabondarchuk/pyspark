from pyspark import SparkConf
from pyspark.sql import SparkSession
from PCA import apply_pca
from dataprep import dataprep
from oja import apply_oja
from reading import read_to_df, to_vectors

WINE_PATH = r'C:\Users\Tonya\PycharmProjects\research\resourсes\wine\wine.data'
WINE_NAMES = ('id', 'alcohol', 'malicAcid', 'ash', 'alcalinityOfAsh', 'magnesium',
              'totalPhenols', 'flavanoids', 'nonflavanoidPhenols', 'proanthocyanins',
              'colorIntensity', 'hue', 'od', 'proline')
BREAST_CANCER_PATH = r'C:\Users\Tonya\PycharmProjects\research\resourсes\breast_cancer\breast-cancer-wisconsin.data'
CANCER_NAMES = ('sampleCodeNumber', 'clumpThickness', 'uniformityOfCellSize',
                'uniformityOfCellShape', 'marginalAdhesion', 'singleEpithelialCellSize',
                'bareNuclei', 'blandChromatin', 'normalNucleoli', 'mitoses', 'class')
WINE_ROWS_NUM = 178
CANCES_ROWS_NUM = 699

spark_session = SparkSession.builder \
     .master("local") \
     .appName("research app") \
     .config(conf=SparkConf()) \
     .getOrCreate()


if __name__ == '__main__':
    wine_raw_df = read_to_df(spark_session, WINE_PATH, WINE_NAMES, exclude=['id', ])
    cancer_raw_df = read_to_df(spark_session, BREAST_CANCER_PATH, CANCER_NAMES,
                                   exclude=['sampleCodeNumber', 'class', ], null_value='?')

    wine_prep_df = dataprep(wine_raw_df)
    cancer_prep_df = dataprep(cancer_raw_df)
    wine_prep_df.show(178, False)

    wine_vector_df = to_vectors(wine_prep_df)
    cancer_vector_df = to_vectors(cancer_prep_df)
    wine_pca = apply_pca(wine_vector_df)
    cancer_pca = apply_pca(cancer_vector_df)
    wine_pca.show(100, False)
    cancer_pca.show(100, False)

    wine_oja_df = apply_oja(spark_session, wine_prep_df)
    wine_oja_df.show(178, False)
    cancer_oja_df = apply_oja(spark_session, cancer_prep_df)
    cancer_oja_df.show(100, False)

