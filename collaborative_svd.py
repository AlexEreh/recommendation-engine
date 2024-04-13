import polars as pl
from polars import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def get_collaborative_svd_recs(movies_metadata: DataFrame, user_ratings: DataFrame, title: str):
    pass
