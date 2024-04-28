import polars as pl
from polars import DataFrame

from collaborative_knn import get_collaborative_knn_recs
from content_keywords import get_content_keywords_recs
from content_tfidf import get_content_tfidf_recs
from content_simple import get_content_simple_recs


def main():
    movies_df: DataFrame = (pl.read_csv('dataset/movies_metadata.csv', infer_schema_length=100000)
                            .select(pl.col("title", "overview", "vote_average", "vote_count", "genres")))
    user_ratings_df: DataFrame = (pl.read_csv('dataset/ratings_small.csv')
                                  .select(pl.col("userId", "movieId", "rating")))
    #print("---Результат работы простейшего алгоритма---")
    #print(get_content_simple_recs(movies_df, 10))
    #print("---Результат работы алгоритма на базе TF-IDF---")
    #print(get_content_tfidf_recs(movies_df, 'The Dark Knight Rises').head(10))
    #print("---Результат работы алгоритма на базе заготовленных ключевых слов---")
    #print(get_content_keywords_recs('The Dark Knight Rises').head(10))

    print(get_collaborative_knn_recs(movies_df.to_pandas(), user_ratings_df.to_pandas(), 'Jumanji'))


if __name__ == '__main__':
    main()
