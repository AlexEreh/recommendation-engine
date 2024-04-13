import polars as pl
from polars import DataFrame

from content_cosine import get_content_cosine_recs
from content_simple import get_content_simple_recs


def main():
    movies_df: DataFrame = (pl.read_csv('dataset/movies_metadata.csv', infer_schema_length=36000, n_rows=30000)
                            .select(pl.col("title", "overview", "vote_average", "vote_count", "genres")))
    print(get_content_simple_recs(movies_df, 10))
    print(get_content_cosine_recs(movies_df, 'The Dark Knight Rises').head(10))


if __name__ == '__main__':
    main()
