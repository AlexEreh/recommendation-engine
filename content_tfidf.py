import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def get_content_tfidf_recs(movies_metadata: pl.DataFrame, title: str) -> pl.DataFrame:
    tfidf: TfidfVectorizer = TfidfVectorizer(stop_words='english')
    movies_metadata: pl.DataFrame = movies_metadata.with_columns(
        pl.col('overview').fill_null('')
    ).drop('vote_average', 'vote_count')
    print(movies_metadata)
    overview_series: pl.Series = movies_metadata.select('overview').to_series()

    # Составляем матрицу TF-IDF
    from scipy.sparse import csr_matrix
    tfidf_matrix: csr_matrix = tfidf.fit_transform(overview_series)
    cosine_sim: np.ndarray = linear_kernel(tfidf_matrix, tfidf_matrix)
    movies_metadata: pl.DataFrame = movies_metadata.with_row_index()
    # Получаем индекс фильма, название которого совпадает с заданным
    expr: pl.Expr = pl.all_horizontal(
        pl.col('title') == title
    )
    idx = movies_metadata.row(by_predicate=expr, named=True)['index']
    # Получаем попарную схожесть всех фильмов с фильмом, который нам дан
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Сортируем фильмы на основании очков схожести
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Получаем очки для 10 самых похожих фильмов
    sim_scores = sim_scores[1:11]

    # Получаем индексы фильмов
    movie_indices = [i[0] for i in sim_scores]
    return movies_metadata.select('title')[movie_indices]
