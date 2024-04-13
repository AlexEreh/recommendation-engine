import polars as pl
from polars import DataFrame


def weighted_rating(count, avg, quantile: DataFrame, mean: DataFrame):
    quantile = float(quantile.head(1).row(0)[0])
    return (count / (count + quantile) * avg) + (quantile / (quantile + count) * mean)


def get_content_simple_recs(movies_metadata: DataFrame, head_count: int) -> DataFrame:
    """
    Функция, возвращающие общие рекомендации по фильмам, без уточняющих характеристик.
    :param movies_metadata:
    :param head_count: количество фильмов, которые надо возвратить в итоговом датафрейме
    :return:
    """
    # Получаем среднее значение оценок под фильмами
    mean: DataFrame = movies_metadata.select('vote_average').mean()
    # Агрегирем столбцы этого DataFrame до их квантильного значения.
    quantile: DataFrame = movies_metadata.select('vote_count').quantile(0.90)
    # Получаем датафрейм из значений количества голосов больше квантиля
    q_movies: DataFrame = movies_metadata.filter(pl.col('vote_count') >= quantile)
    # Дописываем столбец со взвешенным рейтингом, название столбца - 'score'
    q_movies = q_movies.with_columns(
        weighted_rating(
            pl.col('vote_count'),
            pl.col('vote_average'),
            quantile,
            mean
        ).alias('score')
    )
    # Сортируем датафрейм по убыванию взвешенного рейтинга
    q_movies: DataFrame = q_movies.sort('score', descending=True)
    return q_movies[['title', 'vote_count', 'vote_average', 'score']].head(head_count)
