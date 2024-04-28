import polars as pl


def weighted_rating(count, avg, quantile: pl.DataFrame, mean: pl.DataFrame):
    quantile = float(quantile.head(1).row(0)[0])
    return (count / (count + quantile) * avg) + (quantile / (quantile + count) * mean)


def get_content_simple_recs(movies_metadata: pl.DataFrame, head_count: int) -> pl.DataFrame:
    """
    Функция, возвращающие общие рекомендации по фильмам, без уточняющих характеристик.
    :param movies_metadata:
    :param head_count: Количество фильмов, которые надо возвратить в итоговом дата фрейме
    :return:
    """
    # Получаем среднее значение оценок под фильмами
    mean: pl.DataFrame = movies_metadata.select('vote_average').mean()
    # Агрегируем столбцы этого DataFrame до квантиля 0.90.
    quantile: pl.DataFrame = movies_metadata.select('vote_count').quantile(0.90)
    # Получаем дата фрейм из значений количества голосов больше квантиля
    q_movies: pl.DataFrame = movies_metadata.filter(pl.col('vote_count') >= quantile)
    # Дописываем столбец со взвешенным рейтингом, название столбца - 'score'
    q_movies: pl.DataFrame = q_movies.with_columns(
        weighted_rating(
            pl.col('vote_count'),
            pl.col('vote_average'),
            quantile,
            mean
        ).alias('Оценка алгоритма'),
        pl.col('title').alias('Название'),
        pl.col('vote_count').alias('Количество оценок'),
        pl.col('vote_average').alias('Средняя оценка')
    )
    # Сортируем дата фрейм по убыванию взвешенного рейтинга
    q_movies: pl.DataFrame = q_movies.sort('Оценка алгоритма', descending=True)
    return q_movies[['Название', 'Количество оценок', 'Средняя оценка', 'Оценка алгоритма']].head(head_count)
