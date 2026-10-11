"""Обучение модели прогнозирования стоимости квартиры.

Запуск:
    python train_model.py

Результат — файлы в папке artifacts/: модель, две модели квантилей (для
диапазона прогноза) и справочники для интерфейса.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import data

RANDOM_STATE = 42
TEST_SIZE = 0.2
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def build_model(loss: str = "squared_error", quantile: float | None = None):
    """Градиентный бустинг со встроенной поддержкой категориальных признаков.

    One-hot здесь не подходит: районов больше сотни, и матрица признаков
    раздулась бы в десятки раз. HistGradientBoosting умеет работать
    с категориями напрямую, если столбец имеет тип category.
    """
    params = dict(
        max_iter=400,
        learning_rate=0.08,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        random_state=RANDOM_STATE,
    )
    if quantile is None:
        return HistGradientBoostingRegressor(loss=loss, **params)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **params)


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Метрики считаем в рублях: в логарифмах они неинтерпретируемы."""
    errors = np.abs(actual - predicted)
    return {
        "MAE, руб": int(round(mean_absolute_error(actual, predicted))),
        "медианная ошибка, руб": int(round(median_absolute_error(actual, predicted))),
        "MAPE, %": round(float(np.mean(errors / actual) * 100), 2),
        "R2": round(float(r2_score(actual, predicted)), 4),
    }


def main() -> None:
    started = time.time()
    raw = data.load()
    print(f"прочитано строк: {len(raw)}")
    df = data.clean(raw)

    X = df[data.FEATURES]
    # Цены скошены вправо на два порядка: без логарифма ошибку определяли бы
    # единичные дорогие объекты, а типовые квартиры предсказывались бы плохо
    y = np.log(df[data.TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    actual = np.exp(y_test.to_numpy())
    print(f"обучающая выборка: {len(X_train)}, тестовая: {len(X_test)}")

    model = build_model()
    model.fit(X_train, y_train)
    predicted = np.exp(model.predict(X_test))
    metrics = evaluate(actual, predicted)

    print("\nМодель — качество на отложенной выборке:")
    for name, value in metrics.items():
        print(f"  {name:<24} {value:,}".replace(",", " "))

    # Бейзлайн: медианная цена квадратного метра, умноженная на площадь.
    # Без него непонятно, дала ли модель что-то сверх простого правила
    price_per_square = (np.exp(y_train) / X_train["total_square"]).median()
    baseline_metrics = evaluate(actual, X_test["total_square"].to_numpy() * price_per_square)

    print("\nБейзлайн (медианная цена за м² × площадь):")
    for name, value in baseline_metrics.items():
        print(f"  {name:<24} {value:,}".replace(",", " "))

    # Две дополнительные модели на тех же признаках предсказывают 10-й и 90-й
    # проценты цены. Так интерфейс показывает честный диапазон, а не «плюс-минус
    # средняя ошибка»: у дорогих квартир ошибка в рублях в разы больше
    low_model = build_model(quantile=0.1)
    high_model = build_model(quantile=0.9)
    for quantile_model in (low_model, high_model):
        quantile_model.fit(X_train, y_train)

    low = np.exp(low_model.predict(X_test))
    high = np.exp(high_model.predict(X_test))
    coverage = float(np.mean((actual >= low) & (actual <= high)))
    print(f"\nДиапазон 10–90%: фактическая цена попала внутрь в {coverage * 100:.1f}% случаев")

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib", compress=3)
    joblib.dump(low_model, ARTIFACTS / "quantile_low.joblib", compress=3)
    joblib.dump(high_model, ARTIFACTS / "quantile_high.joblib", compress=3)

    # Справочники для интерфейса: списки городов и районов, типичные координаты
    # (модель обучена на широте и долготе, а вводить их вручную неудобно)
    # и медианная цена метра по району — чтобы было с чем сравнить прогноз
    coordinates = (df.groupby(["city", "district"], observed=True)[["lat", "lon"]]
                     .median().round(6).reset_index())
    price_by_district = (df.assign(price_per_square=df[data.TARGET] / df["total_square"])
                           .groupby(["city", "district"], observed=True)["price_per_square"]
                           .median().round(0))

    meta = {
        "features": data.FEATURES,
        "categories": {column: list(df[column].cat.categories) for column in data.CATEGORICAL},
        "cities": sorted(df["city"].cat.categories),
        "districts_by_city": {
            city: sorted(group["district"].tolist())
            for city, group in coordinates.groupby("city", observed=True)
        },
        "coordinates": {f"{row.city}||{row.district}": [row.lat, row.lon]
                        for row in coordinates.itertuples()},
        "district_price_per_square": {f"{city}||{district}": int(value)
                                      for (city, district), value in price_by_district.items()},
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "interval_coverage": round(coverage, 4),
        "rows_trained": int(len(X_train)),
        "rows_total": int(len(df)),
        "price_range": [int(df[data.TARGET].min()), int(df[data.TARGET].max())],
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    size = sum(f.stat().st_size for f in ARTIFACTS.glob("*.joblib")) / 1024**2
    print(f"\nсохранено в artifacts/: модели ({size:.1f} МБ) и meta.json")
    print(f"обучение заняло {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
