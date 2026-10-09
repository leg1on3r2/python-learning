"""Получение предсказания обученной моделью.

Модуль намеренно отделён от интерфейса: предсказание нужно и Streamlit,
и консоли, а логика должна быть одна.

Из консоли:
    python predict.py --square 56 --rooms 2 --floor 7 --city Москва --district "Тверской район"
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
UNKNOWN = "не указан"


@lru_cache(maxsize=1)
def load_artifacts() -> tuple:
    """Читает модель, модели квантилей и справочники.

    Результат кешируется: Streamlit выполняет скрипт заново после каждого
    действия пользователя, и без кеша модели читались бы с диска каждый раз.
    """
    if not (ARTIFACTS / "model.joblib").exists():
        raise FileNotFoundError("Модель не найдена. Сначала обучите её: python train_model.py")

    model = joblib.load(ARTIFACTS / "model.joblib")
    low_model = joblib.load(ARTIFACTS / "quantile_low.joblib")
    high_model = joblib.load(ARTIFACTS / "quantile_high.joblib")
    meta = json.loads((ARTIFACTS / "meta.json").read_text(encoding="utf-8"))
    return model, low_model, high_model, meta


def coordinates_for(city: str, district: str) -> tuple[float, float]:
    """Координаты точки для города и района — медиана объявлений из данных.

    Модель обучена в том числе на широте и долготе: цена квартиры сильно
    зависит от расположения. Вводить координаты вручную неудобно, поэтому
    подставляем типичную точку выбранного района.
    """
    _, _, _, meta = load_artifacts()
    point = meta["coordinates"].get(f"{city}||{district}")
    if point is None:
        # района нет в справочнике — берём медиану по городу, а если и города нет,
        # то по всем объявлениям
        city_points = [value for key, value in meta["coordinates"].items()
                       if key.startswith(f"{city}||")]
        point = (np.median(np.array(city_points), axis=0).tolist() if city_points
                 else np.median(np.array(list(meta["coordinates"].values())), axis=0).tolist())
    return float(point[0]), float(point[1])


def make_features(total_square: float, rooms: float, floor: int, city: str,
                  district: str, lat: float | None = None,
                  lon: float | None = None) -> pd.DataFrame:
    """Собирает одну строку признаков ровно в том виде, в каком её ждёт модель."""
    _, _, _, meta = load_artifacts()
    if lat is None or lon is None:
        lat, lon = coordinates_for(city, district)

    row = pd.DataFrame([{
        "total_square": float(total_square),
        "rooms": float(rooms),
        "floor": int(floor),
        "lat": float(lat),
        "lon": float(lon),
        "city": city,
        "district": district,
    }])[meta["features"]]

    # Категории должны быть того же типа и с тем же набором значений, что при
    # обучении: модель обучена с categorical_features="from_dtype" и понимает,
    # что столбец категориальный, именно по типу
    for column, categories in meta["categories"].items():
        row[column] = pd.Categorical(row[column], categories=categories)
    return row


def predict_price(total_square: float, rooms: float, floor: int, city: str,
                  district: str = UNKNOWN, lat: float | None = None,
                  lon: float | None = None) -> dict:
    """Возвращает прогноз стоимости, диапазон и цену метра."""
    model, low_model, high_model, meta = load_artifacts()
    features = make_features(total_square, rooms, floor, city, district, lat, lon)

    # Модели обучены на логарифме цены — возвращаем значения в рублях
    price = float(np.exp(model.predict(features)[0]))
    low = float(np.exp(low_model.predict(features)[0]))
    high = float(np.exp(high_model.predict(features)[0]))

    return {
        "price": price,
        "low": min(low, price),
        "high": max(high, price),
        "price_per_square": price / float(total_square),
        "district_price_per_square": meta["district_price_per_square"]
                                      .get(f"{city}||{district}"),
        "lat": float(features["lat"].iloc[0]),
        "lon": float(features["lon"].iloc[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогноз стоимости квартиры")
    parser.add_argument("--square", type=float, required=True, help="площадь, м²")
    parser.add_argument("--rooms", type=float, default=2, help="число комнат")
    parser.add_argument("--floor", type=int, default=5, help="этаж")
    parser.add_argument("--city", default="Москва", help="город")
    parser.add_argument("--district", default=UNKNOWN, help="район")
    args = parser.parse_args()

    result = predict_price(args.square, args.rooms, args.floor, args.city, args.district)
    money = lambda value: f"{value:,.0f}".replace(",", " ")   # noqa: E731

    print(f"{args.square:g} м², {args.rooms:g} комн., этаж {args.floor}, {args.city}, {args.district}")
    print(f"прогноз:          {money(result['price'])} руб.")
    print(f"диапазон 10–90%:  {money(result['low'])} — {money(result['high'])} руб.")


if __name__ == "__main__":
    main()
