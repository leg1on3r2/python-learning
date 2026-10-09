"""Загрузка и очистка датасета объявлений о продаже недвижимости.

Архив train_data.zip ищется в папке DataSet или data рядом с проектом либо
выше по дереву каталогов, поэтому один и тот же код работает и локально,
и в Colab, и на сервере.
"""

from pathlib import Path
from zipfile import BadZipFile

import pandas as pd

ARCHIVE_NAME = "train_data.zip"
CSV_NAME = "realty_data.csv"

# Столбец description не читаем: он единственный тяжёлый в файле (десятки
# мегабайт текста) и для модели на табличных признаках бесполезен
COLUMNS = ["price", "lat", "lon", "total_square", "rooms", "floor",
           "city", "district"]

TARGET = "price"
NUMERIC = ["total_square", "rooms", "floor", "lat", "lon"]
CATEGORICAL = ["city", "district"]
FEATURES = NUMERIC + CATEGORICAL

# Границы отсечения выбросов: единичные объекты за сотни миллионов рублей
# и «квартиры» на тысячи метров портят обучение, а их меньше процента
MIN_PRICE = 1_000_000
MAX_PRICE = 300_000_000
MIN_SQUARE = 8
MAX_SQUARE = 500
MAX_FLOOR = 100

UNKNOWN = "не указан"


def find_archive(filename: str = ARCHIVE_NAME) -> Path:
    """Ищет архив с данными в папках DataSet/ и data/ вверх по дереву каталогов."""
    start = Path(__file__).resolve().parent
    for folder in [start, *start.parents]:
        for subfolder in ("DataSet", "data"):
            candidate = folder / subfolder / filename
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Не найден архив {filename}. Положите его в папку DataSet рядом с проектом."
    )


def load(path: Path | None = None) -> pd.DataFrame:
    """Читает csv прямо из zip-архива, не распаковывая его на диск."""
    if path is None:
        path = find_archive()

    try:
        return pd.read_csv(path, usecols=COLUMNS, compression="zip")
    except (BadZipFile, ValueError):
        # На случай, если архив уже распакован и рядом лежит обычный csv
        csv_path = path.with_name(CSV_NAME)
        if not csv_path.exists():
            raise
        return pd.read_csv(csv_path, usecols=COLUMNS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает строки, непригодные для обучения, и чинит категориальные пропуски."""
    before = len(df)

    # Без цены, площади и координат объявление для модели бесполезно
    df = df.dropna(subset=[TARGET, "total_square", "lat", "lon"])

    df = df[df[TARGET].between(MIN_PRICE, MAX_PRICE)]
    df = df[df["total_square"].between(MIN_SQUARE, MAX_SQUARE)]
    df = df[df["floor"].between(1, MAX_FLOOR)]

    df = df.copy()
    df["rooms"] = df["rooms"].fillna(0)          # 0 — студия или число комнат не указано
    df["floor"] = df["floor"].astype("int16")

    # Пропуск в городе или районе — это «в объявлении не указано», а не потеря
    # данных: такие строки ещё пригодны для обучения, поэтому помечаем их
    # отдельной категорией, а не выбрасываем
    for column in CATEGORICAL:
        df[column] = df[column].fillna(UNKNOWN).astype(str).astype("category")

    print(f"после очистки осталось {len(df)} строк из {before} "
          f"({len(df) / before * 100:.1f}%)")
    return df[COLUMNS[:1] + FEATURES]
