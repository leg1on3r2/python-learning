"""API для модели прогнозирования стоимости квартиры на FastAPI.

Запуск:
    uvicorn api:app --reload
    (адрес 127.0.0.1:8000/docs — интерактивная документация Swagger)

Модель и справочники загружаются из папки artifacts/ — это те же модели, что
обучены в работе 9, переиспользование разрешено условием задания.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from predict import UNKNOWN, load_artifacts, predict_price

app = FastAPI(
    title="Оценка стоимости квартиры",
    description="Модель прогнозирования стоимости квартиры по площади, числу комнат, "
                "этажу и локации. Модель — градиентный бустинг на логарифме цены.",
    version="1.0.0",
)


class FlatRequest(BaseModel):
    """Параметры квартиры для POST-запроса."""

    total_square: float = Field(..., gt=8, le=500, description="Площадь, м²", examples=[56.0])
    rooms: float = Field(2, ge=0, le=15, description="Число комнат, 0 — студия", examples=[2])
    floor: int = Field(7, ge=1, le=100, description="Этаж", examples=[7])
    city: str = Field("Москва", description="Город", examples=["Москва"])
    district: str = Field(UNKNOWN, description="Район", examples=["Тверской район"])


class PredictionResponse(BaseModel):
    """Ответ API: прогноз, диапазон и цена метра."""

    price: float = Field(..., description="Прогноз стоимости, руб.")
    price_per_square: float = Field(..., description="Прогноз цены метра, руб./м²")
    low: float = Field(..., description="Нижняя граница диапазона 10–90%, руб.")
    high: float = Field(..., description="Верхняя граница диапазона 10–90%, руб.")
    model: str = Field(..., description="Название модели, которой получен прогноз")


MODEL_NAME = "HistGradientBoostingRegressor на логарифме цены"


def predict(total_square: float, rooms: float, floor: int,
            city: str, district: str) -> PredictionResponse:
    """Общая логика для обоих методов: и GET, и POST приходят сюда."""
    _, _, _, meta = load_artifacts()

    # Значения из запроса могут быть незнакомыми модели: город или район берём
    # из справочника, иначе FastAPI пропустил бы запрос дальше и модель упала бы
    if city not in meta["cities"]:
        raise HTTPException(status_code=422, detail=f"Неизвестный город: {city}")
    if district not in meta["districts_by_city"].get(city, []):
        raise HTTPException(status_code=422,
                            detail=f"Район «{district}» не найден в городе {city}")

    result = predict_price(total_square, rooms, floor, city, district)
    return PredictionResponse(
        price=round(result["price"], 2),
        price_per_square=round(result["price_per_square"], 2),
        low=round(result["low"], 2),
        high=round(result["high"], 2),
        model=MODEL_NAME,
    )


@app.get("/health", summary="Liveness-проба", tags=["служебные"])
def health() -> dict:
    """Проверка живости сервиса.

    Отвечает «ok», если приложение поднялось И модель со справочниками читаются
    с диска: процесс без модели жив, но как API бесполезен, поэтому одной
    отметки «сервер запущен» для liveness-пробы мало.
    """
    model, _, _, meta = load_artifacts()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "rows_trained": meta["rows_trained"],
        "mape_percent": meta["metrics"]["MAPE, %"],
    }


@app.get("/", include_in_schema=False)
def root() -> dict:
    """Корень подсказывает, куда идти: пустая страница на «/» сбивает с толку."""
    return {
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/predict_get", "/predict_post"],
    }


@app.get("/predict_get", response_model=PredictionResponse, tags=["предсказание"],
         summary="Прогноз стоимости (GET)")
def predict_get(
    total_square: float = Query(..., gt=8, le=500, description="Площадь, м²", examples=[56.0]),
    rooms: float = Query(2, ge=0, le=15, description="Число комнат, 0 — студия"),
    floor: int = Query(7, ge=1, le=100, description="Этаж"),
    city: str = Query("Москва", description="Город"),
    district: str = Query(UNKNOWN, description="Район"),
) -> PredictionResponse:
    """Предсказание по параметрам в строке запроса."""
    return predict(total_square, rooms, floor, city, district)


@app.post("/predict_post", response_model=PredictionResponse, tags=["предсказание"],
          summary="Прогноз стоимости (POST)")
def predict_post(request: FlatRequest) -> PredictionResponse:
    """Предсказание по параметрам в теле запроса (JSON)."""
    return predict(request.total_square, request.rooms, request.floor,
                   request.city, request.district)
