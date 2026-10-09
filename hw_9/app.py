"""Интерфейс для прогнозирования стоимости квартиры.

Запуск:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from predict import UNKNOWN, load_artifacts, predict_price

st.set_page_config(page_title="Оценка стоимости квартиры", page_icon="🏙")


def money(value: float) -> str:
    """Форматирует сумму с пробелами между разрядами: 12 345 678 ₽."""
    return f"{value:,.0f}".replace(",", " ")


_, _, _, meta = load_artifacts()

st.title("🏙 Оценка стоимости квартиры")
st.caption(
    f"Модель обучена на {meta['rows_total']:,} объявлениях о продаже квартир "
    "в Москве и Московской области.".replace(",", " ")
)

# --- Ввод признаков --------------------------------------------------------
# Всё, что вводит пользователь, собрано в боковой панели: так основная область
# остаётся под результат, и на телефоне форма не растягивается на два экрана

with st.sidebar:
    st.header("Параметры квартиры")

    city = st.selectbox(
        "Город",
        meta["cities"],
        index=meta["cities"].index("Москва") if "Москва" in meta["cities"] else 0,
        help="Населённый пункт, в котором находится квартира",
    )

    # Список районов зависит от города: показывать районы Химок при выбранной
    # Москве бессмысленно
    districts = meta["districts_by_city"].get(city, [UNKNOWN])
    district = st.selectbox("Район", districts,
                            help="Если район неизвестен, оставьте «не указан»")

    total_square = st.number_input("Площадь, м²", min_value=8.0, max_value=500.0,
                                   value=56.0, step=1.0)
    rooms = st.number_input("Комнат", min_value=0, max_value=15, value=2, step=1,
                            help="0 — студия или число комнат не указано")
    floor = st.number_input("Этаж", min_value=1, max_value=100, value=7, step=1)

    st.divider()
    st.caption("Модель — градиентный бустинг на площади, числе комнат, этаже, "
               "координатах, городе и районе.")

area_left, area_right = st.columns([2, 1])
with area_left:
    st.write("Выберите параметры слева и нажмите кнопку — прогноз появится здесь.")
with area_right:
    calculate = st.button("Рассчитать стоимость", type="primary", use_container_width=True)

# --- Результат по нажатию кнопки ------------------------------------------

if calculate:
    result = predict_price(total_square, rooms, floor, city, district)

    st.divider()
    columns = st.columns(3)
    columns[0].metric("Прогноз стоимости", f"{money(result['price'])} ₽")
    columns[1].metric("Цена за метр", f"{money(result['price_per_square'])} ₽/м²")

    # Диапазон считают две отдельные модели на 10-й и 90-й проценты: у дорогих
    # квартир ошибка в рублях в разы больше, и «плюс-минус средняя ошибка»
    # вводил бы в заблуждение
    spread = (result["high"] - result["low"]) / result["price"] * 100
    columns[2].metric("Разброс прогноза", f"{spread:.0f}%")

    st.info(f"Диапазон, в который цена попадает в 8 случаях из 10: "
            f"**{money(result['low'])} — {money(result['high'])} ₽**")

    # Сравнение со средней ценой метра по выбранному району: один прогноз без
    # ориентира ничего не говорит о том, дорого это или дёшево
    reference = result["district_price_per_square"]
    if reference:
        difference = (result["price_per_square"] / reference - 1) * 100
        if abs(difference) < 1:
            st.caption(f"Медиана по локации «{city}, {district}» — {money(reference)} ₽/м²: "
                       "прогноз практически совпадает с ней.")
        else:
            sign = "дороже" if difference > 0 else "дешевле"
            st.caption(f"Медиана по локации «{city}, {district}» — {money(reference)} ₽/м². "
                       f"Прогноз на {abs(difference):.0f}% {sign} этой медианы.")

    left, right = st.columns([1, 1])
    with left:
        st.write("**Параметры расчёта**")
        st.table(pd.DataFrame({
            "Признак": ["Площадь, м²", "Комнат", "Этаж", "Город", "Район"],
            "Значение": [f"{total_square:g}", "студия" if rooms == 0 else f"{rooms:g}",
                         f"{floor:g}", city, district],
        }))
    with right:
        st.write("**Расположение**")
        st.map(pd.DataFrame({"lat": [result["lat"]], "lon": [result["lon"]]}), zoom=10)
        st.caption("Точка — типичное расположение объявлений выбранной локации.")

# --- Справка о качестве модели --------------------------------------------

with st.expander("Качество модели"):
    metrics, baseline = meta["metrics"], meta["baseline_metrics"]
    rows = ["Средняя ошибка (MAE)", "Медианная ошибка",
            "Средняя ошибка в процентах (MAPE)", "R²"]
    st.table(pd.DataFrame({
        "Метрика": rows,
        "Модель": [f"{money(metrics['MAE, руб'])} ₽", f"{money(metrics['медианная ошибка, руб'])} ₽",
                   f"{metrics['MAPE, %']}%", str(metrics["R2"])],
        "Бейзлайн": [f"{money(baseline['MAE, руб'])} ₽",
                     f"{money(baseline['медианная ошибка, руб'])} ₽",
                     f"{baseline['MAPE, %']}%", str(baseline["R2"])],
    }))
    st.caption(
        f"Метрики посчитаны на отложенной выборке, которую модель не видела при обучении. "
        f"Бейзлайн — медианная цена метра, умноженная на площадь. "
        f"Фактическая цена попала в показанный диапазон в "
        f"{meta['interval_coverage'] * 100:.0f}% случаев."
    )
