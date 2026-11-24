import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def load_events(file_path: str) -> list[dict]:
    """Загружает события из JSON-файла.

    Учитывает особенность формата: в ключах и значениях могут быть
    лишние пробелы ('events ', 'timestamp ', 'signature ').
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path.absolute()}")

    if path.stat().st_size == 0:
        raise ValueError(f"Файл пуст: {path.absolute()}")

    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Нормализуем ключи: убираем пробелы по краям
    events = []
    for event in raw_data.get("events ", raw_data.get("events", [])):
        normalized = {k.strip(): str(v).strip() for k, v in event.items()}
        events.append(normalized)

    return events


def validate_events(events: list[dict]) -> None:
    """Проверяет корректность загруженных данных."""
    required_keys = {"timestamp", "signature"}

    for i, event in enumerate(events):
        missing = required_keys - event.keys()
        if missing:
            raise ValueError(f"В событии #{i} отсутствуют поля: {missing}")
        if not event["timestamp"] or not event["signature"]:
            raise ValueError(f"В событии #{i} пустое значение поля")

    print(f"[OK] Валидация пройдена. Загружено событий: {len(events)}")


def create_dataframe(events: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(events)

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def analyze_signature_distribution(
    df: pd.DataFrame,
    print_report: bool = False,
) -> Dict[str, Any]:
    """Анализирует распределение событий по типу подписи (signature).

    Args:
        df: DataFrame с колонкой 'signature'.
        print_report: если True — выводит текстовый отчёт в консоль.

    Returns:
        Словарь с данными для визуализации:
            - 'summary'         : DataFrame с колонками 'count', 'percentage'
                                  (индекс — название сигнатуры, отсортирован
                                  по убыванию частоты)
            - 'total_events'    : int, общее число событий
            - 'unique_count'    : int, число уникальных сигнатур
            - 'top5'            : DataFrame с топ-5 сигнатурами
    """
    if not df.empty and "signature" not in df.columns:
        raise ValueError(
            "DataFrame не содержит обязательную колонку 'signature'"
        )

    empty_result: Dict[str, Any] = {
        "summary": pd.DataFrame(columns=["count", "percentage"]),
        "total_events": 0,
        "unique_count": 0,
        "top5": pd.DataFrame(columns=["count", "percentage"]),
    }

    if df.empty:
        if print_report:
            print("DataFrame пуст, анализ невозможен.")
        return empty_result

    total = len(df)
    unique_count = df["signature"].nunique()

    distribution = df["signature"].value_counts()
    distribution_pct = df["signature"].value_counts(normalize=True) * 100

    summary = pd.DataFrame({
        "count": distribution,
        "percentage": distribution_pct.round(2),
    })
    summary.index.name = "signature"

    top5 = summary.head(5)

    if print_report:
        print("=" * 70)
        print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ СОБЫТИЙ ПО ТИПАМ СИГНАТУР (signature)")
        print("=" * 70)
        print(f"Всего событий        : {total}")
        print(f"Уникальных сигнатур  : {unique_count}")
        print("-" * 70)
        print("\nПолное распределение:")
        print(summary.rename(columns={"count": "Количество",
                                      "percentage": "Доля, %"}).to_string())
        print("\n" + "-" * 70)
        print("Топ-5 наиболее частых типов подписей:")
        print("-" * 70)
        print(top5.rename(columns={"count": "Количество",
                                   "percentage": "Доля, %"}).to_string())

    return {
        "summary": summary,
        "total_events": total,
        "unique_count": unique_count,
        "top5": top5,
    }


def analyze_visualization_example(
    df: pd.DataFrame,
    show: bool = True,
) -> None:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="signature")
    plt.title("Распределение типов событий безопасности")
    plt.xticks(rotation=90)
    if show:
        plt.show()


def analyze_visualization_diagram(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (12, 8),
    print_report: bool = False,
    save_path: str | None = None,
    # [fix 6] добавлен параметр show
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Строит круговую диаграмму распределения событий по сигнатурам.

    Args:
        df: DataFrame с колонкой 'signature'.
        figsize: размер фигуры (ширина, высота) в дюймах.
        print_report: если True - выводит текстовый отчёт в консоль.
        save_path: если указан - сохраняет график в файл (png, pdf, svg).
        show: если True - отображает график через plt.show().

    Returns:
        Кортеж (fig, ax) для дальнейшей работы с графиком.
    """
    result = analyze_signature_distribution(df, print_report=print_report)
    summary = result["summary"]
    total = result["total_events"]

    if summary.empty:
        print("[!] Нет данных для построения графика.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Нет данных",
                ha="center", va="center", fontsize=16)
        ax.axis("off")
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)

    explode = [0.05 if i == 0 else 0.0 for i in range(len(summary))]

    colors = sns.color_palette("Set3", n_colors=len(summary))

    wedges, _, autotexts = ax.pie(
        summary["count"],
        explode=explode,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    plt.setp(autotexts, size=10, weight="bold")

    ax.legend(
        wedges,
        summary.index,
        title="Типы сигнатур",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    ax.set_title(
        f"Распределение событий по сигнатурам (всего: {total})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[i] График сохранён: {path.absolute()}")

    if show:
        plt.show()

    return fig, ax


def main() -> None:
    data_file = os.path.join("data", "events.json")

    print(f"Загрузка данных из: {os.path.abspath(data_file)}")
    events = load_events(data_file)
    validate_events(events)

    df = create_dataframe(events)
    print(f"\nDataFrame создан. Форма: {df.shape}")
    print(f"Колонки: {list(df.columns)}")
    print(f"Типы данных:\n{df.dtypes}")


    while True:
        change = input("Выберите, какой вариант диаграммы отразить после анализа файла? (1 - как в примере или 2 - круговая диаграмма): ")
        
        if change == "1":
            analyze_visualization_example(df)
            break
        elif change == "2":
            analyze_visualization_diagram(
                df,
                print_report=True,
                save_path="output/signature_distribution.png", 
            )
            break
        else:
            print("Пожалуйста, введите либо 1, либо 2.")


if __name__ == "__main__":
    main()