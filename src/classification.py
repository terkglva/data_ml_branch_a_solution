import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-яәіңғүұқөһ\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_categories(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_keyword_patterns(categories: dict) -> dict[str, re.Pattern]:
    patterns = {}
    for code, info in categories.items():
        parts = [re.escape(normalize_text(keyword)) for keyword in info["keywords"]]
        patterns[code] = re.compile("|".join(parts), flags=re.IGNORECASE)
    return patterns


def keyword_predict_one(text: object, patterns: dict[str, re.Pattern]) -> tuple[str, float]:
    normalized = normalize_text(text)
    scores = []
    for code, pattern in patterns.items():
        matches = pattern.findall(normalized)
        if matches:
            score = len(matches) + sum(len(match) for match in matches) / 100
            scores.append((score, code))
    if not scores:
        return "UNKNOWN", 0.0
    scores.sort(reverse=True)
    total = sum(score for score, _ in scores)
    return scores[0][1], float(scores[0][0] / total)


def keyword_predict(texts, categories: dict) -> pd.DataFrame:
    patterns = build_keyword_patterns(categories)
    rows = [keyword_predict_one(text, patterns) for text in texts]
    return pd.DataFrame(rows, columns=["pred_keyword", "confidence_keyword"])


def make_weak_training_data(categories: dict) -> pd.DataFrame:
    templates = [
        "{kw}", "оплата {kw}", "услуги {kw}", "счет за {kw}", "договор на {kw}",
        "поставка {kw}", "акт по {kw}", "накладная {kw}", "закуп {kw}",
        "{name}", "оплата: {name}", "услуги: {name}", "договор: {name}",
    ]
    rows = []
    for code, info in categories.items():
        for keyword in info["keywords"]:
            for template in templates[:9]:
                rows.append((template.format(kw=keyword, name=info["name"]), code))
        for template in templates[9:]:
            rows.append((template.format(kw="", name=info["name"]), code))
    return pd.DataFrame(rows, columns=["text", "category_code"])


def build_ml_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=1000, solver="liblinear", C=3.0, class_weight="balanced")
                ),
            ),
        ]
    )


def predict_with_confidence(model: Pipeline, texts, labels: list[str] | None = None) -> pd.DataFrame:
    proba = model.predict_proba(texts)
    best_idx = np.argmax(proba, axis=1)
    class_labels = list(model.named_steps["clf"].classes_)
    return pd.DataFrame({
        "pred_ml": [class_labels[i] for i in best_idx],
        "confidence_ml": proba[np.arange(len(best_idx)), best_idx],
    })


def evaluate_predictions(y_true, y_pred, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    return report_df, cm


def top_confusions(cm: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    rows = []
    for true_code in cm.index:
        for pred_code in cm.columns:
            count = int(cm.loc[true_code, pred_code])
            if true_code != pred_code and count > 0:
                rows.append({"true_code": true_code, "pred_code": pred_code, "count": count})
    if not rows:
        return pd.DataFrame(columns=["true_code", "pred_code", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(n)
