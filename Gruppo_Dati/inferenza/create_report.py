#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm


# -------------------------
# CONFIGURAZIONE DEFAULT
# -------------------------

predictions_path = "inference_results/predictions_output.csv"
shap_pos_path    = "inference_results/shap_top10_positive.csv"
shap_neg_path    = "inferenza/inference_results/shap_top10_negative.csv"
shap_features_path = "inferenza/inference_results/shap_top30_features.csv"

os.makedirs("./outputs", exist_ok=True)

out_pdf  = "./outputs/report_jupyter.pdf"

top_k_conf = 3
shap_k = 5
id_col = None


# -------------------------
# FUNZIONI DI SUPPORTO
# -------------------------

def load_csv_if_exists(path):
    if path is None or not os.path.exists(path):
        print(f"[WARN] File non trovato: {path}")
        return None
    return pd.read_csv(path)


def detect_probability_columns(df):
    prob_cols = []
    for c in df.columns:
        lc = c.lower()
        if "prob" in lc or "conf" in lc or lc.startswith("p_") or lc.endswith("_p"):
            prob_cols.append(c)

    if len(prob_cols) == 0:
        num_cols = df.select_dtypes(include=[np.number]).columns
        prob_cols = [c for c in num_cols if df[c].between(0,1).all()]

    return prob_cols


def extract_top_k(df, prob_cols, k=3):
    rows = []
    for _, r in df.iterrows():
        sub = r[prob_cols]
        top = sub.sort_values(ascending=False).head(k)
        rows.append([(cls, float(val)) for cls, val in top.items()])
    return rows


def load_shap_long_format(df, k=5):
    if df is None:
        return [], []

    df_sorted = df.sort_values("shap_value", ascending=False)
    pos = df_sorted.head(k)

    df_sorted = df.sort_values("shap_value", ascending=True)
    neg = df_sorted.head(k)

    pos_list = list(zip(pos["feature"], pos["shap_value"], pos["feature_value"]))
    neg_list = list(zip(neg["feature"], neg["shap_value"], neg["feature_value"]))

    return pos_list, neg_list


def plot_shap_bar(data, title, filename):
    features = [d[0] for d in data]
    values = [d[1] for d in data]

    plt.figure(figsize=(8,4))
    plt.barh(features, values)
    plt.title(title)
    plt.grid(True, axis='x', linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# -------------------------
# GENERAZIONE REPORT
# -------------------------

def generate_report():

    print("Caricamento CSV...")
    pred_df = load_csv_if_exists(predictions_path)
    shap_pos = load_csv_if_exists(shap_pos_path)
    shap_neg = load_csv_if_exists(shap_neg_path)
    shap_feat = load_csv_if_exists(shap_features_path)

    if pred_df is None:
        raise ValueError("File predizioni mancante!")

    print("Rilevamento colonne probabilità...")
    prob_cols = detect_probability_columns(pred_df)
    print("Colonne trovate:", prob_cols)

    if id_col and id_col in pred_df.columns:
        ids = pred_df[id_col].astype(str).tolist()
    else:
        ids = [f"ROW_{i}" for i in pred_df.index]

    topk = extract_top_k(pred_df, prob_cols, top_k_conf)
    shap_pos_list, shap_neg_list = load_shap_long_format(shap_pos, shap_k)

    # grafici SHAP
    plot_shap_bar(shap_pos_list, "SHAP - Fattori Positivi", "./outputs/shap_positive.png")
    plot_shap_bar(shap_neg_list, "SHAP - Fattori Negativi", "./outputs/shap_negative.png")

    # PDF
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Report Predizioni e Spiegazioni SHAP</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    for idx, row in pred_df.iterrows():

        rid = ids[idx]

        story.append(Paragraph("<b>Predizione finale:</b>", styles["Heading3"]))

        pred_col = [c for c in pred_df.columns if "pred" in c.lower()]
        pred_val = row[pred_col[0]] if pred_col else "(N/A)"
        story.append(Paragraph(f"Predizione finale: <b>{pred_val}</b>", styles["Normal"]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Top 3 Categorie:</b>", styles["Heading3"]))
        story.append(Paragraph(
            f"1) {row['CATEGORIA_top1']} — {row['CATEGORIA_top1_conf']:.4f}<br/>"
            f"2) {row['CATEGORIA_top2']} — {row['CATEGORIA_top2_conf']:.4f}<br/>"
            f"3) {row['CATEGORIA_top3']} — {row['CATEGORIA_top3_conf']:.4f}",
            styles["Normal"]
        ))
        story.append(Spacer(1,6))

        story.append(Paragraph("<b>Top 3 Classi:</b>", styles["Heading3"]))
        story.append(Paragraph(
            f"1) {row['CLASSE_top1']} — {row['CLASSE_top1_conf']:.4f}<br/>"
            f"2) {row['CLASSE_top2']} — {row['CLASSE_top2_conf']:.4f}<br/>"
            f"3) {row['CLASSE_top3']} — {row['CLASSE_top3_conf']:.4f}",
            styles["Normal"]
        ))
        story.append(Spacer(1,12))

        story.append(Paragraph("SHAP - Fattori Positivi", styles["Heading3"]))
        story.append(Paragraph(
            "Quanto segue rappresenta i 5 fattori che hanno avvicinato maggiormente "
            "(in accordo con predizione) il modello alla predizione:", 
            styles["Normal"])
        )
        story.append(Image("./outputs/shap_positive.png", width=14*cm, height=5*cm))

        story.append(Paragraph("SHAP - Fattori Negativi", styles["Heading3"]))
        story.append(Paragraph(
            "Quanto segue rappresenta i 5 fattori che hanno allontanato il modello dalla predizione:",
            styles["Normal"])
        )
        story.append(Image("./outputs/shap_negative.png", width=14*cm, height=5*cm))

        story.append(Spacer(1,20))

    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    doc.build(story)

    print(f"✔️ PDF creato: {out_pdf}")


# -------------------------
# AVVIO DA TERMINALE
# -------------------------

if __name__ == "__main__":
    generate_report()