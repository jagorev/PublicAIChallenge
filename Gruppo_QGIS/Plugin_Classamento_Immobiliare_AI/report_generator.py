#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo per la generazione del report PDF con predizioni e spiegazioni SHAP.

Richiede:
- reportlab
- matplotlib
- pandas
"""

import os
import pandas as pd
import numpy as np

# Matplotlib senza GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Verifica disponibilità reportlab
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ReportGenerator:
    """
    Genera report PDF con predizioni e spiegazioni SHAP.
    """
    
    def __init__(self, inference_results_dir: str):
        """
        Args:
            inference_results_dir: Cartella con i file CSV dei risultati
        """
        self.results_dir = inference_results_dir
        self.predictions_path = os.path.join(inference_results_dir, "predictions_output.csv")
        self.shap_pos_path = os.path.join(inference_results_dir, "shap_top10_positive.csv")
        self.shap_neg_path = os.path.join(inference_results_dir, "shap_top10_negative.csv")
        self.shap_top30_path = os.path.join(inference_results_dir, "shap_top30_features.csv")
        
        self.temp_dir = os.path.join(inference_results_dir, "temp_images")
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def _load_csv(self, path: str) -> pd.DataFrame:
        """Carica un CSV se esiste."""
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
    
    def _plot_shap_bar(self, data: pd.DataFrame, title: str, filename: str, color: str = 'steelblue'):
        """Crea un grafico a barre orizzontali per i valori SHAP."""
        if data is None or data.empty:
            return None
        
        # Prendi le prime 5 righe
        data = data.head(5)
        
        features = data['feature'].tolist()
        values = data['shap_value'].tolist()
        
        plt.figure(figsize=(10, 4))
        bars = plt.barh(features[::-1], values[::-1], color=color)
        
        # Colora le barre in base al segno
        for bar, val in zip(bars, values[::-1]):
            if val > 0:
                bar.set_color('#4CAF50')  # Verde
            else:
                bar.set_color('#f44336')  # Rosso
        
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel('SHAP Value')
        plt.grid(True, axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        
        filepath = os.path.join(self.temp_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_report(self, output_path: str) -> bool:
        """
        Genera il report PDF.
        
        Args:
            output_path: Path del file PDF da generare
            
        Returns:
            True se il report è stato generato con successo
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab non è installato. "
                "Installa con: pip install reportlab"
            )
        
        # Carica i dati
        pred_df = self._load_csv(self.predictions_path)
        if pred_df is None:
            raise FileNotFoundError(f"File predizioni non trovato: {self.predictions_path}")
        
        shap_pos = self._load_csv(self.shap_pos_path)
        shap_neg = self._load_csv(self.shap_neg_path)
        
        # Genera grafici SHAP
        shap_pos_img = None
        shap_neg_img = None
        
        if shap_pos is not None:
            shap_pos_img = self._plot_shap_bar(
                shap_pos, 
                "Fattori che AUMENTANO la probabilità", 
                "shap_positive.png"
            )
        
        if shap_neg is not None:
            shap_neg_img = self._plot_shap_bar(
                shap_neg,
                "Fattori che DIMINUISCONO la probabilità",
                "shap_negative.png"
            )
        
        # Crea il PDF
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Titolo
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=30
        )
        story.append(Paragraph("Report Classamento Immobiliare AI", title_style))
        story.append(Spacer(1, 12))
        
        # Predizione finale
        row = pred_df.iloc[0]
        final_pred = row.get('final_prediction', 'N/A')
        
        # Estrai i dati per il testo descrittivo
        cat_top1 = row.get('CATEGORIA_top1', 'N/A')
        cat_top1_conf = row.get('CATEGORIA_top1_conf', 0)
        cat_top2 = row.get('CATEGORIA_top2', 'N/A')
        cat_top2_conf = row.get('CATEGORIA_top2_conf', 0)
        classe_top1 = row.get('CLASSE_top1', 'N/A')
        classe_top1_conf = row.get('CLASSE_top1_conf', 0)
        classe_top2 = row.get('CLASSE_top2', 'N/A')
        classe_top2_conf = row.get('CLASSE_top2_conf', 0)
        
        # Testo descrittivo dei risultati
        intro_style = ParagraphStyle(
            'IntroText',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            spaceAfter=12,
            alignment=4  # Justified
        )
        
        intro_text = f"""
        Il modello di intelligenza artificiale ha analizzato le caratteristiche dell'unità immobiliare 
        fornita e ha prodotto una stima del classamento catastale. 
        <br/><br/>
        La <b>categoria catastale</b> più probabile risulta essere <b>{cat_top1}</b> con una 
        confidenza del <b>{cat_top1_conf:.1%}</b>. La seconda categoria più probabile è 
        <b>{cat_top2}</b> con una confidenza del <b>{cat_top2_conf:.1%}</b>.
        <br/><br/>
        Per quanto riguarda la <b>classe</b>, il modello stima come più probabile la classe 
        <b>{classe_top1}</b> con una confidenza del <b>{classe_top1_conf:.1%}</b>, seguita dalla 
        classe <b>{classe_top2}</b> con una confidenza del <b>{classe_top2_conf:.1%}</b>.
        <br/><br/>
        La predizione finale combinata è quindi <b>{final_pred}</b>.
        <br/><br/>
        Di seguito sono riportati i dettagli completi delle predizioni e l'analisi dei fattori 
        che hanno maggiormente influenzato la decisione del modello (spiegazione SHAP).
        """
        
        story.append(Paragraph(intro_text.strip(), intro_style))
        story.append(Spacer(1, 20))
        
        # Separatore
        story.append(Paragraph("<hr/>", styles['Normal']))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("<b>PREDIZIONE FINALE</b>", styles['Heading2']))
        story.append(Spacer(1, 6))
        
        pred_style = ParagraphStyle(
            'Prediction',
            parent=styles['Normal'],
            fontSize=24,
            textColor=colors.HexColor('#2196F3'),
            alignment=1  # Center
        )
        story.append(Paragraph(f"<b>{final_pred}</b>", pred_style))
        story.append(Spacer(1, 20))
        
        # Top 3 Categorie
        story.append(Paragraph("<b>Top 3 Categorie</b>", styles['Heading3']))
        cat_data = []
        for i in range(1, 4):
            cat_col = f'CATEGORIA_top{i}'
            conf_col = f'CATEGORIA_top{i}_conf'
            if cat_col in row and conf_col in row:
                cat_data.append([
                    f"{i}.",
                    str(row[cat_col]),
                    f"{row[conf_col]:.2%}"
                ])
        
        if cat_data:
            cat_table = Table(cat_data, colWidths=[1*cm, 4*cm, 3*cm])
            cat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e3f2fd')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ]))
            story.append(cat_table)
        story.append(Spacer(1, 15))
        
        # Top 3 Classi
        story.append(Paragraph("<b>Top 3 Classi</b>", styles['Heading3']))
        classe_data = []
        for i in range(1, 4):
            classe_col = f'CLASSE_top{i}'
            conf_col = f'CLASSE_top{i}_conf'
            if classe_col in row and conf_col in row:
                classe_data.append([
                    f"{i}.",
                    str(row[classe_col]),
                    f"{row[conf_col]:.2%}"
                ])
        
        if classe_data:
            classe_table = Table(classe_data, colWidths=[1*cm, 4*cm, 3*cm])
            classe_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff3e0')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ]))
            story.append(classe_table)
        story.append(Spacer(1, 25))
        
        # Sezione SHAP
        story.append(Paragraph("<b>SPIEGAZIONE DELLA DECISIONE (SHAP)</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Testo descrittivo SHAP
        shap_intro_style = ParagraphStyle(
            'ShapIntro',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            spaceAfter=12,
            alignment=4  # Justified
        )
        
        shap_intro = """
        
        """
        story.append(Paragraph(shap_intro.strip(), shap_intro_style))
        story.append(Spacer(1, 10))
        
        # Costruisci testo descrittivo con le feature più importanti
        if shap_pos is not None and not shap_pos.empty:
            top_pos_features = shap_pos.head(3)
            pos_description = "Le caratteristiche che hanno maggiormente <font color='green'><b>contribuito positivamente</b></font> alla predizione sono: "
            pos_items = []
            for _, feat in top_pos_features.iterrows():
                feat_name = feat['feature']
                feat_value = feat.get('feature_value', 'N/A')
                shap_val = feat['shap_value']
                if isinstance(feat_value, float):
                    pos_items.append(f"<b>{feat_name}</b> (valore: {feat_value:.2f}, contributo: +{shap_val:.4f})")
                else:
                    pos_items.append(f"<b>{feat_name}</b> (valore: {feat_value}, contributo: +{shap_val:.4f})")
            pos_description += ", ".join(pos_items) + "."
            story.append(Paragraph(pos_description, shap_intro_style))
            story.append(Spacer(1, 8))
        
        if shap_neg is not None and not shap_neg.empty:
            top_neg_features = shap_neg.head(3)
            neg_description = "Le caratteristiche che hanno maggiormente <font color='red'><b>contribuito negativamente</b></font> alla predizione sono: "
            neg_items = []
            for _, feat in top_neg_features.iterrows():
                feat_name = feat['feature']
                feat_value = feat.get('feature_value', 'N/A')
                shap_val = feat['shap_value']
                if isinstance(feat_value, float):
                    neg_items.append(f"<b>{feat_name}</b> (valore: {feat_value:.2f}, contributo: {shap_val:.4f})")
                else:
                    neg_items.append(f"<b>{feat_name}</b> (valore: {feat_value}, contributo: {shap_val:.4f})")
            neg_description += ", ".join(neg_items) + "."
            story.append(Paragraph(neg_description, shap_intro_style))
            story.append(Spacer(1, 8))
        
        # Interpretazione generale
        if shap_pos is not None and shap_neg is not None:
            interpretation = """
            
            """
            story.append(Paragraph(interpretation.strip(), shap_intro_style))
        
        story.append(Spacer(1, 15))
        
        # Grafico fattori positivi
        if shap_pos_img and os.path.exists(shap_pos_img):
            story.append(Paragraph("<b>Fattori Positivi</b>", styles['Heading3']))
            story.append(Paragraph(
                "Caratteristiche che hanno <font color='green'>aumentato</font> "
                "la probabilità della categoria predetta:",
                styles['Normal']
            ))
            story.append(Spacer(1, 6))
            story.append(Image(shap_pos_img, width=14*cm, height=5*cm))
            story.append(Spacer(1, 15))
        
        # Grafico fattori negativi
        if shap_neg_img and os.path.exists(shap_neg_img):
            story.append(Paragraph("<b>Fattori Negativi</b>", styles['Heading3']))
            story.append(Paragraph(
                "Caratteristiche che hanno <font color='red'>diminuito</font> "
                "la probabilità della categoria predetta:",
                styles['Normal']
            ))
            story.append(Spacer(1, 6))
            story.append(Image(shap_neg_img, width=14*cm, height=5*cm))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=1
        )
        story.append(Paragraph(
            "Report generato automaticamente dal plugin Classamento Immobiliare AI",
            footer_style
        ))
        
        # Genera il PDF
        doc.build(story)
        
        # Pulizia file temporanei
        self._cleanup_temp_files()
        
        return True
    
    def _cleanup_temp_files(self):
        """Rimuove i file temporanei."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def generate_report_from_results(results_dir: str, output_pdf: str) -> bool:
    """
    Funzione di convenienza per generare un report PDF.
    
    Args:
        results_dir: Cartella con i risultati dell'inferenza
        output_pdf: Path del PDF da generare
        
    Returns:
        True se generato con successo
    """
    generator = ReportGenerator(results_dir)
    return generator.generate_report(output_pdf)
