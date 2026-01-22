from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QWidget, QTabWidget, QTextEdit,
    QGroupBox, QGridLayout, QScrollArea, QMessageBox,
    QFileDialog
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QBrush, QFont


class PredictionDialog(QDialog):
    """
    Dialog per visualizzare le predizioni del modello (CATEGORIA + CLASSE) 
    e la spiegazione SHAP.
    """
    
    def __init__(self, predizioni, shap_explanation=None, inference_results_dir=None, parent=None):
        super().__init__(parent)
        self.predizioni = predizioni
        self.shap_explanation = shap_explanation
        self.inference_results_dir = inference_results_dir
        self.init_ui()
    
    def init_ui(self):
        """Inizializza l'interfaccia utente"""
        self.setWindowTitle("Classamento Immobiliare - Predizioni AI")
        self.setMinimumWidth(700)
        self.setMinimumHeight(550)
        
        layout = QVBoxLayout()
        
        # Titolo
        title = QLabel("<h2>🏠 Predizione Classamento Immobiliare</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Box risultato principale
        if self.predizioni:
            pred_principale = self.predizioni[0]
            cat = pred_principale.get("categoria", "N/A")
            classe = pred_principale.get("classe", "N/A")
            prob_cat = pred_principale.get("probabilita_cat", pred_principale.get("probabilita", 0))
            prob_classe = pred_principale.get("probabilita_classe", 0)
            
            result_box = QGroupBox("Predizione Principale")
            result_layout = QGridLayout()
            
            # Predizione finale grande
            final_label = QLabel(f"<h1 style='color: #2196F3;'>{cat}/{classe}</h1>")
            final_label.setAlignment(Qt.AlignCenter)
            result_layout.addWidget(final_label, 0, 0, 1, 2)
            
            # Probabilità
            result_layout.addWidget(QLabel(f"<b>CATEGORIA:</b> {cat}"), 1, 0)
            result_layout.addWidget(QLabel(f"<b>Confidenza:</b> {prob_cat:.1%}"), 1, 1)
            result_layout.addWidget(QLabel(f"<b>CLASSE:</b> {classe}"), 2, 0)
            result_layout.addWidget(QLabel(f"<b>Confidenza:</b> {prob_classe:.1%}"), 2, 1)
            
            result_box.setLayout(result_layout)
            result_box.setStyleSheet("QGroupBox { font-weight: bold; padding: 15px; }")
            layout.addWidget(result_box)
        
        # Tab widget per predizioni e SHAP
        tabs = QTabWidget()
        
        # Tab 1: Predizioni
        tab_pred = QWidget()
        tab_pred_layout = QVBoxLayout()
        
        desc = QLabel(
            "Top 2 predizioni per CATEGORIA e CLASSE.\n"
            "1a: Verde sgargiante | 2a: Arancione"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        tab_pred_layout.addWidget(desc)
        
        # Tabella predizioni
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["#", "Categoria", "Prob. Cat.", "Classe", "Prob. Classe"])
        self.table.setRowCount(len(self.predizioni))
        
        for i, pred in enumerate(self.predizioni):
            # Rank
            rank_item = QTableWidgetItem(f"{i+1}")
            rank_item.setFlags(Qt.ItemIsEnabled)
            color = self.get_preview_color(pred.get("probabilita_cat", pred.get("probabilita", 0)), i)
            rank_item.setBackground(QBrush(color))
            self.table.setItem(i, 0, rank_item)
            
            # Categoria
            cat_item = QTableWidgetItem(pred.get("categoria", "N/A"))
            cat_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 1, cat_item)
            
            # Prob Categoria
            prob_cat = pred.get("probabilita_cat", pred.get("probabilita", 0))
            prob_cat_item = QTableWidgetItem(f"{prob_cat:.1%}")
            prob_cat_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 2, prob_cat_item)
            
            # Classe
            classe_item = QTableWidgetItem(str(pred.get("classe", "N/A")))
            classe_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 3, classe_item)
            
            # Prob Classe
            prob_classe = pred.get("probabilita_classe", 0)
            prob_classe_item = QTableWidgetItem(f"{prob_classe:.1%}")
            prob_classe_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 4, prob_classe_item)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        tab_pred_layout.addWidget(self.table)
        tab_pred.setLayout(tab_pred_layout)
        tabs.addTab(tab_pred, "📊 Predizioni")
        
        # Tab 2: Spiegazione SHAP
        tab_shap = QWidget()
        tab_shap_layout = QVBoxLayout()
        
        if self.shap_explanation:
            shap_desc = QLabel(
                f"<b>Spiegazione locale per CATEGORIA = {self.shap_explanation.get('predicted_class', 'N/A')}</b><br>"
                "Le feature positive aumentano la probabilità, quelle negative la diminuiscono."
            )
            shap_desc.setWordWrap(True)
            shap_desc.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 5px;")
            tab_shap_layout.addWidget(shap_desc)
            
            # Features positive
            pos_box = QGroupBox("Top Features che AUMENTANO la probabilità")
            pos_layout = QVBoxLayout()
            pos_text = QTextEdit()
            pos_text.setReadOnly(True)
            pos_content = ""
            for feat in self.shap_explanation.get('top_positive', [])[:5]:
                pos_content += f"• <b>{feat['feature']}</b>: {feat['feature_value']:.2f} → SHAP: <span style='color:green;'>+{feat['shap_value']:.4f}</span><br>"
            pos_text.setHtml(pos_content if pos_content else "Nessuna feature positiva significativa")
            pos_text.setMaximumHeight(120)
            pos_layout.addWidget(pos_text)
            pos_box.setLayout(pos_layout)
            tab_shap_layout.addWidget(pos_box)
            
            # Features negative
            neg_box = QGroupBox("Top Features che DIMINUISCONO la probabilità")
            neg_layout = QVBoxLayout()
            neg_text = QTextEdit()
            neg_text.setReadOnly(True)
            neg_content = ""
            for feat in self.shap_explanation.get('top_negative', [])[:5]:
                neg_content += f"• <b>{feat['feature']}</b>: {feat['feature_value']:.2f} → SHAP: <span style='color:red;'>{feat['shap_value']:.4f}</span><br>"
            neg_text.setHtml(neg_content if neg_content else "Nessuna feature negativa significativa")
            neg_text.setMaximumHeight(120)
            neg_layout.addWidget(neg_text)
            neg_box.setLayout(neg_layout)
            tab_shap_layout.addWidget(neg_box)
        else:
            no_shap = QLabel("⚠️ Spiegazione SHAP non disponibile.\nInstalla il pacchetto 'shap' per abilitarla.")
            no_shap.setAlignment(Qt.AlignCenter)
            no_shap.setStyleSheet("padding: 20px; color: #666;")
            tab_shap_layout.addWidget(no_shap)
        
        tab_shap.setLayout(tab_shap_layout)
        tabs.addTab(tab_shap, "🔍 Spiegazione SHAP")
        
        layout.addWidget(tabs)
        
        # Pulsanti
        button_layout = QHBoxLayout()
        
        self.button_applica = QPushButton("✓ Applica Colorazione")
        self.button_applica.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 12px; font-weight: bold; border-radius: 5px; font-size: 14px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        button_layout.addWidget(self.button_applica)
        
        self.button_reset = QPushButton("↺ Reset")
        self.button_reset.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "padding: 12px; font-weight: bold; border-radius: 5px; font-size: 14px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        button_layout.addWidget(self.button_reset)
        
        # Bottone Genera Report PDF
        self.button_report = QPushButton("📄 Genera Report PDF")
        self.button_report.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 12px; font-weight: bold; border-radius: 5px; font-size: 14px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.button_report.clicked.connect(self.genera_report_pdf)
        # Abilita solo se ci sono risultati salvati
        self.button_report.setEnabled(self.inference_results_dir is not None)
        button_layout.addWidget(self.button_report)
        
        button_chiudi = QPushButton("Chiudi")
        button_chiudi.clicked.connect(self.close)
        button_chiudi.setStyleSheet("QPushButton { padding: 12px; border-radius: 5px; font-size: 14px; }")
        button_layout.addWidget(button_chiudi)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_preview_color(self, probabilita, categoria_index=0):
        """Restituisce un colore di anteprima in base alla probabilità."""
        if categoria_index == 0:
            r, g, b = 50, 255, 50
            alpha = int(20 + (probabilita ** 2 * 235))
        else:
            r, g, b = 255, 165, 0
            alpha = int(80 + (probabilita ** 0.5 * 175))
        
        alpha = max(20, min(255, alpha))
        return QColor(r, g, b, alpha)
    
    def get_predizioni(self):
        """Restituisce la lista delle predizioni"""
        return self.predizioni
    
    def genera_report_pdf(self):
        """Genera un report PDF con le predizioni e le spiegazioni SHAP."""
        if not self.inference_results_dir:
            QMessageBox.warning(
                self,
                "Errore",
                "Nessun risultato disponibile per generare il report."
            )
            return
        
        # Chiedi dove salvare il PDF
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Salva Report PDF",
            "report_classamento.pdf",
            "File PDF (*.pdf)"
        )
        
        if not output_path:
            return  # Utente ha annullato
        
        try:
            from .report_generator import ReportGenerator
            
            generator = ReportGenerator(self.inference_results_dir)
            generator.generate_report(output_path)
            
            QMessageBox.information(
                self,
                "Report Generato",
                f"Report PDF salvato con successo:\n{output_path}"
            )
        except ImportError as e:
            error_msg = str(e)
            if 'reportlab' in error_msg.lower():
                QMessageBox.critical(
                    self,
                    "Dipendenze Mancanti",
                    "Per generare il report PDF è necessario installare 'reportlab'.\n\n"
                    "Esegui nella console Python di QGIS:\n"
                    "import subprocess, sys\n"
                    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'reportlab'])"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Errore Import",
                    f"Errore nell'importazione:\n{error_msg}"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Errore",
                f"Errore nella generazione del report:\n{str(e)}"
            )
