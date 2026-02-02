import os
from qgis.PyQt.QtWidgets import (
    QAction, QMessageBox, QFileDialog, QDialog, 
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from qgis.PyQt.QtGui import QIcon, QColor
from qgis.PyQt.QtCore import Qt
from qgis.core import (
    QgsProject, 
    QgsFeatureRequest, 
    QgsExpression,
    QgsSymbol,
    QgsFillSymbol,
    QgsRuleBasedRenderer
)
from qgis.utils import iface

from .prediction_dialog import PredictionDialog
from .model_inference import ModelInference


class WelcomeDialog(QDialog):
    """
    Dialog semplice che chiede all'utente di selezionare un file XML (Docfa).
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Classamento Immobiliare AI")
        self.setMinimumWidth(400)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)
        
        msg = QLabel("Seleziona il file XML (Docfa) contenente i dati dell'immobile da classificare.")
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        layout.addWidget(msg)
        
        # Bottoni
        button_layout = QHBoxLayout()
        
        self.btn_cancel = QPushButton("Annulla")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        button_layout.addStretch()
        
        self.btn_continue = QPushButton("Seleziona File...")
        self.btn_continue.setDefault(True)
        self.btn_continue.clicked.connect(self.accept)
        button_layout.addWidget(self.btn_continue)
        
        layout.addLayout(button_layout)


class ColorazioneClassi:
    """
    Plugin QGIS per colorare particelle in base alle classi predette dal modello AI.
    """
    
    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        self.action = None
        self.plugin_dir = os.path.dirname(__file__)
        self.layer_name = "catasto_fabbricati"
        self.original_renderer = None
        self.model_inference = ModelInference()
        self.predizioni_correnti = []
        self.shap_explanation = None
        self.inference_results_dir = None
    
    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        self.action = QAction(
            QIcon(icon_path), 
            "Classamento Immobiliare AI", 
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Classamento AI", self.action)
    
    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&Classamento AI", self.action)
    
    def run(self):
        """Esegue il plugin"""
        # Verifica che il layer esista
        layer_list = QgsProject.instance().mapLayersByName(self.layer_name)
        if not layer_list:
            QMessageBox.critical(None, "Errore", f"Il layer '{self.layer_name}' non è caricato nel progetto.")
            return
        layer = layer_list[0]
        
        welcome = WelcomeDialog()
        if welcome.exec_() != QDialog.Accepted:
            return
        
        xml_path, _ = QFileDialog.getOpenFileName(
            None,
            "📂 Seleziona il file XML Docfa",
            "",
            "File XML (*.xml);;Tutti i file (*.*)"
        )
        
        if not xml_path:
            QMessageBox.information(None, "Operazione annullata", "Nessun file selezionato.")
            return
        
        try:
            # Chiamata aggiornata per passare il percorso XML
            self.predizioni_correnti, self.shap_explanation, self.inference_results_dir = self.esegui_inferenza(xml_path)
        except Exception as e:
            QMessageBox.critical(None, "Errore nell'inferenza", f"Impossibile eseguire l'inferenza:\n{str(e)}")
            return
        
        if not self.predizioni_correnti:
            QMessageBox.warning(None, "Attenzione", "Nessuna predizione generata dal modello.")
            return
        
        self.dialog = PredictionDialog(
            self.predizioni_correnti, 
            self.shap_explanation,
            self.inference_results_dir
        )
        self.dialog.button_applica.clicked.connect(lambda: self.applica_colorazione(layer))
        self.dialog.button_reset.clicked.connect(lambda: self.reset_colorazione(layer))
        self.dialog.show()
    
    def esegui_inferenza(self, xml_path):
        """
        Gestisce l'orchestrazione dell'inferenza sul file XML.
        """
        # Cartella output per report (stessa del file input)
        base_dir = os.path.dirname(xml_path)
        results_dir = os.path.join(base_dir, "inference_results")
        
        # che gestisce parsing, predizione, shap e salvataggio
        result = self.model_inference.run_full_analysis(
            xml_path=xml_path, 
            output_dir=results_dir
        )
        
        if "error" in result:
            raise ValueError(result["error"])
            
        return result['predizioni'], result['shap_data'], results_dir
    
    def applica_colorazione(self, layer):
        predizioni = self.dialog.get_predizioni()
        if not predizioni: return
        
        if self.original_renderer is None:
            self.original_renderer = layer.renderer().clone()
        
        # Le predizioni sono già ordinate
        top_predizioni = predizioni[:2]
        root_rule = QgsRuleBasedRenderer.Rule(None)

        for idx, pred in enumerate(top_predizioni):
            categoria = pred["categoria"]
            classe = pred.get("classe", "N/A")
            prob_cat = pred.get("probabilita_cat", 0)
            prob_classe = pred.get("probabilita_classe", 0)

            color = self.get_color_for_rank(idx, prob_cat)
            symbol = QgsFillSymbol.createSimple({
                'color': color.name(),
                'outline_color': 'black',
                'outline_width': '0.26'
            })
            symbol.setColor(color)

            expression = f'"CATEGORIA" = \'{categoria}\''
            rule = QgsRuleBasedRenderer.Rule(symbol)
            rule.setFilterExpression(expression)
            rule.setLabel(f"{categoria}/{classe} (Cat:{prob_cat:.0%})")
            root_rule.appendChild(rule)

        layer.setRenderer(QgsRuleBasedRenderer(root_rule))
        layer.triggerRepaint()
        self.iface.messageBar().pushSuccess("Successo", f"Colorazione applicata per {top_predizioni[0]['categoria']}.")
    
    def get_color_for_rank(self, rank, probabilita):
        if rank == 0:
            r, g, b = 50, 255, 50
            alpha = int(20 + (probabilita ** 2 * 235))
        else:
            r, g, b = 255, 165, 0
            alpha = int(80 + (probabilita ** 0.5 * 175))
        return QColor(r, g, b, max(20, min(255, alpha)))
    
    def reset_colorazione(self, layer):
        if self.original_renderer:
            layer.setRenderer(self.original_renderer.clone())
            layer.triggerRepaint()
            self.iface.messageBar().pushInfo("Reset", "Colorazione ripristinata.")