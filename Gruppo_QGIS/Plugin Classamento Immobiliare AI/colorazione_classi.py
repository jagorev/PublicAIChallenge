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
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsFillSymbol,
    QgsRuleBasedRenderer
)
from qgis.utils import iface

from .prediction_dialog import PredictionDialog
from .model_inference import ModelInference


class WelcomeDialog(QDialog):
    """
    Dialog semplice che chiede all'utente di selezionare un file CSV.
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
        
        # Messaggio
        msg = QLabel("Seleziona il file CSV contenente i dati dell'immobile da classificare.")
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
    
    Il plugin:
    1. Carica dati da CSV ed esegue inferenza con pipeline a cascata (CATEGORIA -> CLASSE)
    2. Mostra le top 2 predizioni per CATEGORIA e CLASSE con spiegazioni SHAP
    3. Colora le particelle in base alla categoria predetta
    """
    
    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        self.action = None
        self.plugin_dir = os.path.dirname(__file__)
        
        # Layer fisso da utilizzare
        self.layer_name = "catasto_fabbricati"
        
        # Salva il renderer originale per poterlo ripristinare
        self.original_renderer = None
        
        # Inizializza il modulo di inferenza
        self.model_inference = ModelInference()
        
        # Predizioni correnti
        self.predizioni_correnti = []
        self.shap_explanation = None
        self.inference_results_dir = None  # Cartella con i risultati per il report
    
    def initGui(self):
        """Inizializza l'interfaccia grafica del plugin"""
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
        """Rimuove il plugin dall'interfaccia"""
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&Classamento AI", self.action)
    
    def run(self):
        """Esegue il plugin"""
        # Verifica che il layer esista
        layer_list = QgsProject.instance().mapLayersByName(self.layer_name)
        if not layer_list:
            QMessageBox.critical(
                None, 
                "Errore", 
                f"Il layer '{self.layer_name}' non è caricato nel progetto."
            )
            return
        
        layer = layer_list[0]
        
        # Mostra il dialog di benvenuto che guida l'utente
        welcome = WelcomeDialog()
        if welcome.exec_() != QDialog.Accepted:
            return  # L'utente ha annullato
        
        # Ora chiedi di selezionare il file CSV
        csv_path, _ = QFileDialog.getOpenFileName(
            None,
            "📂 Seleziona il file CSV con i dati dell'immobile",
            "",
            "File CSV (*.csv);;Tutti i file (*.*)"
        )
        
        if not csv_path:
            QMessageBox.information(
                None,
                "Operazione annullata",
                "Nessun file selezionato. Puoi riprovare quando vuoi!"
            )
            return
        
        # Esegui l'inferenza con il modello
        try:
            self.predizioni_correnti, self.shap_explanation, self.inference_results_dir = self.esegui_inferenza(csv_path)
        except Exception as e:
            QMessageBox.critical(
                None,
                "Errore nell'inferenza",
                f"Impossibile eseguire l'inferenza:\n{str(e)}"
            )
            return
        
        if not self.predizioni_correnti:
            QMessageBox.warning(
                None,
                "Attenzione",
                "Nessuna predizione generata dal modello."
            )
            return
        
        # Mostra dialog con le predizioni
        self.dialog = PredictionDialog(
            self.predizioni_correnti, 
            self.shap_explanation,
            self.inference_results_dir
        )
        self.dialog.button_applica.clicked.connect(
            lambda: self.applica_colorazione(layer)
        )
        self.dialog.button_reset.clicked.connect(
            lambda: self.reset_colorazione(layer)
        )
        self.dialog.show()
    
    def esegui_inferenza(self, csv_path):
        """
        Esegue l'inferenza sul file CSV fornito.
        
        Args:
            csv_path: Percorso al file CSV con i dati dell'immobile
            
        Returns:
            Tuple (predizioni, shap_explanation, results_dir):
            - predizioni: Lista di dizionari con categoria, classe e probabilità
            - shap_explanation: Dizionario con spiegazione SHAP (o None)
            - results_dir: Cartella con i file CSV per il report
        """
        import pandas as pd
        
        # Carica i dati dal CSV
        data = pd.read_csv(csv_path)
        
        # Cartella per salvare i risultati (nella stessa cartella del CSV)
        csv_dir = os.path.dirname(csv_path)
        results_dir = os.path.join(csv_dir, "inference_results")
        
        # Salva i risultati per il report
        try:
            self.model_inference.save_inference_results(data, results_dir, top_n=3)
        except Exception as e:
            print(f"Errore nel salvare i risultati: {e}")
        
        # Esegui l'inferenza (restituisce formato compatibile)
        predizioni = self.model_inference.get_single_prediction(data, top_n=2)
        
        # Calcola spiegazione SHAP (opzionale)
        shap_explanation = None
        try:
            shap_explanation = self.model_inference.compute_shap_explanation(data, target='categoria')
        except Exception as e:
            print(f"SHAP non disponibile: {e}")
        
        return predizioni, shap_explanation, results_dir
    
    def applica_colorazione(self, layer):
        """
        Applica la colorazione al layer in base alle predizioni.
        
        Args:
            layer: Layer QGIS su cui applicare la colorazione
        """
        predizioni = self.dialog.get_predizioni()
        
        if not predizioni:
            QMessageBox.warning(
                None, 
                "Attenzione", 
                "Nessuna predizione disponibile da colorare."
            )
            return
        
        # Salva il renderer originale prima di modificarlo (solo la prima volta)
        if self.original_renderer is None:
            self.original_renderer = layer.renderer().clone()
        
        # Prende le categorie predette (ordinate per probabilità)
        predizioni_sorted = sorted(predizioni, key=lambda x: x.get("probabilita_cat", x.get("probabilita", 0)), reverse=True)
        top_predizioni = predizioni_sorted[:2]
        
        # Costruiamo un renderer basato su regole
        root_rule = QgsRuleBasedRenderer.Rule(None)

        for idx, pred in enumerate(top_predizioni):
            categoria = pred["categoria"]
            classe = pred.get("classe", "N/A")
            prob_cat = pred.get("probabilita_cat", pred.get("probabilita", 0))
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
            rule.setLabel(f"{categoria}/{classe} (Cat:{prob_cat:.0%}, Cl:{prob_classe:.0%})")
            root_rule.appendChild(rule)

        renderer = QgsRuleBasedRenderer(root_rule)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
        
        self.iface.messageBar().pushSuccess(
            "Successo", 
            f"Colorazione applicata. Predizione: {top_predizioni[0]['categoria']}/{top_predizioni[0].get('classe', 'N/A')}"
        )
    
    def get_color_for_rank(self, rank, probabilita):
        """
        Restituisce un colore basato sul ranking (1a o 2a categoria).
        
        - Rank 0 (1a categoria): Verde sgargiante
        - Rank 1 (2a categoria): Arancione
        
        L'opacità è proporzionale alla probabilità (alta % = più opaco).
        
        Args:
            rank (int): 0 per prima categoria, 1 per seconda
            probabilita (float): Valore tra 0 e 1 (confidence)
            
        Returns:
            QColor: Colore corrispondente
        """
        if rank == 0:
            # Prima categoria: Verde sgargiante (lime green)
            r, g, b = 50, 255, 50
            # Usa funzione esponenziale per accentuare le differenze
            alpha = int(20 + (probabilita ** 2 * 235))
        else:
            # Seconda categoria: Arancione (più visibile, meno sbiadita)
            r, g, b = 255, 165, 0
            # Formula meno aggressiva per la 2a classe: usa radice quadrata invece di quadrato
            # per rendere più visibili anche le probabilità medie/basse
            alpha = int(80 + (probabilita ** 0.5 * 175))
        
        alpha = max(20, min(255, alpha))
        
        return QColor(r, g, b, alpha)
    
    def zoom_to_predicted_classes(self, layer, predizioni):
        """
        Fa zoom alle particelle delle classi predette.
        
        Args:
            layer: Layer QGIS
            predizioni: Lista di dizionari con categoria e probabilità
        """
        # Usa le categorie direttamente (già in formato A01, B02)
        categorie = [p["categoria"] for p in predizioni]

        # Crea espressione per filtrare
        expr_parts = [f'"CATEGORIA" = \'{cat}\'' for cat in categorie if cat]
        if not expr_parts:
            return
        expr = " OR ".join(expr_parts)

        request = QgsFeatureRequest(QgsExpression(expr))
        ids = [f.id() for f in layer.getFeatures(request)]

        if ids:
            layer.selectByIds(ids)
            canvas = self.iface.mapCanvas()
            canvas.zoomToSelected(layer)
            canvas.zoomOut()  # Zoom out un po' per vedere meglio
            layer.removeSelection()
    
    def reset_colorazione(self, layer):
        """
        Resetta la colorazione al renderer originale (com'era prima di "applica colorazione").
        
        Args:
            layer: Layer QGIS
        """
        if self.original_renderer is not None:
            # Ripristina il renderer salvato
            layer.setRenderer(self.original_renderer.clone())
            layer.triggerRepaint()
            
            self.iface.messageBar().pushInfo(
                "Reset", 
                "Colorazione ripristinata allo stato originale."
            )
        else:
            self.iface.messageBar().pushWarning(
                "Attenzione", 
                "Nessuna colorazione da ripristinare. Applica prima una colorazione."
            )
    
    def set_predizioni_da_modello(self, predizioni_lista):
        """
        Metodo per impostare le predizioni dal modello esterno.
        
        Args:
            predizioni_lista: Lista di dizionari con formato:
                [{"categoria": "A02", "probabilita": 0.85}, ...]
                Nomenclatura: A02, A01, B06, C02, ecc.
        """
        self.predizioni_correnti = predizioni_lista
