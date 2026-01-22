import os
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon, QColor
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


class ColorazioneClassi:
    """
    Plugin QGIS per colorare particelle in base alle classi predette dal modello AI.
    
    Il plugin:
    1. Riceve una lista di classi con probabilità associate
    2. Colora le particelle di quelle classi con intensità proporzionale alla probabilità
    3. Alta probabilità = colore acceso
    4. Bassa probabilità = colore sbiadito
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
        
        # Lista di esempio delle predizioni (sarà sostituita dal modello reale)
        # Formato: [{"categoria": "A02", "probabilita": 0.85}, ...]
        # Il plugin mostrerà solo le prime 2 categorie con maggiore confidence
        # Nomenclatura: A02, A01, B06, C02, ecc.
        self.predizioni_esempio = [
            {"categoria": "A02", "probabilita": 0.90},
            {"categoria": "A03", "probabilita": 0.10}
        ]
    
    def initGui(self):
        """Inizializza l'interfaccia grafica del plugin"""
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        self.action = QAction(
            QIcon(icon_path), 
            "Colorazione Classi Predette", 
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Colorazione Classi", self.action)
    
    def unload(self):
        """Rimuove il plugin dall'interfaccia"""
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&Colorazione Classi", self.action)
    
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
        
        # Mostra dialog con le predizioni
        self.dialog = PredictionDialog(self.predizioni_esempio)
        self.dialog.button_applica.clicked.connect(
            lambda: self.applica_colorazione(layer)
        )
        self.dialog.button_reset.clicked.connect(
            lambda: self.reset_colorazione(layer)
        )
        self.dialog.show()
    
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
        
        # Prende solo le prime 2 categorie con maggiore confidence
        # (ordinate per probabilità decrescente)
        predizioni_sorted = sorted(predizioni, key=lambda x: x["probabilita"], reverse=True)
        top_2_predizioni = predizioni_sorted[:2]
        
        # Costruiamo un renderer basato su regole: una regola per ogni categoria predetta
        root_rule = QgsRuleBasedRenderer.Rule(None)

        for idx, pred in enumerate(top_2_predizioni):
            categoria = pred["categoria"]
            probabilita = pred["probabilita"]

            color = self.get_color_for_rank(idx, probabilita)
            # Crea simbolo con colore calcolato
            symbol = QgsFillSymbol.createSimple({
                'color': color.name(),
                'outline_color': 'black',
                'outline_width': '0.26'
            })
            symbol.setColor(color)  # Imposta anche l'alpha

            # Espressione identica a quella usata in RicercaCategoria
            expression = f'"CATEGORIA" = \'{categoria}\''

            rule = QgsRuleBasedRenderer.Rule(symbol)
            rule.setFilterExpression(expression)
            rule.setLabel(f"{categoria} ({probabilita:.0%})")
            root_rule.appendChild(rule)

        # Non aggiungiamo regola di fallback - le altre feature rimangono trasparenti

        renderer = QgsRuleBasedRenderer(root_rule)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
        
        # Non facciamo zoom - manteniamo la vista corrente
        
        self.iface.messageBar().pushSuccess(
            "Successo", 
            f"Colorazione applicata a {len(top_2_predizioni)} classi predette (top 2)."
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
        self.predizioni_esempio = predizioni_lista
