# -*- coding: utf-8 -*-
"""
Finestra di dialogo per il plugin XML Prediction
"""
import os
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
from qgis.core import QgsProject
import xml.etree.ElementTree as ET

# Carica il file UI
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'dialog.ui'))

class XMLPredictionDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(XMLPredictionDialog, self).__init__(parent)
        self.setupUi(self)
        
        # Connessioni dei pulsanti
        self.pushButton_selectXML.clicked.connect(self.select_xml_file)
        self.pushButton_predict.clicked.connect(self.run_prediction)
        self.pushButton_refreshLayers.clicked.connect(self.refresh_layers)
        
        # Inizializza l'interfaccia
        self.refresh_layers()
        
    def select_xml_file(self):
        """Apre il dialog per selezionare il file XML"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona file XML",
            "",
            "File XML (*.xml);;Tutti i file (*)"
        )
        
        if file_path:
            self.lineEdit_xmlPath.setText(file_path)
            self.load_xml_info(file_path)
    
    def load_xml_info(self, file_path):
        """Carica e mostra informazioni sul file XML selezionato"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            info_text = f"File XML caricato: {os.path.basename(file_path)}\n"
            info_text += f"Elemento radice: {root.tag}\n"
            info_text += f"Numero di elementi figli: {len(root)}\n"
            
            # Mostra alcuni attributi se presenti
            if root.attrib:
                info_text += "Attributi principali:\n"
                for key, value in list(root.attrib.items())[:5]:  # Primi 5 attributi
                    info_text += f"  - {key}: {value}\n"
            
            self.textEdit_xmlInfo.setText(info_text)
            
        except Exception as e:
            self.textEdit_xmlInfo.setText(f"Errore nel caricamento del file XML: {str(e)}")
    
    def refresh_layers(self):
        """Aggiorna la lista dei layer QGIS aperti"""
        layers_info = "Layer QGIS attualmente aperti:\n\n"
        
        project = QgsProject.instance()
        layers = project.mapLayers()
        
        if not layers:
            layers_info += "Nessun layer caricato in QGIS."
        else:
            for layer_id, layer in layers.items():
                layers_info += f"• {layer.name()}\n"
                layers_info += f"  Tipo: {layer.type().name if hasattr(layer.type(), 'name') else 'Unknown'}\n"
                layers_info += f"  Sorgente: {layer.source()[:50]}...\n"
                if hasattr(layer, 'featureCount'):
                    layers_info += f"  Features: {layer.featureCount()}\n"
                layers_info += "\n"
        
        self.textEdit_layersInfo.setText(layers_info)
    
    def run_prediction(self):
        """Esegue la predizione demo"""
        xml_path = self.lineEdit_xmlPath.text()
        
        if not xml_path:
            QMessageBox.warning(self, "Attenzione", "Seleziona prima un file XML!")
            return
        
        if not os.path.exists(xml_path):
            QMessageBox.warning(self, "Errore", "Il file XML selezionato non esiste!")
            return
        
        # Simulazione di predizione
        demo_results = """RISULTATI PREDIZIONE (DEMO):
        
✓ File XML analizzato con successo
✓ Layer QGIS identificati e processati
✓ Confronto completato

Classi predette:
• Classe A: 35% di probabilità
• Classe B: 45% di probabilità  
• Classe C: 20% di probabilità

Classe principale identificata: CLASSE B (45%)

Dettagli analisi:
- Elementi XML processati: 156
- Layer QGIS utilizzati: """ + str(len(QgsProject.instance().mapLayers())) + """
- Tempo di elaborazione: 0.8s (simulato)
- Accuratezza stimata: 87%

Nota: Questa è una demo. I risultati sono simulati."""

        self.textEdit_results.setText(demo_results)
        QMessageBox.information(self, "Predizione Completata", "La predizione demo è stata eseguita con successo!")