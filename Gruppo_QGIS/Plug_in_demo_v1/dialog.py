# -*- coding: utf-8 -*-
"""
Finestra di dialogo per il plugin XML Prediction
"""
import os
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
from qgis.core import QgsProject, QgsFeatureRequest, QgsExpression
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
        
        # Parse XML to extract comuneCatastale and classamento info
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 1) find comuneCatastale attribute or element
            comune_val = None
            for el in root.iter():
                # attribute named comuneCatastale
                if 'comuneCatastale' in el.attrib:
                    comune_val = el.attrib.get('comuneCatastale')
                    break
                # or element tag containing comune and catast
                if el.tag.lower().find('comune') != -1 and (el.text and el.text.strip()):
                    comune_val = el.text.strip()
                    break
            QMessageBox.information(self, 'Nessun risultato', f'{comune_val}')

            # 2) find Classamento element with categoriaImmobiliare and sottoCategoriaImmobiliare
            categoria_main = None
            for el in root.iter():
                tag_up = el.tag.upper()
                if 'CLASSAMENT' in tag_up or 'CLASSAM' in tag_up:
                    if 'categoriaImmobiliare' in el.attrib and 'sottoCategoriaImmobiliare' in el.attrib:
                        categoria_main = el.attrib.get('categoriaImmobiliare')
                        break

            # fallback: try to find attributes deeper in Variazione/Ditta/... path
            if categoria_main is None:
                for el in root.iter():
                    if 'categoriaImmobiliare' in el.attrib and 'sottoCategoriaImmobiliare' in el.attrib:
                        categoria_main = el.attrib.get('categoriaImmobiliare')
                        break

            # Build combined categoria value (e.g., A00 + 01 -> A01)
            target_categoria = None
            if categoria_main:
                # If categoria_main like 'A00', take its first letter and append sotto_cat
                target_categoria = categoria_main
            if target_categoria == "A00":
                target_categoria = "A02"  # Example adjustment
            QMessageBox.information(self, 'Nessun risultato', f'{target_categoria}')
            # Now find layer and compute class frequencies for matching features
            results_text = ""
            predizioni = []
            layer_name = 'catasto_fabbricati'
            layers = QgsProject.instance().mapLayersByName(layer_name)
            if not layers:
                QMessageBox.warning(self, 'Errore', f"Layer '{layer_name}' non trovato nel progetto.")
                return
            layer = layers[0]

            # Build attribute filter expression
            expr_parts = []
            def esc(v):
                return str(v).replace("'", "''")
            if comune_val:
                expr_parts.append(f'"ccpart" = \'{esc(comune_val)}\'')
            if target_categoria:
                expr_parts.append(f'"CATEGORIA" = \'{esc(target_categoria)}\'')

            if expr_parts:
                expr = ' AND '.join(expr_parts)
                QMessageBox.information(self, 'Nessun risultato', f'{expr}')
                req = QgsFeatureRequest(QgsExpression(expr))
            else:
                expr = None
                req = QgsFeatureRequest()

            # get class distribution (field name CLASSE or case variants)
            classe_field = None
            for fn in layer.fields().names():
                if fn.upper() == 'CLASSE':
                    classe_field = fn
                    break

            counts = {}
            total = 0
            for f in layer.getFeatures(req):
                # count only features matching filter (across whole layer)
                val = None
                if classe_field and classe_field in f.fields().names():
                    val = f[classe_field]
                else:
                    # try common alternatives
                    for alt in ('CLASSE', 'Classe', 'classe'):
                        if alt in f.fields().names():
                            val = f[alt]
                            break
                if val is None:
                    continue
                val = str(val)
                counts[val] = counts.get(val, 0) + 1
                total += 1

            if total > 0:
                # compute probabilities and pick top 2
                items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                for cls, cnt in items[:2]:
                    prob = cnt / total
                    predizioni.append({'classe': cls, 'probabilita': prob})
                # if only one class present, add a second dummy
                if len(predizioni) == 1:
                    predizioni.append({'classe': 'N/A', 'probabilita': 0.0})
            else:
                # No matching features found
                QMessageBox.information(self, 'Nessun risultato', 'Nessuna unità trovata per i criteri estratti dal XML.')
                return

            # Present results in the PredictionDialog (reuse UI from ColorazioneClassi)
            try:
                from ColorazioneClassi.ColorazioneClassi.prediction_dialog import PredictionDialog as PD
                pdlg = PD([{'categoria': p['classe'], 'probabilita': p['probabilita']} for p in predizioni])
                pdlg.show()
            except Exception:
                pass

            # Apply a rule-based renderer highlighting top 2 classes across the whole layer
            try:
                from qgis.core import QgsRuleBasedRenderer, QgsFillSymbol
                from qgis.PyQt.QtGui import QColor

                root_rule = QgsRuleBasedRenderer.Rule(None)
                colors = [QColor(50,255,50,200), QColor(255,165,0,180)]
                for i, p in enumerate(predizioni[:2]):
                    cls = p['classe']
                    prob = p['probabilita']
                    # create simple symbol
                    symbol = QgsFillSymbol.createSimple({'color': colors[i].name(), 'outline_color':'black'})
                    rule = QgsRuleBasedRenderer.Rule(symbol)
                    filter_expr = (
                        f'"CLASSE" = \'{esc(cls)}\' AND '
                        f'"ccpart" = \'{esc(comune_val)}\' AND '
                        f'"CATEGORIA" = \'{esc(target_categoria)}\''
                    )
                    rule.setFilterExpression(filter_expr)
                    rule.setLabel(f"{cls} ({prob:.1%})")
                    root_rule.appendChild(rule)

                renderer = QgsRuleBasedRenderer(root_rule)
                # save original renderer to allow reset (if plugin ColorazioneClassi not used)
                try:
                    self._old_renderer = layer.renderer().clone()
                except Exception:
                    self._old_renderer = None
                layer.setRenderer(renderer)
                layer.triggerRepaint()
            except Exception as e:
                print('Renderer apply error:', e)

            # Show summary in results text
            results_text += f"XML analizzato: {os.path.basename(xml_path)}\n"
            results_text += f"Comune catastale estratto: {comune_val}\n"
            results_text += f"Categoria target: {target_categoria}\n"
            results_text += "\nPredizioni (top 2):\n"
            color_symbols = ["🟩", "🟧"]
            for i, p in enumerate(predizioni[:2]):
                results_text += f" {color_symbols[i]} Classe {p['classe']}: {p['probabilita']:.1%}\n"
            self.textEdit_results.setText(results_text)
            QMessageBox.information(self, "Predizione Completata", "Analisi XML e colorazione applicata con successo.")

        except Exception as ex:
            self.textEdit_results.setText(f"Errore nell'elaborazione XML: {str(ex)}")
            QMessageBox.critical(self, "Errore", f"Errore nell'elaborazione XML: {str(ex)}")