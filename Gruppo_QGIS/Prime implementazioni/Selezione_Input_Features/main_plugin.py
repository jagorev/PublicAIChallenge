#main_plugin.py
import os
from qgis.PyQt.QtCore import Qt, QPoint
from qgis.PyQt.QtWidgets import QAction, QToolTip
from qgis.PyQt.QtGui import QIcon, QCursor, QColor
from qgis.core import (
    QgsProject,
    QgsExpression,
    QgsFeatureRequest,
    QgsRectangle,
    QgsVectorLayer,
    QgsRuleBasedRenderer,
)
from qgis.gui import QgsMapToolEmitPoint, QgsMapTool
from qgis.utils import iface

from .search_dialog import SearchDialog

class _ClickToggleMapTool(QgsMapTool):
    """Custom map tool that captures mouse clicks and calls a callback method."""
    def __init__(self, canvas, callback):
        super().__init__(canvas)
        self.canvas = canvas
        self.callback = callback

    def canvasReleaseEvent(self, event):
        # Convert mouse position to map coordinates
        point = self.toMapCoordinates(event.pos())
        button = event.button()
        try:
            self.callback(point, button)
        except Exception as e:
            print("Error in click callback:", e)

class RicercaCategoria:
    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        self.action = None
        self.cid = None  # Connessione segnale per il tracking mouse
        self.selected_ids = set()
        self.selected_layer = None
        self.map_tool = None
        self.prev_map_tool = None
        self.prev_renderer = None

    def initGui(self):
        plugin_dir = os.path.dirname(__file__)
        icon_path = os.path.join(plugin_dir, 'icon.png')

        self.action = QAction(QIcon(icon_path), "Selezione Input", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("Selezione Input", self.action)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("Selezione Input", self.action)

    def run(self):
        # Try to gather distinct values for completers from the layer
        layer_name = "catasto_fabbricati"
        categories = None
        cc_list = None
        layer = QgsProject.instance().mapLayersByName(layer_name)
        if layer:
            lyr = layer[0]
            if "CATEGORIA" in lyr.fields().names():
                try:
                    categories = [str(f["CATEGORIA"]) for f in lyr.getFeatures() if f["CATEGORIA"] not in (None, "")]
                except Exception:
                    categories = None
            if "ccpart" in lyr.fields().names():
                try:
                    cc_list = [str(f["ccpart"]) for f in lyr.getFeatures() if f["ccpart"] not in (None, "")]
                except Exception:
                    cc_list = None

        self.dialog = SearchDialog(categoria_choices=categories, cc_choices=cc_list)
        self.dialog.button_cerca.clicked.connect(self.cerca_categoria)
        self.dialog.finished.connect(self.termina_interazione)
        self.dialog.show()
        self.dialog.raise_()

    def cerca_categoria(self):
        categoria = self.dialog.get_categoria().strip()
        cc = self.dialog.get_cc().strip()

        layer_name = "catasto_fabbricati"
        layer = QgsProject.instance().mapLayersByName(layer_name)
        if not layer:
            self.iface.messageBar().pushCritical("Errore", f"Layer '{layer_name}' non trovato.")
            return

        layer = layer[0]
        layer.removeSelection()

        # Validate categoria if provided against distinct values (if field exists)
        if categoria and "CATEGORIA" in layer.fields().names():
            # build set of existing categories
            try:
                existing = set(str(f["CATEGORIA"]) for f in layer.getFeatures() if f["CATEGORIA"] not in (None, ""))
            except Exception:
                existing = None
            if existing is not None and categoria not in existing:
                self.iface.messageBar().pushWarning("Categoria non valida", f"{categoria} non è una categoria valida.")
                return

        # Build expression parts; empty inputs mean wildcard (no clause)
        parts = []
        def esc(v):
            return v.replace("'", "''")

        if categoria:
            parts.append(f'"CATEGORIA" = \'{esc(categoria)}\'')
        if cc:
            parts.append(f'"CCPART" = \'{esc(cc)}\'')

        if parts:
            expr = ' AND '.join(parts)
        else:
            # no filters: select features within extent
            expr = None

        if expr:
            # when filtering by attributes, select across the whole layer (not limited to current extent)
            request = QgsFeatureRequest(QgsExpression(expr))
        else:
            request = QgsFeatureRequest()
            request.setFilterRect(self.iface.mapCanvas().extent())

        ids = [f.id() for f in layer.getFeatures(request)]
        if ids:
            layer.selectByIds(ids)
            self.iface.messageBar().pushSuccess("Selezione completata", f"Selezionati {len(ids)} poligoni.")

            # Attiva tracking del mouse
            self.cid = self.iface.mapCanvas().xyCoordinates.connect(self.show_popup_on_hover)
            self.selected_ids = set(ids)
            self.selected_layer = layer
            # enable interactive click to toggle selection
            self.activate_toggle_tool()
            # If we used attribute filter (expr present), apply a rule-based renderer to highlight matches across the whole layer
            if expr:
                try:
                    # save current renderer so we can restore later
                    try:
                        self.prev_renderer = layer.renderer().clone()
                    except Exception:
                        self.prev_renderer = None

                    # create a symbol based on current renderer's symbol and tint it
                    base_sym = None
                    try:
                        base_sym = layer.renderer().symbol().clone()
                        base_sym.setColor(QColor(255, 0, 0, 120))
                    except Exception:
                        base_sym = None

                    # build rule-based renderer
                    root_rule = QgsRuleBasedRenderer.Rule(None)
                    if base_sym is not None:
                        rule = QgsRuleBasedRenderer.Rule(base_sym, QgsExpression(expr), 'Match')
                        root_rule.appendChild(rule)
                    renderer = QgsRuleBasedRenderer(root_rule)
                    layer.setRenderer(renderer)
                    layer.triggerRepaint()
                except Exception as e:
                    print("Could not apply rule-based renderer:", e)
        else:
            if categoria or cc:
                self.iface.messageBar().pushWarning("Nessun risultato", f"Nessun poligono trovato per i criteri specificati.")
            else:
                self.iface.messageBar().pushWarning("Nessun risultato", "Nessun poligono trovato nell'estensione corrente.")

    def show_popup_on_hover(self, point):
        if not self.selected_layer or not self.selected_ids:
            return

        request = QgsFeatureRequest().setFilterRect(self.iface.mapCanvas().extent())
        for feat in self.selected_layer.getFeatures(request):
            if feat.id() in self.selected_ids and feat.geometry().contains(point):
                chiave = feat["CHIAVE"]
                expr = QgsExpression(f""""CHIAVE" = '{chiave}'""")
                features = self.selected_layer.getFeatures(QgsFeatureRequest(expr))

                comuni = ["TIPOPARTEF", "PU_CODE", "CCPART", "NUMPART", "DENOMPART", "DCAmi", "CCAMA", "DAMmi", "DAMMA"]
                variabili = ["ZONA", "CATEGORIA", "CLASSE", "NUMUNITA"]

                val_comuni = {}
                tabella = []

                for f in features:
                    row = [f[v] if v in f.fields().names() else "" for v in variabili]
                    tabella.append(row)
                    for c in comuni:
                        if c not in f.fields().names():
                            continue
                        valore = f[c]
                        if c not in val_comuni:
                            val_comuni[c] = valore
                        elif val_comuni[c] != valore:
                            val_comuni[c] = "..."

                testo = "<b>Dati comuni:</b><br>"
                for k, v in val_comuni.items():
                    testo += f"{k}: {v}<br>"

                testo += "<br><b>Valori variabili:</b><br><table border='1'><tr>"
                for v in variabili:
                    testo += f"<th>{v}</th>"
                testo += "</tr>"

                for row in tabella:
                    testo += "<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>"
                testo += "</table>"

                QToolTip.showText(QCursor.pos(), testo)
                return

        QToolTip.hideText()

    def activate_toggle_tool(self):
        """Activate custom map tool that toggles selection by clicking."""
        try:
            self.prev_map_tool = self.iface.mapCanvas().mapTool()
            self.map_tool = _ClickToggleMapTool(self.iface.mapCanvas(), self.on_map_click)
            self.iface.mapCanvas().setMapTool(self.map_tool)
            self.iface.messageBar().pushInfo(
                "Modalità interattiva",
                "Clicca sulle unità per selezionare o deselezionare manualmente."
            )
        except Exception as e:
            print("activate_toggle_tool error:", e)
            self.map_tool = None

    def on_map_click(self, point, button):
        """Toggle selection of the first feature under the click."""
        try:
            if not self.selected_layer:
                return

            # Compute tolerance in map units
            mup = self.iface.mapCanvas().mapUnitsPerPixel()
            tol = mup * 5  # 5 pixels radius
            rect = QgsRectangle(point.x() - tol, point.y() - tol, point.x() + tol, point.y() + tol)
            request = QgsFeatureRequest().setFilterRect(rect)

            clicked_fid = None
            for feat in self.selected_layer.getFeatures(request):
                clicked_fid = feat.id()
                break

            if clicked_fid is None:
                return

            # toggle the feature ID in our internal set
            if clicked_fid in self.selected_ids:
                self.selected_ids.remove(clicked_fid)
            else:
                self.selected_ids.add(clicked_fid)

            # explicitly update selection on the layer
            self.selected_layer.removeSelection()
            self.selected_layer.selectByIds(list(self.selected_ids))

            # force visual update (especially important with custom renderer)
            self.selected_layer.triggerRepaint()
            self.iface.mapCanvas().refresh()

            self.iface.messageBar().pushInfo(
                "Selezione aggiornata", f"Selezionati {len(self.selected_ids)} poligoni."
            )

        except Exception as e:
            print("on_map_click error:", e)

    def termina_interazione(self):
        if self.cid:
            self.iface.mapCanvas().xyCoordinates.disconnect(self.cid)
            self.cid = None

        # Restore previous map tool
        try:
            if self.prev_map_tool is not None:
                self.iface.mapCanvas().setMapTool(self.prev_map_tool)
                self.prev_map_tool = None
        except Exception:
            pass

        # Restore previous renderer if we replaced it
        try:
            if self.prev_renderer is not None and self.selected_layer is not None:
                self.selected_layer.setRenderer(self.prev_renderer)
                self.selected_layer.triggerRepaint()
                self.prev_renderer = None
        except Exception:
            pass

        # Keep final selection (optional) or clear it
        # Comment this line out if you want to keep user selections after closing dialog
        # self.selected_layer.removeSelection()

        QToolTip.hideText()
        self.selected_layer = None
        self.selected_ids = set()