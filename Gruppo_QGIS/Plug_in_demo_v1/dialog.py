# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
from qgis.core import QgsProject

# --------------------------
# Load UI file
# --------------------------
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'dialog.ui'
))

# --------------------------
# Your data pipeline modules
# --------------------------
from data_prepro import apartments_from_xml, read_fab, xml_merge_fab, filter_df, feature_cleaning



class XMLPredictionDialog(QtWidgets.QDialog, FORM_CLASS):

    def __init__(self, iface=None, parent=None):
        super().__init__(parent)

        self.iface = iface      # keeps QGIS interface reference
        self.setupUi(self)      # loads all widgets from dialog.ui

        # Connect buttons (assuming names from your .ui)
        self.pushButton_selectXML.clicked.connect(self.select_xml_file)
        self.pushButton_selectFAB.clicked.connect(self.select_fab_file)
        self.pushButton_predict.clicked.connect(self.run_prediction)

    # ----------------------------------------------------------------------
    # FILE SELECTORS
    # ----------------------------------------------------------------------
    def select_xml_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona XML", "", "File XML (*.xml)"
        )
        if path:
            self.lineEdit_xmlPath.setText(path)

    def select_fab_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona FAB", "", "File FAB (*.fab *.FAB)"
        )
        if path:
            self.lineEdit_fabPath.setText(path)

    # ----------------------------------------------------------------------
    # MAIN PREDICTION PIPELINE
    # ----------------------------------------------------------------------
    def run_prediction(self):

        # -----------------------
        # 1) Load XML
        # -----------------------
        xml_path = self.lineEdit_xmlPath.text()

        if not xml_path:
            QMessageBox.warning(self, "Errore", "Seleziona un file XML.")
            return

        df_xml = apartments_from_xml(xml_path)

        if df_xml.empty:
            QMessageBox.warning(self, "Errore", "XML non contiene UICostituzione valide.")
            return

        df_xml = df_xml.iloc[:1]   # Only one building

        # -----------------------
        # 2) Load FAB (with caching)
        # -----------------------
        cached_fab = QgsProject.instance().customProperty("cached_fab_df")

        if cached_fab:
            df_fab = pd.read_json(cached_fab)
        else:
            fab_path = self.lineEdit_fabPath.text()
            if not fab_path:
                QMessageBox.warning(self, "Errore", "Seleziona un file .FAB.")
                return
            if not os.path.exists(fab_path):
                QMessageBox.warning(self, "Errore", "File FAB non esiste.")
                return

            df_fab = read_fab(fab_path)

            # Save in QGIS project cache
            QgsProject.instance().setCustomProperty(
                "cached_fab_df", df_fab.to_json()
            )

        # -----------------------
        # 3) Merge XML + FAB
        # -----------------------
        df_merged = xml_merge_fab(df_xml, df_fab)

        if df_merged.empty:
            QMessageBox.warning(self, "Errore", "Nessuna corrispondenza trovata fra XML e FAB.")
            return

        # -----------------------
        # 4) Select columns
        # -----------------------
        df_filtered = filter_df(df_merged)

        # -----------------------
        # 5) Feature cleaning
        # -----------------------
        X = feature_cleaning(df_filtered)

        # -----------------------
        # 6) Load model + predict
        # -----------------------
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_proba = model.predict_proba(X)[0]
        classes = model.classes_

        probs_sorted = sorted(
            zip(classes, y_proba),
            key=lambda x: x[1],
            reverse=True
        )
        top2 = probs_sorted[:2]

        # -----------------------
        # 7) Show results
        # -----------------------
        results = f"XML: {os.path.basename(xml_path)}\n"
        results += "Predizioni top 2:\n\n"

        color_symbols = ["🟩", "🟧"]

        for i, (cls, prob) in enumerate(top2):
            results += f"{color_symbols[i]} Classe {cls}: {prob:.1%}\n"

        self.textEdit_results.setText(results)

        # -----------------------
        # 8) Optional: Color layer
        # -----------------------
        # Your previous code goes here if needed.
