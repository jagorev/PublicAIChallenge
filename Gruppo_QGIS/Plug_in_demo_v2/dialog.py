# -*- coding: utf-8 -*-
import sys, os, types
import pickle
import sklearn
import joblib
from treeinterpreter import treeinterpreter as ti
import pandas as pd
import numpy as np

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
from .data_prepro import apartments_from_xml, read_fab, xml_merge_fab, filter_df, feature_cleaning



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
        fab_path = self.lineEdit_fabPath.text()
        if not fab_path:
            QMessageBox.warning(self, "Errore", "Seleziona un file .FAB.")
            return
        if not os.path.exists(fab_path):
            QMessageBox.warning(self, "Errore", "File FAB non esiste.")
            return

        df_fab = read_fab(fab_path)

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

        feature_names =[
    "lista_piani",
    "datiMetriciNettiManuali",
    "tipoRiferimento",
    "annoRiferimento",
    "spessoreMuri",
    "superficieLordaMq",
    "numeroPiano",
    "superficieMq",
    "giardinoSuperficieLordaMq",
    "accessoCarrabile",
    "altezzaMediaLocaliPrincipaliCm",
    "postoAutoScoperto",
    "altriAccessoriAltro",
    "denominatore",
    "intPorteInterneMetallo",
    "intPorteIngressoLegnoTamburato",
    "intPorteInterneLegnoTamburato",
    "num",
    "superficieUtileMq",
    "bagniNum",
    "bagniSuperficieUtileMq",
    "corridoiNum",
    "corridoiSuperficieUtileMq",
    "altezzaMediaUtileCm",
    "superficieMqVaniAventiAltezzaMediaMinore230Cm",
    "riscaldamento",
    "condizionamento",
    "videoCitofono",
    "ascensoreServizio",
    "acquaCalda",
    "citofonico",
    "camereParquet",
    "cucinaBagnoPiastrelleCeramica",
    "intPorteIngressoAltro",
    "intPorteInterneLegnoMassello",
    "estFinestreLegnoMassello",
    "estVetroCameraLegnoMassello",
    "altriAccessoriPiastrelleCeramica",
    "intPorteIngressoMetallo",
    "estFinestreAltro",
    "estFinestreMetallo",
    "superficieLordaComunicantiMq",
    "camerePiastrelleCeramica",
    "estVetroCameraAltro",
    "intPorteIngressoLegnoMassello",
    "camereAltro",
    "cucinaBagnoParquet",
    "altriAccessoriParquet",
    "superficieLordaNonComunicantiMq",
    "camereMarmo",
    "cucinaBagnoMarmo",
    "ascensoriNumero",
    "camereGomme",
    "pianiFuoriTerraNum",
    "pianiFuoriTerraMc",
    "pianiEntroTerraNum",
    "pianiEntroTerraMc",
    "estVetroCameraMetallo",
    "cucinaBagnoAltro",
    "cucinaBagnoPiastrelleScaglie",
    "altriAccessoriPiastrelleScaglie",
    "ascensoreUsoEsclusivo",
    "montacarichi",
    "altriAccessoriGomme",
    "estDoppioInfissoLegnoMassello",
    "intPorteInterneAltro",
    "altriAccessoriMarmo",
    "altriAccessoriMoquette",
    "camereMoquette",
    "estDoppioInfissoAltro",
    "cucinaBagnoGomme",
    "camerePiastrelleScaglie",
    "estDoppioInfissoMetallo",
    "altroSuperficieLordaMq",
    "cucinaBagnoMoquette",
    "ZONA",
    "SUPERFICIE",
    "VALIMIS",
    "PIANI"]
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
                QMessageBox.information(self, "Aggiunta colonna", f"Aggiunta colonna mancante: {col}") 

        # Drop extra columns
        diff_cols= set(X.columns) - set(feature_names)
        X = X[feature_names]
        QMessageBox.information(self, "Drop colonne", f"Rimosse colonne extra: {diff_cols}")
        X = X[[col for col in feature_names if col in X.columns]]

        # -----------------------
        # Load ONNX model
        # -----------------------
        plugin_dir = os.path.dirname(__file__)
        model_path = os.path.join(plugin_dir, "model.pkl")

        with open(model_path, "rb") as f:
            d = pickle.load(f)
        model = d["model"]
        # --- Predict ---

        class _DummyWriter:
            def write(self, *args, **kwargs): pass
            def flush(self, *a, **k): pass

        _saved_stderr = sys.stderr
        try:
            if getattr(sys, "stderr", None) is None:
                sys.stderr = _DummyWriter()
            # Also ensure sys.stdout if needed
            if getattr(sys, "stdout", None) is None:
                sys.stdout = _DummyWriter()

            # run predictions
            model.n_jobs = 1
            y_pred = model.predict(X)[0]
            y_proba = model.predict_proba(X)[0]
        finally:
            # restore
            sys.stderr = _saved_stderr
        classes = model.classes_

        # --- Top 2 ---
        probs = list(zip(classes, y_proba))
        probs_sorted = sorted(probs, key=lambda x: x[1], reverse=True)
        top2 = probs_sorted[:2]

        # -----------------------
        # Show results
        # -----------------------
        results = f"XML: {os.path.basename(xml_path)}\n"
        results += "Predizioni top 2:\n\n"

        color_symbols = ["🟩", "🟧"]

        for i, (cls, prob) in enumerate(top2):
            results += f"{color_symbols[i]} Classe {cls}: {prob:.1%}\n"

        self.textEdit_results.setText(results)

        prediction, bias, contributions = ti.predict(model, X)

        # Average across trees → 1D vector (n_features,)
        contribs = contributions[0].mean(axis=0)

        n_model_features = contribs.shape[0]

        # Keep only the first n_model_features from X
        X_used = X.iloc[:, :n_model_features]

        # Build dataframe
        shap_df = pd.DataFrame({
            "feature": X_used.columns,
            "value": X_used.iloc[0].values,
            "contribution": contribs,
            "abs": np.abs(contribs)
        }).sort_values("abs", ascending=False)

        # Top 5 most influential features
        top5 = shap_df.head(5)

        explanation = "📌 MOTIVO DELLA PREDIZIONE\n\n"
        explanation += f"La classe stimata è **{y_pred}** (probabilità {max(y_proba):.1%})\n\n"
        explanation += "Fattori più influenti:\n"

        for _, row in top5.iterrows():
            direction = "↑ aumenta" if row["contribution"] > 0 else "↓ riduce"
            explanation += (
                f"• **{row['feature']}** = {row['value']}"
                f"  ({direction} la probabilità)\n"
            )

        self.textEdit_results.setText(explanation)