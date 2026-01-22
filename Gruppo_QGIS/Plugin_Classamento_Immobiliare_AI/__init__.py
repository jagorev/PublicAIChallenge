import inspect
import os
import subprocess
import sys

def install_dependencies():
    """Controlla e installa le dipendenze mancanti."""
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    req_file = os.path.join(path, "requirements.txt")
    
    if not os.path.exists(req_file):
        return

    # Lista di pacchetti 'critici' da controllare (nome import : nome pip)
    dependencies = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "reportlab": "reportlab",
        "shap": "shap",
        "joblib": "joblib"
    }
    missing = []
    
    for import_name, pip_name in dependencies.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        from qgis.PyQt.QtWidgets import QMessageBox
        res = QMessageBox.question(None, "Installazione Dipendenze", 
                                 f"Il plugin richiede i seguenti pacchetti: {', '.join(missing)}. \nVuoi installarli adesso? (Potrebbe volerci qualche minuto)",
                                 QMessageBox.Yes | QMessageBox.No)
        
        if res == QMessageBox.Yes:
            try:
                # Usa il comando pip integrato nel python corrente (quello di QGIS)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
                QMessageBox.information(None, "Successo", "Dipendenze installate. Riavvia QGIS per completare.")
            except Exception as e:
                QMessageBox.critical(None, "Errore", f"Errore durante l'installazione: {str(e)}")

def classFactory(iface):
    # Prova a installare prima di caricare il plugin vero e proprio
    install_dependencies()
    
    from .colorazione_classi import ColorazioneClassi
    return ColorazioneClassi(iface)
