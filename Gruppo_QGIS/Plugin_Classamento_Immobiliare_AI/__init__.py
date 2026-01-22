import inspect
import os
import subprocess
import sys
from qgis.PyQt.QtWidgets import QMessageBox

def install_dependencies():
    """Controlla e installa le dipendenze mancanti."""
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    req_file = os.path.join(path, "requirements.txt")
    
    if not os.path.exists(req_file):
        return False

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
        res = QMessageBox.question(None, "Installazione Dipendenze", 
                                 f"Il plugin richiede i seguenti pacchetti: {', '.join(missing)}. \nVuoi installarli adesso? (Potrebbe volerci qualche minuto)",
                                 QMessageBox.Yes | QMessageBox.No)
        
        if res == QMessageBox.Yes:
            try:
                # Usa --user per evitare problemi di permessi su macOS
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-r", req_file])
                QMessageBox.information(None, "Successo", "Dipendenze installate correttamente.\n\nIMPORTANTE: Devi RIAVVIARE QGIS affinché le modifiche abbiano effetto.")
                return True 
            except Exception as e:
                QMessageBox.critical(None, "Errore", f"Errore durante l'installazione: {str(e)}")
                return False
        return False
    return True

class DummyPlugin:
    """Plugin fittizio usato quando mancano le dipendenze per evitare crash."""
    def __init__(self, iface):
        self.iface = iface
    def initGui(self): pass
    def unload(self): pass

def classFactory(iface):
    # Prova a installare/controllare le dipendenze
    install_dependencies()
    
    # Se le dipendenze mancano o sono state appena installate (richiede riavvio),
    # evitiamo di importare il codice principale che causerebbe il crash.
    try:
        from .colorazione_classi import ColorazioneClassi
        return ColorazioneClassi(iface)
    except ImportError:
        # Se siamo qui, significa che le librerie non sono ancora caricate
        return DummyPlugin(iface)
