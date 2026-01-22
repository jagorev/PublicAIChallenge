from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QWidget
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QBrush


class PredictionDialog(QDialog):
    """
    Dialog per visualizzare le predizioni del modello e applicare la colorazione.
    """
    
    def __init__(self, predizioni, parent=None):
        super().__init__(parent)
        self.predizioni = predizioni
        self.init_ui()
    
    def init_ui(self):
        """Inizializza l'interfaccia utente"""
        self.setWindowTitle("Classi Predette dal Modello")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout()
        
        # Titolo
        title = QLabel("<h2>Predizioni del Modello AI</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Descrizione
        desc = QLabel(
            "Vengono visualizzate solo le prime 2 classi con maggiore confidenza.\n"
            "1a categoria: Verde sgargiante | 2a categoria: Arancione\n"
            "Opacità: più alta la percentuale, più opaco/lucido il colore."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(desc)
        
        # Tabella predizioni
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Categoria", "Probabilità", "Anteprima Colore"])
        self.table.setRowCount(len(self.predizioni))
        
        # Popola la tabella
        for i, pred in enumerate(self.predizioni):
            # Categoria
            cat_item = QTableWidgetItem(pred["categoria"])
            cat_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 0, cat_item)
            
            # Probabilità
            prob = pred["probabilita"]
            prob_item = QTableWidgetItem(f"{prob:.1%}")
            prob_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 1, prob_item)
            
            # Anteprima colore (usa stesso algoritmo del plugin)
            color = self.get_preview_color(prob, i)
            color_item = QTableWidgetItem("")
            color_item.setBackground(QBrush(color))
            color_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 2, color_item)
        
        # Ridimensiona colonne
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.table)
        
        # Info aggiuntive
        info_label = QLabel(
            f"<b>Totale classi predette:</b> {len(self.predizioni)}"
        )
        layout.addWidget(info_label)
        
        # Pulsanti
        button_layout = QHBoxLayout()
        
        self.button_applica = QPushButton("Applica Colorazione")
        self.button_applica.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 10px; font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        button_layout.addWidget(self.button_applica)
        
        self.button_reset = QPushButton("Reset Colorazione")
        self.button_reset.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "padding: 10px; font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        button_layout.addWidget(self.button_reset)
        
        button_chiudi = QPushButton("Chiudi")
        button_chiudi.clicked.connect(self.close)
        button_chiudi.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 5px; }"
        )
        button_layout.addWidget(button_chiudi)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_preview_color(self, probabilita, categoria_index=0):
        """
        Restituisce un colore di anteprima in base alla probabilità.
        Stessa logica del plugin principale.
        
        - Categoria index 0 (1a): Verde sgargiante
        - Categoria index 1 (2a): Arancione
        """
        if categoria_index == 0:
            # Prima categoria: Verde sgargiante
            r, g, b = 50, 255, 50
            # Funzione esponenziale
            alpha = int(20 + (probabilita ** 2 * 235))
        else:
            # Seconda categoria: Arancione (più visibile)
            r, g, b = 255, 165, 0
            # Formula meno aggressiva per renderla più visibile
            alpha = int(80 + (probabilita ** 0.5 * 175))
        
        alpha = max(20, min(255, alpha))
        
        return QColor(r, g, b, alpha)
    
    def get_predizioni(self):
        """Restituisce la lista delle predizioni"""
        return self.predizioni
