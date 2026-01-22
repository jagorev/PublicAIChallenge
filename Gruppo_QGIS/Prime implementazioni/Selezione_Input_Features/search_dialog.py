#search_dialog.py
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QCompleter
from PyQt5.QtCore import Qt

class SearchDialog(QDialog):
    def __init__(self, parent=None, categoria_choices=None, cc_choices=None):
        super(SearchDialog, self).__init__(parent)
        self.setWindowTitle("Ricerca per Categoria")

        self.label = QLabel("Categoria:")
        self.input_categoria = QLineEdit()
        self.input_categoria.setPlaceholderText("Lascia vuoto per tutte le categorie")

        # completer for categoria if provided
        if categoria_choices:
            completer = QCompleter(sorted(list(set(map(str, categoria_choices)))))
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            # allow matching anywhere in the string, not only prefix
            try:
                completer.setFilterMode(Qt.MatchContains)
            except Exception:
                # older PyQt versions may not have setFilterMode exposed; ignore
                pass
            self.input_categoria.setCompleter(completer)

        # Comune catastale input
        self.label_cc = QLabel("Comune catastale (CCPART):")
        self.input_cc = QLineEdit()
        self.input_cc.setPlaceholderText("Lascia vuoto per tutti i comuni")
        if cc_choices:
            completer_cc = QCompleter(sorted(list(set(map(str, cc_choices)))))
            completer_cc.setCaseSensitivity(Qt.CaseInsensitive)
            completer_cc.setCompletionMode(QCompleter.PopupCompletion)
            try:
                completer_cc.setFilterMode(Qt.MatchContains)
            except Exception:
                pass
            self.input_cc.setCompleter(completer_cc)

        self.button_cerca = QPushButton("Cerca")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input_categoria)

        # horizontal row for comune catastale label + input
        row = QHBoxLayout()
        row.addWidget(self.label_cc)
        row.addWidget(self.input_cc)
        layout.addLayout(row)

        layout.addWidget(self.button_cerca)
        self.setLayout(layout)

    def get_categoria(self):
        return self.input_categoria.text()

    def get_cc(self):
        return self.input_cc.text()
