# -*- coding: utf-8 -*-
"""
Plugin principale per la predizione XML
"""
import os
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsProject
from .dialog import XMLPredictionDialog

class XMLPredictionPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        
        # Inizializza locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'XMLPrediction_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        self.actions = []
        self.menu = self.tr(u'&XML Prediction')
        self.first_start = True  # initialize as True
        self.dlg = None           # initialize dlg attribute

    def tr(self, message):
        return QCoreApplication.translate('XMLPredictionPlugin', message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        self.add_action(
            icon_path,
            text=self.tr(u'XML Prediction'),
            callback=self.run,
            parent=self.iface.mainWindow())

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&XML Prediction'),
                action)
            self.iface.removeToolBarIcon(action)

    def run(self):
        # Create the dialog only on first start
        if self.first_start:
            self.first_start = False
            self.dlg = XMLPredictionDialog(iface=self.iface)  # pass iface if needed

        # Show the dialog
        self.dlg.show()
        result = self.dlg.exec_()
        
        if result:
            pass
