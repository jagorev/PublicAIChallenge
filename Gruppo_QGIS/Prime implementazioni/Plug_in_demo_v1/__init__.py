# -*- coding: utf-8 -*-
"""
Plugin di predizione XML per QGIS
"""

def classFactory(iface):
    from .plugin_main import XMLPredictionPlugin
    return XMLPredictionPlugin(iface)