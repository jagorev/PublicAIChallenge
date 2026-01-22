# __init__.py
def classFactory(iface):
    from .main_plugin import RicercaCategoria
    return RicercaCategoria(iface)
