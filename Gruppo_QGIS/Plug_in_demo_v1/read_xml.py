import xml.etree.ElementTree as ET
import pandas as pd

def apartments_from_xml(xml_path):
    all_rows = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Trova ElencoUI
        elenco_ui = root.find('.//ElencoUI')
        # Solo le UICostituzione (non UISoppressione)
        ui_elements = elenco_ui.findall('.//UICostituzione')

        for ui in ui_elements:
            row = {"source_file": xml_path}

            # --- Identificativo Catastale PM ---
            idpm = ui.find('.//ElencoIdentificativiCatastaliPM/IdentificativoCatastalePM')
            if idpm is not None:
                for k, v in idpm.attrib.items():
                    row[k] = v   # comuneCatastale, foglio, numeratore, subalterno

            # --- Classamento (categoria catastale) ---
            classamento = ui.find('.//Classamento')
            if classamento is not None:
                for k, v in classamento.attrib.items():
                    row[k] = v

            # --- Indirizzo ---
            indir = ui.find('.//ElencoIndirizzi/Indirizzo')
            if indir is not None:
                for k, v in indir.attrib.items():
                    row[k] = v

            # --- Piani (lista) ---
            piani = ui.findall('.//ElencoPiani/Piano')
            row["lista_piani"] = ";".join([p.attrib.get("numeroPiano", "") for p in piani])

            # --- Mod1N-2 (tutti gli attributi e i figli) ---
            mod1n2 = ui.find('.//Mod1N-2')
            if mod1n2 is not None:
                for k, v in mod1n2.attrib.items():
                    row[k] = v

                for elem in mod1n2.iter():
                    if elem is not mod1n2:
                        for k, v in elem.attrib.items():
                            row[k] = v

            all_rows.append(row)

    except Exception as e:
        print(f"Errore parsing {xml_path}: {e}")


    df_xml=pd.DataFrame(all_rows)

    int_cols = ['comuneCatastale', 'foglio', 'numeratore', 'subalterno']

    for col in int_cols:
        if col in df_xml.columns:
            df_xml[col] = (
                df_xml[col]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)  # keep only digits
                .astype(float)                        # allow NaN
                )
    return df_xml