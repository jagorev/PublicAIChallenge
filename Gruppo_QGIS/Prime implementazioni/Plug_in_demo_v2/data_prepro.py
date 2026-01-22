import xml.etree.ElementTree as ET
import pandas as pd
import json
import numpy as np
import os

def apartments_from_xml(xml_path):
    all_rows = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        elenco_ui = root.find('.//ElencoUI')
        if elenco_ui is None:
            return pd.DataFrame()

        # Accept both UICostituzione and UIVariazione
        ui_elements = elenco_ui.findall('.//UICostituzione')
        if not ui_elements:
            ui_elements = elenco_ui.findall('.//UIVariazione')

        for ui in ui_elements:
            row = {"source_file": xml_path}

            idpm = ui.find('.//ElencoIdentificativiCatastaliPM/IdentificativoCatastalePM')
            if idpm is not None:
                for k, v in idpm.attrib.items():
                    row[k] = v

            classamento = ui.find('.//Classamento')
            if classamento is not None:
                for k, v in classamento.attrib.items():
                    row[k] = v

            indir = ui.find('.//ElencoIndirizzi/Indirizzo')
            if indir is not None:
                for k, v in indir.attrib.items():
                    row[k] = v

            piani = ui.findall('.//ElencoPiani/Piano')
            row["lista_piani"] = ";".join([p.attrib.get("numeroPiano", "") for p in piani])

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

    df_xml = pd.DataFrame(all_rows)

    int_cols = ['comuneCatastale', 'foglio', 'numeratore', 'subalterno']
    for col in int_cols:
        if col in df_xml.columns:
            df_xml[col] = (
                df_xml[col].astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )
    return df_xml


def read_fab(fab_path):

    def safe(parts, i):
        return parts[i].strip() if i < len(parts) and parts[i].strip() != "" else None

    fab_records = {str(i): [] for i in range(1, 7)}  # tipi 1–6

    current_type2 = None # ci serve per i campi ripetuti del record 2.

    with open(fab_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 6:
                continue

            CODAMM = safe(parts, 0)
            SEZ = safe(parts, 1)
            IDIMMO = safe(parts, 2)
            TIPOIMMO = safe(parts, 3)
            PROGRES = safe(parts, 4)
            TIPOREC = safe(parts, 5)

            base = {
                "CODAMM": CODAMM,
                "SEZ": SEZ,
                "IDIMMO": IDIMMO,
                "TIPOIMMO": TIPOIMMO,
                "PROGRES": PROGRES,
                "TIPOREC": TIPOREC,
            }

            if TIPOREC == "1":
                fab_records["1"].append({
                    **base,
                    "ZONA": safe(parts, 6),
                    "CATEGORIA": safe(parts, 7),
                    "CLASSE": safe(parts, 8),
                    "CONSISTENZA": safe(parts, 9),
                    "SUPERFICIE": safe(parts, 10),
                    "RENDITA_EURO": safe(parts, 13),
                    "VALIMIS": safe(parts, 14),
                    "PIANI": "|".join([p for p in parts[18:22] if p]),
                    "DATAEFFINI": safe(parts, 29),
                    "DATAEFFFIN": safe(parts, 30),
                    "TIPONOTAINI": safe(parts, 31),
                    "NUMNOTAINI": safe(parts, 32),
                    "ANNO_NOTA": safe(parts, 34),
                    "PARTITAIMM": safe(parts, 42),
                    "IDMUTFIN": safe(parts, 44),
                })

            elif TIPOREC == "2":
                
                # Caso: inizia un nuovo record
                    if CODAMM and IDIMMO and PROGRES:
                        # Se c'era un record accumulato, lo chiudiamo
                        if current_type2 is not None:
                            fab_records["2"].append(current_type2)

                        # creiamo un nuovo contenitore
                        current_type2 = {
                            "CODAMM": CODAMM,
                            "SEZ": SEZ,
                            "IDIMMO": IDIMMO,
                            "TIPOIMMO": TIPOIMMO,
                            "PROGRES": PROGRES,
                            "COMUNE_CATASTALE": [],
                            "FOGLIO": [],
                            "PARTICELLA": [],
                            "SUBALTERNO": [],
                            "PM": []
                        }

                    # record sempre aggiuntivo (anche la prima riga)
                    comune   = safe(parts, 6)
                    foglio   = safe(parts, 7)
                    numpart = safe(parts, 8)
                    denom   = safe(parts, 9)
                    sub     = safe(parts, 10)

                    pm_fields = [safe(parts, i) for i in range(11, 21)]

                    

                    if comune:
                        current_type2["COMUNE_CATASTALE"].append(comune)
                    if foglio:
                        current_type2["FOGLIO"].append(foglio)
                    if numpart:
                        current_type2["PARTICELLA"].append(numpart)
                    if sub:
                        current_type2["SUBALTERNO"].append(sub)

                    pm_clean = [p for p in pm_fields if p]
                    if pm_clean:
                        current_type2["PM"].extend(pm_clean)

            elif TIPOREC == "3":
                fab_records["3"].append({
                    **base,
                    "TOPONIMO": safe(parts, 6),
                    "INDIRIZZO_ITA": safe(parts, 7),
                    "CIVICO": safe(parts, 9),
                })

            elif TIPOREC == "4":
                fab_records["4"].append({
                    **base,
                    "UCOM_CCPART": safe(parts, 6),
                    "UCOM_FOGLIO": safe(parts, 7),
                    "UCOM_NUMPART": safe(parts, 8),
                    "UCOM_SUB": safe(parts, 9),
                })

            elif TIPOREC == "5":
                fab_records["5"].append({
                    **base,
                    "RISERVA_COD": safe(parts, 6),
                    "RISERVA_DESCR": safe(parts, 7),
                })

            elif TIPOREC == "6":
                fab_records["6"].append({
                    **base,
                    "COD_ANN": safe(parts, 6),
                    "TESTOANN": safe(parts, 7),
                })

    # --- Conversione in DataFrame
    dfs = {k: pd.DataFrame(v) for k, v in fab_records.items() if len(v) > 0}
    for k, df in dfs.items():
        print(f"Type {k}: {len(df)} record")

    fab1 = dfs.get("1", pd.DataFrame())
    fab2 = dfs.get("2", pd.DataFrame())
    fab3 = dfs.get("3", pd.DataFrame())

    # --- Funzione di aggregazione
    def agg_text(df, cols, keycols=["CODAMM", "IDIMMO", "PROGRES"]):
        """Aggrega più colonne testuali in base alle chiavi principali"""
        if isinstance(cols, str):
            cols = [cols]
        grouped = df.groupby(keycols)[cols].agg(lambda x: "; ".join([str(i).strip() for i in x if pd.notna(i) and str(i).strip() not in ("", ";")])).reset_index()
        return grouped

    # --- Aggrega tutte le colonne utili del tipo 2 e 3 --- NO, CI SERVONO I RECORD NON AGGREGATI PER IL MERGE
    #fab2agg = agg_text(fab2, ["COMUNE_CATASTALE", "FOGLIO", "PARTICELLA", "SUBALTERNO"])
    #fab3agg = agg_text(fab3, ["INDIRIZZO_ITA"])

    # --- Merge per ottenere il flat completo
    flat = (
        fab1
        .merge(fab2, on=["CODAMM", "IDIMMO", "PROGRES"], how="inner")
        .merge(fab3, on=["CODAMM", "IDIMMO", "PROGRES"], how="inner")
    )

    # --- Convert TIPOREC 2 fields to integers
    int_cols = ["COMUNE_CATASTALE", "FOGLIO", "PARTICELLA", "SUBALTERNO"]

    for col in int_cols:
        if col in flat.columns:
            flat[col] = (
                flat[col]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)  # keep only digits
                .astype(float)                        # allows NaN
            )

    # --- Pulizia
    flat["RENDITA_EURO"] = (
        flat["RENDITA_EURO"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([\d.]+)", expand=False)
        .astype(float)
    )

    flat = flat.sort_values(["CODAMM", "IDIMMO", "PROGRES"])
    flat = flat.dropna(how="all", axis=1) 
    return flat

def xml_merge_fab(df_xml, df_fab):
    df_final = df_xml.merge(
    df_fab,
    how='inner',
    left_on=['comuneCatastale','foglio','numeratore','subalterno'],
    right_on=['COMUNE_CATASTALE','FOGLIO','PARTICELLA','SUBALTERNO'],
    suffixes=('_xml','_fab')
    )
    return df_final

def filter_df(df):
    columns = [
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
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    df_filtered = df[columns].copy()

    return df_filtered

def feature_cleaning(df):
    # Example cleaning: fill NaN with zeros for numeric columns
    boolean_cols = ['altriAccessoriAltro', 'intPorteInterneMetallo', 'intPorteIngressoLegnoTamburato', 'intPorteInterneLegnoTamburato', 'riscaldamento', 'condizionamento', 'videoCitofono', 'ascensoreServizio', 'acquaCalda', 'citofonico', 'camereParquet', 'cucinaBagnoPiastrelleCeramica', 'intPorteIngressoAltro', 'intPorteInterneLegnoMassello', 'estFinestreLegnoMassello', 'estVetroCameraLegnoMassello', 'altriAccessoriPiastrelleCeramica', 'intPorteIngressoMetallo', 'estFinestreAltro', 'estFinestreMetallo', 'camerePiastrelleCeramica', 'estVetroCameraAltro', 'intPorteIngressoLegnoMassello', 'camereAltro', 'cucinaBagnoParquet', 'altriAccessoriParquet', 'camereMarmo', 'cucinaBagnoMarmo', 'camereGomme', 'estVetroCameraMetallo', 'cucinaBagnoAltro', 'cucinaBagnoPiastrelleScaglie', 'altriAccessoriPiastrelleScaglie', 'ascensoreUsoEsclusivo', 'montacarichi', 'altriAccessoriGomme', 'estDoppioInfissoLegnoMassello', 'intPorteInterneAltro', 'altriAccessoriMarmo', 'altriAccessoriMoquette', 'camereMoquette', 'estDoppioInfissoAltro', 'cucinaBagnoGomme', 'camerePiastrelleScaglie', 'estDoppioInfissoMetallo', 'cucinaBagnoMoquette', 'VALIMIS']
    b_cols=df.columns.intersection(boolean_cols)
    df_final_clean = df.copy()
    for col in b_cols:
        # Map boolean columns: "s" -> 1, everything else -> 0
        df_final_clean[col] = (
            df_final_clean[col]
            .astype(str)              # convert everything to string
            .str.lower()              # normalize case
            .str.strip()              # remove whitespace
            .apply(lambda x: 1 if x == "s" else 0)  # map "s" to 1, everything else to 0
            .astype(int)              # final numeric dtype
        )
    categorical_cols = ['source_file', 'categoriaImmobiliare', 'sottoCategoriaImmobiliare', 'lista_piani', 'datiMetriciNettiManuali', 'tipoRiferimento', 'accessoCarrabile', 'postoAutoScoperto', 'CATEGORIA', 'CONSISTENZA', 'PIANI']  # Columns with text/category values
    numeric_cols = ['annoRiferimento', 'spessoreMuri', 'superficieLordaMq', 'num', 'superficieUtileMq', 'bagniNum', 'bagniSuperficieUtileMq', 'corridoiNum', 'corridoiSuperficieUtileMq', 'superficieLordaNonComunicantiMq', 'superficieLordaComunicantiMq', 'altezzaMediaUtileCm', 'riscaldamento', 'acquaCalda', 'camereParquet', 'cucinaBagnoPiastrelleCeramica', 'intPorteIngressoLegnoTamburato', 'intPorteInterneLegnoTamburato', 'estVetroCameraAltro', 'denominatore', 'numeroPiano', 'superficieMq', 'giardinoSuperficieLordaMq', 'altezzaMediaLocaliPrincipaliCm', 'altriAccessoriAltro', 'intPorteIngressoAltro', 'estVetroCameraLegnoMassello', 'altriAccessoriPiastrelleCeramica', 'intPorteIngressoLegnoMassello', 'intPorteInterneLegnoMassello', 'superficieMqVaniAventiAltezzaMediaMinore230Cm', 'estDoppioInfissoLegnoMassello', 'estFinestreLegnoMassello', 'condizionamento', 'camereMarmo', 'cucinaBagnoMarmo', 'camereGomme', 'citofonico', 'estVetroCameraMetallo', 'intPorteInterneAltro', 'intPorteIngressoMetallo', 'cucinaBagnoParquet', 'videoCitofono', 'altriAccessoriGomme', 'camerePiastrelleCeramica', 'altriAccessoriParquet', 'camereAltro', 'cucinaBagnoAltro', 'estFinestreAltro', 'pianiFuoriTerraNum', 'pianiFuoriTerraMc', 'pianiEntroTerraNum', 'pianiEntroTerraMc', 'camereMoquette', 'estFinestreMetallo', 'intPorteInterneMetallo', 'ascensoriNumero', 'ascensoreUsoEsclusivo', 'cucinaBagnoPiastrelleScaglie', 'altriAccessoriPiastrelleScaglie', 'estDoppioInfissoAltro', 'cucinaBagnoGomme', 'altriAccessoriMoquette', 'ascensoreServizio', 'camerePiastrelleScaglie', 'altriAccessoriMarmo', 'estDoppioInfissoMetallo', 'altroSuperficieLordaMq', 'montacarichi', 'cucinaBagnoMoquette', 'ZONA', 'CLASSE', 'SUPERFICIE', 'RENDITA_EURO', 'VALIMIS']      # Columns with numeric values
    list_col = 'PM'         # Columns containing lists (cannot be used in RF)
    if list_col in df_final_clean.columns:
        df_final_clean = df_final_clean.drop(columns=list_col)

    median_dict = {'annoRiferimento': 2017.0, 'spessoreMuri': 1.0, 'superficieLordaMq': 32.0, 'num': 3.0, 'superficieUtileMq': 63.0, 'bagniNum': 1.0, 'bagniSuperficieUtileMq': 7.0, 'corridoiNum': 2.0, 'corridoiSuperficieUtileMq': 11.0, 'superficieLordaNonComunicantiMq': 17.0, 'superficieLordaComunicantiMq': 15.0, 'altezzaMediaUtileCm': 265.0, 'riscaldamento': 0.0, 'acquaCalda': 0.0, 'camereParquet': 0.0, 'cucinaBagnoPiastrelleCeramica': 0.0, 'intPorteIngressoLegnoTamburato': 0.0, 'intPorteInterneLegnoTamburato': 0.0, 'estVetroCameraAltro': 0.0, 'denominatore': 1.0, 'numeroPiano': 0.0, 'superficieMq': 14.0, 'giardinoSuperficieLordaMq': 108.0, 'altezzaMediaLocaliPrincipaliCm': 250.0, 'altriAccessoriAltro': 0.0, 'intPorteIngressoAltro': 0.0, 'estVetroCameraLegnoMassello': 0.0, 'altriAccessoriPiastrelleCeramica': 0.0, 'intPorteIngressoLegnoMassello': 0.0, 'intPorteInterneLegnoMassello': 0.0, 'superficieMqVaniAventiAltezzaMediaMinore230Cm': 20.0, 'estDoppioInfissoLegnoMassello': 0.0, 'estFinestreLegnoMassello': 0.0, 'condizionamento': 0.0, 'camereMarmo': 0.0, 'cucinaBagnoMarmo': 0.0, 'camereGomme': 0.0, 'citofonico': 0.0, 'estVetroCameraMetallo': 0.0, 'intPorteInterneAltro': 0.0, 'intPorteIngressoMetallo': 0.0, 'cucinaBagnoParquet': 0.0, 'videoCitofono': 0.0, 'altriAccessoriGomme': 0.0, 'camerePiastrelleCeramica': 0.0, 'altriAccessoriParquet': 0.0, 'camereAltro': 0.0, 'cucinaBagnoAltro': 0.0, 'estFinestreAltro': 0.0, 'pianiFuoriTerraNum': 2.5, 'pianiFuoriTerraMc': 3065.0, 'pianiEntroTerraNum': 1.0, 'pianiEntroTerraMc': 1874.5, 'camereMoquette': 0.0, 'estFinestreMetallo': 0.0, 'intPorteInterneMetallo': 0.0, 'ascensoriNumero': 1.0, 'ascensoreUsoEsclusivo': 0.0, 'cucinaBagnoPiastrelleScaglie': 0.0, 'altriAccessoriPiastrelleScaglie': 0.0, 'estDoppioInfissoAltro': 0.0, 'cucinaBagnoGomme': 0.0, 'altriAccessoriMoquette': 0.0, 'ascensoreServizio': 0.0, 'camerePiastrelleScaglie': 0.0, 'altriAccessoriMarmo': 0.0, 'estDoppioInfissoMetallo': 0.0, 'altroSuperficieLordaMq': 47.0, 'montacarichi': 0.0, 'cucinaBagnoMoquette': 0.0, 'ZONA': 2.0, 'CLASSE': 3.0, 'SUPERFICIE': 41.0, 'RENDITA_EURO': 19027.68, 'VALIMIS': 0.0}
    
    for col in numeric_cols:
        if col in df_final_clean.columns:
            # Convert to numeric type, converting errors to NaN
            df_final_clean[col] = pd.to_numeric(df_final_clean[col], errors='coerce')
            m = median_dict[col]
            df_final_clean[col] = df_final_clean[col].fillna(m)
    

    # Load JSON with class lists
    plugin_dir = os.path.dirname(__file__)
    label_path = os.path.join(plugin_dir, "label_classes.json")

    with open(label_path, "r") as f:
        encoder_mapping = json.load(f)   # dict: {column_name: [class1, class2, ...]}

    # Encoding loop
    for col in categorical_cols:
        if col in df_final_clean.columns:

            if col not in encoder_mapping:
                print(f"[WARNING] No encoder found for column: {col}")
                continue

            classes_list = encoder_mapping[col]   # e.g. ["A01","A02","MISSING"]

            # Safe text + missing
            df_final_clean[col] = df_final_clean[col].astype(str).fillna("MISSING")

            # Ensure "MISSING" is in classes_list
            if "MISSING" not in classes_list:
                classes_list.append("MISSING")

            # Build map category → int
            class_to_int = {cls: i for i, cls in enumerate(classes_list)}

            # Apply mapping, fallback to MISSING
            df_final_clean[col] = df_final_clean[col].apply(
                lambda x: class_to_int.get(x, class_to_int["MISSING"])
            )

    return df_final_clean
