import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    categorical_cols = []  # Columns with text/category values
    numeric_cols = []      # Columns with numeric values
    list_cols = []         # Columns containing lists (cannot be used in RF)
    other_cols = []        # Other types

    # Iterate through all columns and categorize them
    for col in df_final_clean.columns:
        dtype = df_final_clean[col].dtype
        # Get a sample value to check if it's a list
        sample_val = df_final_clean[col].dropna().iloc[0] if len(df_final_clean[col].dropna()) > 0 else None
        
        if isinstance(sample_val, list):
            # Column contains lists - cannot be used directly
            list_cols.append(col)
        elif dtype == 'object':
            # Object type: could be categorical or numeric stored as string
            # Try to convert to numeric to check
            try:
                pd.to_numeric(df_final_clean[col].dropna().head(10))
                # Successfully converted - it's numeric stored as string
                numeric_cols.append(col)
            except:
                # Cannot convert - it's categorical
                categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            # Already numeric
            numeric_cols.append(col)
        else:
            # Other types
            other_cols.append(col)
    
    df_final_clean = df_final_clean.drop(columns=list_cols)

    for col in numeric_cols:
        if col in df_final_clean.columns:
            # Convert to numeric type, converting errors to NaN
            df_final_clean[col] = pd.to_numeric(df_final_clean[col], errors='coerce')
            df_final_clean[col] = df_final_clean[col].fillna(0)

    return df_final_clean
