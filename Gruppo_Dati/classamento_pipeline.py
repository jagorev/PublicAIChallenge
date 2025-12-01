"""
Clean Pipeline for Catasto Problem - Training Script
Extracted from clean_pipeline.ipynb
"""

import pandas as pd
import os
import numpy as np
import pickle
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DATA DIRECTORY SETUP
# ============================================================================

DATA_DIR = "data"

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✅ Created directory: {DATA_DIR}")

# ============================================================================
# STEP 1: Read and process FAB file
# ============================================================================

def safe(parts, i):
    return parts[i].strip() if i < len(parts) and parts[i].strip() != "" else None

def read_fab_file(fab_file):
    """Read FAB file and create dataframes for each record type."""
    fab_records = {str(i): [] for i in range(1, 7)}
    current_type2 = None

    with open(fab_file, encoding="utf-8", errors="ignore") as f:
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
                if CODAMM and IDIMMO and PROGRES:
                    if current_type2 is not None:
                        fab_records["2"].append(current_type2)
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

                comune = safe(parts, 6)
                foglio = safe(parts, 7)
                numpart = safe(parts, 8)
                denom = safe(parts, 9)
                sub = safe(parts, 10)
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

    # Convert to DataFrames
    dfs = {k: pd.DataFrame(v) for k, v in fab_records.items() if len(v) > 0}
    return dfs

def agg_mixed(series):
    """Aggregate mixed-type series (lists or strings)."""
    if series.apply(lambda x: isinstance(x, list)).any():
        out = []
        for v in series:
            if isinstance(v, list):
                out.extend(v)
        clean = [i for i in out if i not in ("", None)]
        seen = set()
        dedup = [x for x in clean if not (x in seen or seen.add(x))]
        return dedup
    return "; ".join({str(i).strip() for i in series if pd.notna(i) and str(i).strip() != ""})

def process_fab_data(fab_file):
    """Process FAB file and return flat dataframe."""
    dfs = read_fab_file(fab_file)
    fab1 = dfs.get("1", pd.DataFrame())
    fab2 = dfs.get("2", pd.DataFrame())
    fab3 = dfs.get("3", pd.DataFrame())

    # Merge records
    flat = (
        fab1
        .merge(fab2, on=["CODAMM", "IDIMMO", "PROGRES"], how="inner")
        .merge(fab3, on=["CODAMM", "IDIMMO", "PROGRES"], how="inner")
    )

    # Aggregate duplicated records
    keycols = ["CODAMM", "IDIMMO", "PROGRES"]
    value_cols = [c for c in flat.columns if c not in keycols]
    flat = flat.groupby(keycols)[value_cols].agg(agg_mixed).reset_index()

    # Convert to integers
    int_cols = ["COMUNE_CATASTALE", "FOGLIO", "PARTICELLA", "SUBALTERNO"]
    for col in int_cols:
        if col in flat.columns:
            flat[col] = (
                flat[col]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )

    # Clean RENDITA_EURO
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

def load_or_create_flat(fab_file):
    """Load flat dataframe from CSV or create it from FAB file."""
    flat_path = os.path.join(DATA_DIR, "flat.csv")
    
    if os.path.exists(flat_path):
        print(f"📂 Loading flat dataframe from {flat_path}...")
        flat = pd.read_csv(flat_path, low_memory=False)
        print(f"✅ Loaded {len(flat)} records from CSV")
        return flat
    else:
        print(f"📝 Creating flat dataframe from FAB file...")
        flat = process_fab_data(fab_file)
        print(f"💾 Saving flat dataframe to {flat_path}...")
        flat.to_csv(flat_path, index=False)
        print(f"✅ Saved {len(flat)} records to CSV")
        return flat

# ============================================================================
# STEP 2: Read and process XML files
# ============================================================================

def parse_apartments_from_folder(folder_path):
    """Parse XML files and extract apartment data."""
    all_rows = []

    for filename in os.listdir(folder_path):
        if not filename.endswith('.xml'):
            continue
        
        xml_file = os.path.join(folder_path, filename)

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            elenco_ui = root.find('.//ElencoUI')
            if elenco_ui is None:
                continue

            ui_elements = elenco_ui.findall('.//UICostituzione')

            for ui in ui_elements:
                row = {"source_file": filename}

                # Identificativo Catastale PM
                idpm = ui.find('.//ElencoIdentificativiCatastaliPM/IdentificativoCatastalePM')
                if idpm is not None:
                    for k, v in idpm.attrib.items():
                        row[k] = v

                # Classamento
                classamento = ui.find('.//Classamento')
                if classamento is not None:
                    for k, v in classamento.attrib.items():
                        row[k] = v

                # Indirizzo
                indir = ui.find('.//ElencoIndirizzi/Indirizzo')
                if indir is not None:
                    for k, v in indir.attrib.items():
                        row[k] = v

                # Piani
                piani = ui.findall('.//ElencoPiani/Piano')
                row["lista_piani"] = ";".join([p.attrib.get("numeroPiano", "") for p in piani])

                # Mod1N-2
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
            print(f"Errore parsing {filename}: {e}")
            continue

    df_xml = pd.DataFrame(all_rows)

    # Convert cadastral identifiers to nullable integers
    int_cols = ['comuneCatastale', 'foglio', 'numeratore', 'subalterno']
    for col in int_cols:
        if col in df_xml.columns:
            df_xml[col] = (
                df_xml[col]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )

    return df_xml

def load_or_create_xml(xml_folder):
    """Load XML dataframe from CSV or create it from XML files."""
    xml_path = os.path.join(DATA_DIR, "df_xml.csv")
    
    if os.path.exists(xml_path):
        print(f"📂 Loading XML dataframe from {xml_path}...")
        df_xml = pd.read_csv(xml_path, low_memory=False)
        print(f"✅ Loaded {len(df_xml)} records from CSV")
        return df_xml
    else:
        print(f"📝 Creating XML dataframe from XML files...")
        df_xml = parse_apartments_from_folder(xml_folder)
        print(f"💾 Saving XML dataframe to {xml_path}...")
        df_xml.to_csv(xml_path, index=False)
        print(f"✅ Saved {len(df_xml)} records to CSV")
        return df_xml

# ============================================================================
# STEP 3: Merge and clean data
# ============================================================================

def clean_dataframe(df_final):
    """Clean the merged dataframe."""
    # Drop columns
    columns_to_drop = ['TIPOREC_x', 'TIPOREC_y', 'TIPOIMMO_x', 'TIPOIMMO_y', 'source_file', 'SEZ_y']
    columns_to_drop_location = [
        'codiceVia', 'indirizzoIT', 'civico1', 'civico2', 'civico3', 'CIVICO',
        'foglio', 'FOGLIO', 'numeratore', 'PARTICELLA', 'subalterno', 'SUBALTERNO',
        'comuneCatastale', 'COMUNE_CATASTALE', 'INDIRIZZO_ITA'
    ]
    columns_to_drop_other = [
        'CODAMM', 'IDIMMO', 'PROGRES', 'TOPONIMO', 'TIPONOTAINI', 'NUMNOTAINI',
        'ANNO_NOTA', 'IDMUTFIN', 'DATAEFFINI', 'DATAEFFFIN',
        'estAltroDescrizione', 'intAltroDescrizione', 'altroDescrizione',
        'pavimentazioneAltroDescrizione'
    ]

    df_clean = df_final.drop(columns=columns_to_drop, axis=1, errors='ignore')
    df_clean = df_clean.drop(columns=columns_to_drop_location, axis=1, errors='ignore')
    df_clean = df_clean.drop(columns=columns_to_drop_other, axis=1, errors='ignore')

    # Remove columns with single unique value and remap boolean columns
    columns_unique_to_drop = []
    columns_boolean = []
    
    for col in list(df_clean):
        try:
            unique_vals = df_clean[col].unique()
            unique_count = len(unique_vals)
        except TypeError:
            unique_vals = df_clean[col].apply(lambda x: str(x) if isinstance(x, list) else x).unique()
            unique_count = len(unique_vals)
        
        if unique_count == 1:
            columns_unique_to_drop.append(col)
        if unique_count == 2:
            columns_boolean.append(col)

    df_clean = df_clean.drop(columns=columns_unique_to_drop, axis=1)

    # Map boolean columns
    pd.set_option('future.no_silent_downcasting', True)
    for col in columns_boolean:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .apply(lambda x: 1 if x == "s" else 0)
            .astype(int)
        )

    # Filter categories A, B, C
    df_clean = df_clean[df_clean['CATEGORIA'].str.startswith(('A', 'B', 'C'))]

    return df_clean

# ============================================================================
# STEP 4: Preprocessing for model
# ============================================================================

def identify_column_types(df):
    """Identify categorical, numeric, and list columns."""
    categorical_cols = []
    numeric_cols = []
    list_cols = []
    other_cols = []

    for col in df.columns:
        dtype = df[col].dtype
        sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
        
        if isinstance(sample_val, list):
            list_cols.append(col)
        elif dtype == 'object':
            try:
                pd.to_numeric(df[col].dropna().head(10))
                numeric_cols.append(col)
            except:
                categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        else:
            other_cols.append(col)

    return categorical_cols, numeric_cols, list_cols, other_cols

def preprocess_data(df_final_clean, categorical_cols, numeric_cols, list_cols):
    """Preprocess data for model training."""
    df_model = df_final_clean.copy()

    # Drop list columns
    if list_cols:
        df_model = df_model.drop(columns=list_cols)

    # Process numeric columns
    for col in numeric_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
            median_val = df_model[col].median()
            if pd.notna(median_val):
                df_model[col] = df_model[col].fillna(median_val)
            else:
                df_model[col] = df_model[col].fillna(0)

    # Process categorical columns
    label_encoders = {}
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = df_model[col].fillna('MISSING')
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le

    return df_model, label_encoders

def load_or_create_model_data(df_final_clean):
    """Load preprocessed model dataframe from CSV or create it."""
    model_path = os.path.join(DATA_DIR, "df_model.csv")
    encoders_path = os.path.join(DATA_DIR, "label_encoders.pkl")
    
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        print(f"📂 Loading preprocessed model dataframe from {model_path}...")
        df_model = pd.read_csv(model_path, low_memory=False)
        print(f"📂 Loading label encoders from {encoders_path}...")
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        print(f"✅ Loaded {len(df_model)} records and {len(label_encoders)} encoders from CSV/pickle")
        return df_model, label_encoders
    else:
        print(f"📝 Creating preprocessed model dataframe...")
        categorical_cols, numeric_cols, list_cols, other_cols = identify_column_types(df_final_clean)
        df_model, label_encoders = preprocess_data(df_final_clean, categorical_cols, numeric_cols, list_cols)
        print(f"💾 Saving preprocessed model dataframe to {model_path}...")
        df_model.to_csv(model_path, index=False)
        print(f"💾 Saving label encoders to {encoders_path}...")
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"✅ Saved {len(df_model)} records to CSV and {len(label_encoders)} encoders to pickle")
        return df_model, label_encoders

# ============================================================================
# STEP 5: Prepare data splits and apply augmentation
# ============================================================================

def prepare_xy(df_split, target_column, exclude_columns):
    """Prepare features (X) and target (y) from dataframe."""
    available_cols = list(df_split.columns)
    
    if target_column not in available_cols:
        raise ValueError(f"Target '{target_column}' not found!")
    
    X_cols = [c for c in available_cols if c != target_column and c not in exclude_columns]
    X = df_split[X_cols].copy()
    y = df_split[target_column].copy()
    
    return X, y

def apply_simple_augmentation(df_model, min_examples_threshold=50, target_examples=100, random_state=42):
    """
    Simple augmentation: generates synthetic data by sampling from normal distribution
    for numeric columns and random sampling for categorical columns.
    Balances CATEGORIA+CLASSE combinations.
    
    This is an alternative to SMOTENC that may work better in some cases.
    """
    np.random.seed(random_state)
    
    print(f"\n📊 Simple Augmentation (balancing CATEGORIA+CLASSE combinations)...")
    print(f"   Threshold: <{min_examples_threshold} examples = rare")
    print(f"   Target: {target_examples} examples per combination")
    
    # Identify rare combinations
    class_counts = df_model.groupby(['CATEGORIA', 'CLASSE']).size().reset_index(name='count')
    rare_classes = class_counts[class_counts['count'] < min_examples_threshold]
    
    print(f"   Total combinations: {len(class_counts)}")
    print(f"   Rare combinations: {len(rare_classes)}")
    
    if len(rare_classes) == 0:
        print("   ✅ No rare combinations found, skipping augmentation")
        df_augmented = df_model.copy()
        df_augmented['is_augmented'] = 0
        return df_augmented
    
    def augment_class(df, categoria, classe, n_synthetic):
        """Generate n_synthetic examples for CATEGORIA+CLASSE combination"""
        mask = (df['CATEGORIA'] == categoria) & (df['CLASSE'] == classe)
        class_data = df[mask].copy()
        
        if len(class_data) == 0:
            return pd.DataFrame()
        
        synthetic_records = []
        
        for _ in range(n_synthetic):
            new_record = {}
            
            # For each column, generate synthetic value
            for col in df.columns:
                if col in ['CATEGORIA', 'CLASSE']:
                    # Keep target values fixed
                    new_record[col] = categoria if col == 'CATEGORIA' else classe
                else:
                    values = class_data[col].dropna()
                    if len(values) == 0:
                        new_record[col] = 0
                        continue
                    
                    # If numeric, sample from normal distribution
                    if pd.api.types.is_numeric_dtype(df[col]):
                        mean = values.mean()
                        std = values.std()
                        
                        if pd.isna(std) or std == 0:
                            new_val = mean
                        else:
                            new_val = np.random.normal(mean, std)
                            
                            # Domain constraints
                            if 'superficie' in col.lower() or 'mq' in col.lower():
                                new_val = max(0, new_val)
                            elif 'num' in col.lower() or col.lower().endswith('_encoded'):
                                new_val = max(0, int(round(new_val)))
                            elif 'anno' in col.lower():
                                new_val = int(round(np.clip(new_val, 1800, 2024)))
                        
                        new_record[col] = new_val
                    else:
                        # If categorical, random sample
                        new_record[col] = np.random.choice(values)
            
            synthetic_records.append(new_record)
        
        return pd.DataFrame(synthetic_records)
    
    # Generate synthetic data for rare combinations
    synthetic_dfs = []
    
    for _, row in rare_classes.iterrows():
        categoria = row['CATEGORIA']
        classe = row['CLASSE']
        current_count = row['count']
        
        n_to_generate = max(0, target_examples - current_count)
        n_to_generate = int(n_to_generate)
        
        if n_to_generate > 0:
            synthetic_df = augment_class(df_model, categoria, classe, n_to_generate)
            if len(synthetic_df) > 0:
                synthetic_dfs.append(synthetic_df)
    
    # Combine original + synthetic
    if synthetic_dfs:
        df_synthetic = pd.concat(synthetic_dfs, ignore_index=True)
        df_model_copy = df_model.copy()
        df_model_copy['is_augmented'] = 0
        df_synthetic['is_augmented'] = 1
        df_augmented = pd.concat([df_model_copy, df_synthetic], ignore_index=True)
        
        n_original = len(df_model_copy)
        n_synthetic = len(df_synthetic)
        print(f"   ✅ Augmentation completed:")
        print(f"      Original: {n_original} samples")
        print(f"      Synthetic: {n_synthetic} samples (+{n_synthetic/n_original*100:.1f}%)")
        print(f"      Total: {len(df_augmented)} samples")
    else:
        print("   ⚠️  No synthetic data generated")
        df_augmented = df_model.copy()
        df_augmented['is_augmented'] = 0
    
    return df_augmented

# ============================================================================
# STEP 6: Train model
# ============================================================================

def train_model(X_train, y_train, model_name="Model"):
    """
    Train Random Forest model using best parameters found previously (no grid search).
    
    Args:
        X_train: Training features (may include augmented data)
        y_train: Training labels
        model_name: Name for logging purposes
    
    Returns:
        rf: Trained RandomForestClassifier model
        class_weight_dict: Dictionary of class weights used for training
    """
    # Remove is_augmented column if present (it's not a feature)
    if 'is_augmented' in X_train.columns:
        X_train_clean = X_train.drop(columns=['is_augmented'])
    else:
        X_train_clean = X_train.copy()
    
    # Calculate class weights
    classes = np.unique(y_train)
    class_weights_balanced = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, class_weights_balanced)}

    # Identify rare classes
    class_counts_train = pd.Series(y_train).value_counts()
    rare_classes = class_counts_train[class_counts_train < 100].index.tolist()
    very_rare_classes = class_counts_train[class_counts_train < 20].index.tolist()

    # Increase weights for rare classes
    for cls in rare_classes:
        if cls in class_weight_dict:
            original_weight = class_weight_dict[cls]
            if cls in very_rare_classes:
                class_weight_dict[cls] = original_weight * 20.0
            else:
                class_weight_dict[cls] = original_weight * 10.0

    # Use best parameters found previously (no grid search for speed)
    print(f"\n🔍 Training {model_name} with best parameters (no grid search)...")
    print("   Parameters: n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=1, max_features='sqrt'")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=60,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight=class_weight_dict,
        n_jobs=-1,
        random_state=42
    )

    # Fit on training data
    rf.fit(X_train_clean, y_train)
    
    print(f"✅ Model trained successfully")

    return rf, class_weight_dict

def group_rare_categories_early(df_model, label_encoders, min_samples=5, target_column="CATEGORIA"):
    """
    Raggruppa le categorie rare a monte, subito dopo il caricamento del dataframe.
    Categorie con <min_samples esempi → "A", "B", o "C" in base alla prima lettera.
    DOPO questo cambio, crea un nuovo label encoder.
    
    Args:
        df_model: DataFrame con i dati (con CATEGORIA già encoded)
        label_encoders: Dizionario con i label encoders
        min_samples: Soglia minima di esempi (default: 5)
        target_column: Nome della colonna target (default: "CATEGORIA")
    
    Returns:
        df_grouped: DataFrame con categorie raggruppate (con CATEGORIA come stringa)
        label_encoders: Label encoders aggiornati
    """
    if target_column not in label_encoders:
        print(f"⚠️  Label encoder for '{target_column}' not found. Skipping grouping.")
        return df_model.copy(), label_encoders
    
    le_categoria = label_encoders[target_column]
    
    # Decode tutte le categorie a stringhe
    df_grouped = df_model.copy()
    df_grouped[target_column] = le_categoria.inverse_transform(df_grouped[target_column])
    
    # Get class counts (ora sono stringhe)
    counts = df_grouped[target_column].value_counts()
    
    # Identifica classi rare (<min_samples)
    rare_classes = counts[counts < min_samples].index.tolist()
    
    if len(rare_classes) == 0:
        print(f"\n✅ No rare categories found (<{min_samples} samples). No grouping needed.")
        # Ricrea label encoder
        new_le = LabelEncoder()
        new_le.fit(df_grouped[target_column].unique())
        label_encoders[target_column] = new_le
        return df_grouped, label_encoders
    
    print(f"\n📊 Grouping rare categories (<{min_samples} samples) by type (A/B/C)...")
    print(f"   Found {len(rare_classes)} rare categories: {sorted(rare_classes)}")
    
    # Raggruppa: cambia le categorie rare in A, B, o C in base alla prima lettera
    for rare_cat in rare_classes:
        if rare_cat.startswith('A'):
            df_grouped.loc[df_grouped[target_column] == rare_cat, target_column] = 'A'
        elif rare_cat.startswith('B'):
            df_grouped.loc[df_grouped[target_column] == rare_cat, target_column] = 'B'
        elif rare_cat.startswith('C'):
            df_grouped.loc[df_grouped[target_column] == rare_cat, target_column] = 'C'
    
    # Mostra statistiche
    rare_by_type = {'A': [], 'B': [], 'C': []}
    for rare_cat in rare_classes:
        if rare_cat.startswith('A'):
            rare_by_type['A'].append(rare_cat)
        elif rare_cat.startswith('B'):
            rare_by_type['B'].append(rare_cat)
        elif rare_cat.startswith('C'):
            rare_by_type['C'].append(rare_cat)
    
    print(f"\n   Rare categories by type:")
    for cat_type in ['A', 'B', 'C']:
        if rare_by_type[cat_type]:
            n_samples = (df_grouped[target_column] == cat_type).sum()
            print(f"      Type {cat_type}: {len(rare_by_type[cat_type])} rare categories → '{cat_type}' ({n_samples} campioni)")
    
    # Crea nuovo label encoder sul dataframe modificato
    new_le = LabelEncoder()
    new_le.fit(df_grouped[target_column].unique())
    
    # Encode di nuovo le categorie
    df_grouped[target_column] = new_le.transform(df_grouped[target_column])
    
    # Update label encoder
    label_encoders[target_column] = new_le
    
    print(f"\n✅ Grouped rare categories:")
    print(f"   Total classes: {len(new_le.classes_)}")
    
    return df_grouped.reset_index(drop=True), label_encoders

def group_rare_classes_by_category(df_model, label_encoders, min_samples=20, target_column="CATEGORIA"):
    """
    Raggruppa classi rare per categoria (A, B, C).
    Classi rare di categoria A → "altro_A"
    Classi rare di categoria B → "altro_B"
    Classi rare di categoria C → "altro_C"
    Raggruppa SOLO se ci sono almeno 2 classi rare della stessa categoria.
    """
    df_grouped = df_model.copy()
    
    # Get label encoder for CATEGORIA
    if target_column not in label_encoders:
        print(f"⚠️  Label encoder for '{target_column}' not found. Skipping grouping.")
        return df_grouped, label_encoders
    
    le_categoria = label_encoders[target_column]
    
    # Get class counts
    counts = df_grouped[target_column].value_counts()
    
    # Identify rare classes
    rare_classes = counts[counts < min_samples].index.tolist()
    
    if len(rare_classes) == 0:
        print(f"\n✅ No rare classes found (<{min_samples} samples). No grouping needed.")
        return df_grouped, label_encoders
    
    print(f"\n📊 Grouping rare classes (<{min_samples} samples) by category...")
    print(f"   Found {len(rare_classes)} rare classes: {sorted(rare_classes)}")
    
    # Decode numeric classes to original category names
    all_classes = sorted(df_grouped[target_column].unique())
    class_to_category = {}
    for num_class in all_classes:
        try:
            original_cat = le_categoria.inverse_transform([int(num_class)])[0]
            class_to_category[num_class] = original_cat
        except:
            continue
    
    # Group rare classes by category (A, B, C)
    rare_by_category = {'A': [], 'B': [], 'C': []}
    
    for num_class in rare_classes:
        if num_class in class_to_category:
            original_cat = class_to_category[num_class]
            if original_cat.startswith('A'):
                rare_by_category['A'].append(num_class)
            elif original_cat.startswith('B'):
                rare_by_category['B'].append(num_class)
            elif original_cat.startswith('C'):
                rare_by_category['C'].append(num_class)
    
    print(f"\n   Rare classes by category:")
    for cat in ['A', 'B', 'C']:
        if rare_by_category[cat]:
            print(f"      Category {cat}: {sorted(rare_by_category[cat])}")
    
    # Create new label encoder with grouped classes
    common_classes = counts[counts >= min_samples].index.tolist()
    
    # Build new class names: common classes + single rare classes (kept separate) + "altro_X" (for groups)
    new_class_names = []
    
    # Add common classes (keep as is)
    for num_class in sorted(common_classes):
        if num_class in class_to_category:
            original_name = class_to_category[num_class]
            new_class_names.append(original_name)
    
    # Add single rare classes (keep them separate, don't group)
    for cat in ['A', 'B', 'C']:
        if len(rare_by_category[cat]) == 1:
            single_rare_class = rare_by_category[cat][0]
            if single_rare_class in class_to_category:
                original_name = class_to_category[single_rare_class]
                new_class_names.append(original_name)
                print(f"   ⚠️  Category {cat}: Only 1 rare class ({single_rare_class}, '{original_name}'), keeping it separate (not grouping)")
    
    # Add "altro_A", "altro_B", "altro_C" ONLY if there are at least 2 rare classes of that category
    altro_indices = {}
    for cat in ['A', 'B', 'C']:
        if len(rare_by_category[cat]) >= 2:  # Only group if at least 2 rare classes
            altro_name = f"altro_{cat}"
            new_class_names.append(altro_name)
            altro_indices[cat] = len(new_class_names) - 1
    
    # Create new label encoder
    new_le = LabelEncoder()
    new_le.fit(new_class_names)
    
    # Apply grouping to dataframe (only groups with >=2 classes)
    # IMPORTANT: Do this AFTER creating the new encoder, but BEFORE remapping common classes
    for cat in ['A', 'B', 'C']:
        if len(rare_by_category[cat]) >= 2:  # Only group if at least 2 rare classes
            altro_name = f"altro_{cat}"
            altro_encoded = new_le.transform([altro_name])[0]
            mask = df_grouped[target_column].isin(rare_by_category[cat])
            n_samples_before = mask.sum()
            print(f"\n📊 DEBUG: Raggruppamento 'altro_{cat}':")
            print(f"   Classi rare da raggruppare: {sorted(rare_by_category[cat])}")
            print(f"   Campioni trovati con mask: {n_samples_before}")
            if n_samples_before == 0:
                print(f"   ⚠️  ATTENZIONE: Nessun campione trovato per queste classi!")
                for rare_class in rare_by_category[cat]:
                    count_in_df = (df_grouped[target_column] == rare_class).sum()
                    try:
                        cat_name = le_categoria.inverse_transform([int(rare_class)])[0]
                        print(f"      Classe {rare_class} ({cat_name}): {count_in_df} campioni nel dataframe")
                    except:
                        print(f"      Classe {rare_class}: {count_in_df} campioni nel dataframe")
            # Use .copy() to avoid SettingWithCopyWarning and ensure the assignment works
            df_grouped = df_grouped.copy()
            df_grouped.loc[mask, target_column] = altro_encoded
            # DEBUG: Verifica subito dopo l'assegnazione
            n_samples_after = (df_grouped[target_column] == altro_encoded).sum()
            print(f"   Campioni dopo assegnazione: {n_samples_after}")
            if n_samples_after != n_samples_before:
                print(f"   ⚠️  PROBLEMA: Perduti {n_samples_before - n_samples_after} campioni durante l'assegnazione!")
        elif len(rare_by_category[cat]) == 1:
            single_rare_class = rare_by_category[cat][0]
            if single_rare_class in class_to_category:
                original_name = class_to_category[single_rare_class]
                new_encoded = new_le.transform([original_name])[0]
                mask = df_grouped[target_column] == single_rare_class
                df_grouped.loc[mask, target_column] = new_encoded
    
    # Also remap common classes to their new encoded values
    # IMPORTANT: Only remap classes that are NOT in the rare classes that were grouped
    all_rare_classes_grouped = []
    for cat in ['A', 'B', 'C']:
        if len(rare_by_category[cat]) >= 2:
            all_rare_classes_grouped.extend(rare_by_category[cat])
    
    # Get the encoded values for "altro_X" to avoid overwriting them
    altro_encoded_values = set()
    for cat in ['A', 'B', 'C']:
        if len(rare_by_category[cat]) >= 2:
            altro_name = f"altro_{cat}"
            altro_encoded = new_le.transform([altro_name])[0]
            altro_encoded_values.add(altro_encoded)
    
    # Only remap common classes that are NOT in the rare classes list
    for num_class in common_classes:
        if num_class in class_to_category and num_class not in all_rare_classes_grouped:
            original_name = class_to_category[num_class]
            new_encoded = new_le.transform([original_name])[0]
            
            # Only remap if the new_encoded is NOT an "altro_X" value
            if new_encoded not in altro_encoded_values:
                mask = df_grouped[target_column] == num_class
                # Exclude rows that already have "altro_X" values
                mask = mask & ~df_grouped[target_column].isin(altro_encoded_values)
                if mask.sum() > 0:
                    df_grouped.loc[mask, target_column] = new_encoded
    
    # Update label encoder
    label_encoders[target_column] = new_le
    
    print(f"\n✅ Grouped rare classes:")
    grouped_count = 0
    for cat in ['A', 'B', 'C']:
        if len(rare_by_category[cat]) >= 2:
            altro_name = f"altro_{cat}"
            altro_encoded = new_le.transform([altro_name])[0]
            # Count how many samples ended up in this "altro_X" group
            altro_count = (df_grouped[target_column] == altro_encoded).sum()
            
            # DEBUG: Verifica cosa c'è nel dataframe
            print(f"\n📊 DEBUG: Verifica 'altro_{cat}' dopo assegnazione:")
            print(f"   altro_encoded value: {altro_encoded}")
            print(f"   altro_count: {altro_count}")
            print(f"   Valori unici in df_grouped[target_column]: {sorted(df_grouped[target_column].unique())}")
            # Verifica se ci sono ancora i vecchi valori
            for rare_class in rare_by_category[cat]:
                old_count = (df_grouped[target_column] == rare_class).sum()
                if old_count > 0:
                    print(f"   ⚠️  ATTENZIONE: Classe {rare_class} ha ancora {old_count} campioni (non è stata raggruppata!)")
            
            print(f"   Category {cat}: {len(rare_by_category[cat])} rare classes → 'altro_{cat}' ({altro_count} campioni totali)")
            grouped_count += 1
        elif len(rare_by_category[cat]) == 1:
            print(f"   Category {cat}: 1 rare class ({rare_by_category[cat][0]}) kept separate")
    
    print(f"   Total classes: {len(common_classes)} common + {grouped_count} 'altro' groups = {len(new_class_names)}")
    
    return df_grouped.reset_index(drop=True), label_encoders

def is_altro_category(categoria_encoded, label_encoder):
    """
    Check if a categoria (encoded) is one of the grouped rare categories.
    Returns True if categoria is "A", "B", or "C" (these are the grouped rare categories).
    """
    try:
        categoria_name = label_encoder.inverse_transform([int(categoria_encoded)])[0]
        return categoria_name in ['A', 'B', 'C']
    except:
        return False

def analyze_confusion_matrix(y_true, y_pred, label_encoder=None, save_path=None, plot_path=None):
    """
    Calcola e analizza la confusion matrix, stampando le classi più confuse.
    Salva la confusion matrix in CSV e crea un plot.
    Usa il label_encoder per convertire gli ID numerici nei nomi reali delle categorie.
    """
    
    # Get all possible labels from label encoder (to show all classes, even if not in test set)
    if label_encoder is not None:
        try:
            # Get all classes from label encoder
            all_possible_labels = np.arange(len(label_encoder.classes_))
            all_class_names = label_encoder.classes_
            
            # Get unique labels that actually appear in y_true or y_pred
            unique_labels_in_data = np.unique(np.concatenate((y_true, y_pred)))
            
            # Debug: check if all labels in data are valid
            invalid_labels = set(unique_labels_in_data) - set(all_possible_labels)
            if len(invalid_labels) > 0:
                print(f"⚠️  WARNING: Found invalid labels in data: {invalid_labels}")
                print(f"   Valid labels range: 0 to {len(all_possible_labels)-1}")
                # Filter out invalid labels
                valid_mask = np.isin(y_true, all_possible_labels) & np.isin(y_pred, all_possible_labels)
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]
                print(f"   Filtered data: {len(y_true)} samples remaining")
            
            # Use all possible labels to show complete confusion matrix
            unique_labels = all_possible_labels
            class_names = all_class_names
        except Exception as e:
            print(f"⚠️  Error getting all labels from encoder: {e}")
            # Fallback: use only labels that appear in data
            unique_labels = np.unique(np.concatenate((y_true, y_pred)))
            class_names = [str(label) for label in unique_labels]
    else:
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        class_names = [str(label) for label in unique_labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Convert to percentages (row-wise normalization)
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
    
    # Create DataFrame with percentages
    cm_df = pd.DataFrame(cm_percent, index=class_names, columns=class_names)
    
    print("\n" + "="*70)
    print("ANALISI CONFUSION MATRIX")
    print("="*70)
    
    print(f"\n📊 Classi totali: {len(class_names)}")
    print(f"📊 Campioni totali: {len(y_true)}")
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"📊 Accuratezza complessiva: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Analyze misclassifications
    misclassifications = []
    for i, true_class_name in enumerate(class_names):
        true_class_label = unique_labels[i]
        total_samples = np.sum(cm[i, :])
        correct_predictions = cm[i, i]
        errors = total_samples - correct_predictions
        
        if total_samples == 0:
            error_rate = 0.0
        else:
            error_rate = (errors / total_samples) * 100
        
        if errors > 0:
            # Get most common misclassifications for this true class
            misclassified_counts = {
                class_names[j]: cm[i, j]
                for j in range(len(class_names)) if i != j and cm[i, j] > 0
            }
            sorted_misclassified = sorted(misclassified_counts.items(), key=lambda item: item[1], reverse=True)
            
            misclassifications.append({
                'true_class': true_class_name,
                'total_samples': total_samples,
                'correct': correct_predictions,
                'errors': errors,
                'error_rate': error_rate,
                'most_common_misclassifications': sorted_misclassified
            })
    
    # Sort by error rate descending
    misclassifications = sorted(misclassifications, key=lambda x: x['error_rate'], reverse=True)
    
    # Analyze symmetric confusions
    symmetric_confusions = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            if cm[i, j] > 0 and cm[j, i] > 0:
                symmetric_confusions.append({
                    'class1': class_names[i],
                    'class2': class_names[j],
                    'count1_to_2': cm[i, j],
                    'count2_to_1': cm[j, i],
                    'total_confusions': cm[i, j] + cm[j, i]
                })
    
    symmetric_confusions = sorted(symmetric_confusions, key=lambda x: x['total_confusions'], reverse=True)
    
    if save_path:
        cm_df.to_csv(save_path)
        print(f"\n💾 Confusion matrix (percentuali) salvata in: {save_path}")
    
    # Plot confusion matrix
    if plot_path is None and save_path:
        plot_path = save_path.replace('.csv', '_plot.png')
    
    if plot_path:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_df, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentuale (%)'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Categoria Vera', fontsize=12, fontweight='bold')
        plt.xlabel('Categoria Predetta', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix plot saved to: {plot_path}")
        plt.close()
    
    return cm_df, misclassifications, symmetric_confusions

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution."""
    print("="*70)
    print("CATASTO TRAINING PIPELINE")
    print("="*70)

    # Ensure data directory exists
    ensure_data_dir()

    # Configuration
    fab_file = "IDR0000192470_TIPOFACSN_CAMML378.FAB"
    xml_folder = "preprocessed_1N_xml/preprocessed_xml"
    exclude_columns_categoria = [
        "source_file",
        "categoriaImmobiliare",
        "sottoCategoriaImmobiliare",
        "RENDITA_EURO",
        "CLASSE",
        "sottoclasse",
        "CONSISTENZA"
    ]
    
    exclude_columns_classe = [
        "source_file",
        "categoriaImmobiliare",
        "sottoCategoriaImmobiliare",
        "RENDITA_EURO",
        "sottoclasse",
        "CONSISTENZA"
        # NOTA: CATEGORIA NON è esclusa perché serve come feature per predire CLASSE
    ]

    # Step 1: Load or create FAB dataframe
    print("\n[1/7] Processing FAB file...")
    flat = load_or_create_flat(fab_file)
    print(f"✅ FAB records: {len(flat)}")

    # Step 2: Load or create XML dataframe
    print("\n[2/7] Processing XML files...")
    df_xml = load_or_create_xml(xml_folder)
    print(f"✅ XML records: {len(df_xml)}")

    # Step 3: Merge and clean
    print("\n[3/7] Merging and cleaning data...")
    df_final = df_xml.merge(
        flat,
        how='inner',
        left_on=['comuneCatastale', 'foglio', 'numeratore', 'subalterno'],
        right_on=['COMUNE_CATASTALE', 'FOGLIO', 'PARTICELLA', 'SUBALTERNO'],
        suffixes=('_xml', '_fab')
    )
    df_final_clean = clean_dataframe(df_final)
    print(f"✅ Cleaned records: {len(df_final_clean)}")

    # Step 4: Load or create preprocessed model data
    print("\n[4/7] Preprocessing data...")
    df_model, label_encoders = load_or_create_model_data(df_final_clean)
    print(f"✅ Preprocessed: {df_model.shape}")

    # Step 4.5: Group rare categories EARLY (before splitting)
    print("\n[4.5/7] Grouping rare categories (early grouping)...")
    MIN_SAMPLES_FOR_GROUPING = 5  # Parametro configurabile
    df_model, label_encoders = group_rare_categories_early(
        df_model, label_encoders, min_samples=MIN_SAMPLES_FOR_GROUPING, target_column="CATEGORIA"
    )
    
    # Show distribution after early grouping
    print(f"\n📊 Distribuzione categorie dopo raggruppamento iniziale:")
    counts_after = df_model["CATEGORIA"].value_counts().sort_index()
    le_cat_check = label_encoders["CATEGORIA"]
    for idx, count in zip(counts_after.index, counts_after.values):
        try:
            cat_name = le_cat_check.inverse_transform([int(idx)])[0]
            print(f"   {cat_name}: {count} campioni")
        except:
            print(f"   Classe {idx}: {count} campioni")

    # Step 5: Filter classes with <2 samples (needed for stratify) and split data
    print("\n[5/7] Filtering classes with <2 samples and splitting data...")
    
    # Filter CATEGORIA with <2 samples
    counts_before_split_cat = df_model["CATEGORIA"].value_counts()
    categorie_valid = counts_before_split_cat[counts_before_split_cat > 1].index
    df_filtered_cat = df_model[df_model["CATEGORIA"].isin(categorie_valid)].reset_index(drop=True)
    
    print(f"\n📊 Dopo filtro CATEGORIA (<2 campioni):")
    counts_final_cat = df_filtered_cat["CATEGORIA"].value_counts().sort_index()
    le_cat_check = label_encoders["CATEGORIA"]
    for idx, count in zip(counts_final_cat.index, counts_final_cat.values):
        try:
            cat_name = le_cat_check.inverse_transform([int(idx)])[0]
            print(f"   {cat_name}: {count} campioni")
        except:
            print(f"   Classe {idx}: {count} campioni")
    
    # Filter CLASSE with <2 samples (to avoid classes that end up only in test)
    print(f"\n📊 Filtering CLASSE with <2 samples...")
    counts_before_split_classe = df_filtered_cat["CLASSE"].value_counts()
    classi_valid = counts_before_split_classe[counts_before_split_classe > 1].index
    df_filtered = df_filtered_cat[df_filtered_cat["CLASSE"].isin(classi_valid)].reset_index(drop=True)
    
    n_removed_classe = len(df_filtered_cat) - len(df_filtered)
    if n_removed_classe > 0:
        print(f"   ⚠️  Removed {n_removed_classe} samples with CLASSE having <2 samples")
        print(f"   Removed CLASSE classes: {sorted(set(counts_before_split_classe[counts_before_split_classe == 1].index))}")
    else:
        print(f"   ✅ No CLASSE classes with <2 samples found")
    
    df_train_final, df_test = train_test_split(
        df_filtered,
        test_size=0.2,
        random_state=42,
        stratify=df_filtered["CATEGORIA"]
    )

    print(f"✅ Train: {len(df_train_final)}, Test: {len(df_test)}")

    # Prepare X, y for CATEGORIA prediction
    X_train_cat, y_train_cat = prepare_xy(df_train_final, "CATEGORIA", exclude_columns_categoria)
    X_test_cat, y_test_cat = prepare_xy(df_test, "CATEGORIA", exclude_columns_categoria)

    print(f"\n📊 Dataset sizes (CATEGORIA):")
    print(f"   Train: {len(X_train_cat)} samples")
    print(f"   Test: {len(X_test_cat)} samples")

    # ========================================================================
    # MODEL 1: Train CATEGORIA prediction (with augmentation)
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL 1: Training CATEGORIA Prediction (with augmentation)")
    print("="*70)
    
    # Option 1: Simple augmentation (like colleague's approach)
    print("\n[6/7] Applying Simple Augmentation for CATEGORIA...")
    
    # Prepare dataframe for simple augmentation
    # For CATEGORIA-only augmentation, we need to create a dummy CLASSE column
    # since apply_simple_augmentation expects both CATEGORIA and CLASSE
    df_train_final_cat_aug = df_train_final.copy()
    df_train_final_cat_aug['CATEGORIA'] = y_train_cat.values  # Use encoded values
    
    # If CLASSE is not in the dataframe, add a dummy column (all same value)
    # This way the augmentation will balance only by CATEGORIA (since all have same CLASSE)
    if 'CLASSE' not in df_train_final_cat_aug.columns:
        df_train_final_cat_aug['CLASSE'] = 0  # Dummy value, all same
    
    # Apply simple augmentation (will balance CATEGORIA+CLASSE, but since CLASSE is constant, 
    # it effectively balances only CATEGORIA)
    df_train_final_cat_augmented = apply_simple_augmentation(
        df_train_final_cat_aug, 
        min_examples_threshold=50, 
        target_examples=100,
        random_state=42
    )
    
    # Prepare X, y from augmented dataframe
    X_train_cat_aug, y_train_cat_aug = prepare_xy(
        df_train_final_cat_augmented, "CATEGORIA", exclude_columns_categoria
    )
    
    # Remove 'is_augmented' column if present (it's not a feature, just a flag)
    if 'is_augmented' in X_train_cat_aug.columns:
        X_train_cat_aug = X_train_cat_aug.drop(columns=['is_augmented'])
    if 'is_augmented' in X_test_cat.columns:
        X_test_cat = X_test_cat.drop(columns=['is_augmented'])
    
    # Show augmentation statistics
    n_original_cat = (df_train_final_cat_augmented['is_augmented'] == 0).sum()
    n_augmented_cat = (df_train_final_cat_augmented['is_augmented'] == 1).sum()
    print(f"✅ Augmented CATEGORIA training set: {len(X_train_cat_aug)} samples")
    print(f"   - Original samples: {n_original_cat}")
    print(f"   - Augmented samples: {n_augmented_cat} ({n_augmented_cat/len(X_train_cat_aug)*100:.1f}%)")
    
    # Train model with augmentation
    model_categoria, _ = train_model(
        X_train_cat_aug, y_train_cat_aug, 
        model_name="Model CATEGORIA"
    )

    # ========================================================================
    # MODEL 2: Train CLASSE prediction (using CATEGORIA as feature)
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL 2: Training CLASSE Prediction (using CATEGORIA as feature)")
    print("="*70)
    
    # Prepare data for CLASSE prediction
    # We need to add CATEGORIA (encoded) as a feature
    le_categoria = label_encoders["CATEGORIA"]
    
    # For training: use actual CATEGORIA from dataframe
    # For test: we'll use predicted CATEGORIA in the final pipeline
    # But for training the CLASSE model, we use the true CATEGORIA
    
    # Add encoded CATEGORIA as feature
    df_train_final_classe = df_train_final.copy()
    df_test_classe = df_test.copy()
    
    # Add CATEGORIA_encoded as feature (CATEGORIA is already encoded in df_model)
    # We need to use the encoded value directly
    if 'CATEGORIA' in df_train_final_classe.columns:
        # CATEGORIA is already encoded from preprocessing, use it directly
        df_train_final_classe['CATEGORIA_encoded'] = df_train_final_classe['CATEGORIA']
        df_test_classe['CATEGORIA_encoded'] = df_test_classe['CATEGORIA']
    else:
        # Fallback: encode if not already encoded
        df_train_final_classe['CATEGORIA_encoded'] = le_categoria.transform(df_train_final_classe['CATEGORIA'])
        df_test_classe['CATEGORIA_encoded'] = le_categoria.transform(df_test_classe['CATEGORIA'])
    
    # IMPORTANT: Always recreate CLASSE label encoder after split to ensure it matches
    # the actual classes in train/test sets (after grouping). The encoder from preprocessing
    # might have been created before grouping and may not match current classes.
    # IMPORTANT: Use only classes from TRAIN set, because the model is trained only on train.
    # Classes that appear only in test will not be known by the model.
    print("🔧 Recreating CLASSE label encoder to match current data (after grouping)...")
    
    # Get unique classes from TRAIN only (model is trained only on train)
    train_unique_classes = sorted(set(df_train_final_classe["CLASSE"].unique()))
    
    # Create new label encoder using only train classes
    le_classe = LabelEncoder()
    le_classe.fit([str(c) for c in train_unique_classes])
    
    # Transform CLASSE columns to new encoding
    df_train_final_classe["CLASSE"] = le_classe.transform([str(c) for c in df_train_final_classe["CLASSE"]])
    
    # For test set, only transform classes that exist in train (others will be mapped to -1 or filtered)
    # First, check which test classes are in train
    test_classes = df_test_classe["CLASSE"].unique()
    test_classes_in_train = [c for c in test_classes if str(c) in le_classe.classes_]
    test_classes_not_in_train = [c for c in test_classes if str(c) not in le_classe.classes_]
    
    if len(test_classes_not_in_train) > 0:
        print(f"   ⚠️  Warning: {len(test_classes_not_in_train)} test classes not in train: {test_classes_not_in_train}")
        print(f"      These will be filtered out during evaluation")
    
    # Transform test classes (only those in train)
    df_test_classe["CLASSE"] = df_test_classe["CLASSE"].apply(
        lambda x: le_classe.transform([str(x)])[0] if str(x) in le_classe.classes_ else -1
    )
    
    # Update label encoder
    label_encoders["CLASSE"] = le_classe
    
    print(f"   ✅ CLASSE label encoder recreated with {len(le_classe.classes_)} classes (from train only)")
    print(f"   Classes: {list(le_classe.classes_)}")
    
    # Prepare X, y for CLASSE prediction (CATEGORIA_encoded is included as feature)
    X_train_classe, y_train_classe = prepare_xy(df_train_final_classe, "CLASSE", exclude_columns_classe)
    X_test_classe, y_test_classe = prepare_xy(df_test_classe, "CLASSE", exclude_columns_classe)
    
    print(f"\n📊 Dataset sizes (CLASSE):")
    print(f"   Train: {len(X_train_classe)} samples")
    print(f"   Test: {len(X_test_classe)} samples")
    print(f"   ✅ Features include CATEGORIA_encoded")
    
    # Apply augmentation for CLASSE (balancing CATEGORIA+CLASSE combinations)
    print("\n[6/7] Applying Simple Augmentation for CLASSE (balancing CATEGORIA+CLASSE combinations)...")
    
    # Prepare dataframe for simple augmentation (with both CATEGORIA and CLASSE)
    df_train_final_classe_aug = df_train_final_classe.copy()
    df_train_final_classe_aug['CATEGORIA'] = df_train_final_classe['CATEGORIA'].values
    df_train_final_classe_aug['CLASSE'] = df_train_final_classe['CLASSE'].values
    
    # Apply simple augmentation (balances CATEGORIA+CLASSE combinations)
    df_train_final_classe_augmented = apply_simple_augmentation(
        df_train_final_classe_aug,
        min_examples_threshold=50,
        target_examples=100,
        random_state=42
    )
    
    # Prepare X, y from augmented dataframe
    X_train_classe_aug, y_train_classe_aug = prepare_xy(
        df_train_final_classe_augmented, "CLASSE", exclude_columns_classe
    )
    
    # Remove 'is_augmented' column if present (it's not a feature, just a flag)
    if 'is_augmented' in X_train_classe_aug.columns:
        X_train_classe_aug = X_train_classe_aug.drop(columns=['is_augmented'])
    if 'is_augmented' in X_test_classe.columns:
        X_test_classe = X_test_classe.drop(columns=['is_augmented'])
    
    # Show augmentation statistics
    n_original_classe = (df_train_final_classe_augmented['is_augmented'] == 0).sum()
    n_augmented_classe = (df_train_final_classe_augmented['is_augmented'] == 1).sum()
    print(f"✅ Augmented CLASSE training set: {len(X_train_classe_aug)} samples")
    print(f"   - Original samples: {n_original_classe}")
    print(f"   - Augmented samples: {n_augmented_classe} ({n_augmented_classe/len(X_train_classe_aug)*100:.1f}%)")
    
    # Train CLASSE model with augmentation using best parameters
    print("\n🔍 Training CLASSE model using best parameters...")
    print("   Best parameters: n_estimators=200, max_depth=60, min_samples_split=10, min_samples_leaf=1, max_features='log2'")
    
    # Calculate class weights
    classes_aug = np.unique(y_train_classe_aug)
    class_weights_balanced_aug = compute_class_weight('balanced', classes=classes_aug, y=y_train_classe_aug)
    class_weight_dict_aug = {int(cls): float(w) for cls, w in zip(classes_aug, class_weights_balanced_aug)}
    
    class_counts_train_aug = pd.Series(y_train_classe_aug).value_counts()
    rare_classes_aug = class_counts_train_aug[class_counts_train_aug < 100].index.tolist()
    very_rare_classes_aug = class_counts_train_aug[class_counts_train_aug < 20].index.tolist()
    
    for cls in rare_classes_aug:
        if cls in class_weight_dict_aug:
            original_weight = class_weight_dict_aug[cls]
            if cls in very_rare_classes_aug:
                class_weight_dict_aug[cls] = original_weight * 20.0
            else:
                class_weight_dict_aug[cls] = original_weight * 10.0
    
    # Remove is_augmented column if present
    X_train_classe_aug_clean = X_train_classe_aug.copy()
    if 'is_augmented' in X_train_classe_aug_clean.columns:
        X_train_classe_aug_clean = X_train_classe_aug_clean.drop(columns=['is_augmented'])
    
    # Train with best parameters for CLASSE
    model_classe = RandomForestClassifier(
        n_estimators=200,
        max_depth=60,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='log2',
        class_weight=class_weight_dict_aug,
        n_jobs=-1,
        random_state=42
    )
    
    model_classe.fit(X_train_classe_aug_clean, y_train_classe_aug)
    print(f"✅ Model trained successfully")
    
    # ========================================================================
    # TEST CLASSE MODEL: Using TRUE CATEGORIA (not predicted)
    # ========================================================================
    print("\n" + "="*70)
    print("CLASSE MODEL: Test with TRUE CATEGORIA (not cascade)")
    print("="*70)
    
    # Test CLASSE model using TRUE CATEGORIA (not predicted)
    y_test_pred_classe_true_cat = model_classe.predict(X_test_classe)
    test_report_classe_true_cat = classification_report(
        y_test_classe, y_test_pred_classe_true_cat, 
        target_names=label_encoders["CLASSE"].classes_,
        zero_division=0, output_dict=True
    )
    test_accuracy_classe_true_cat = accuracy_score(y_test_classe, y_test_pred_classe_true_cat)
    
    print(f"✅ CLASSE prediction accuracy (using TRUE CATEGORIA): {test_accuracy_classe_true_cat:.4f} ({test_accuracy_classe_true_cat*100:.2f}%)")
    print(f"   F1-macro: {test_report_classe_true_cat['macro avg']['f1-score']:.4f}")
    print(f"\n=== CLASSE - TEST RESULTS (True CATEGORIA) ===")
    print(classification_report(
        y_test_classe, y_test_pred_classe_true_cat, 
        target_names=label_encoders["CLASSE"].classes_,
        zero_division=0
    ))
    
    # Save confusion matrix for CLASSE model (with TRUE CATEGORIA)
    cm_path_classe = os.path.join(DATA_DIR, "confusion_matrix_classe.csv")
    cm_plot_path_classe = os.path.join(DATA_DIR, "confusion_matrix_classe_plot.png")
    cm_df_classe, _, _ = analyze_confusion_matrix(
        y_test_classe, y_test_pred_classe_true_cat, 
        label_encoder=label_encoders["CLASSE"],
        save_path=cm_path_classe,
        plot_path=cm_plot_path_classe
    )


    # ========================================================================
    # CASCADE PIPELINE: Test complete pipeline on test set
    # ========================================================================
    print("\n" + "="*70)
    print("CASCADE PIPELINE: Complete Test (CATEGORIA → CLASSE)")
    print("="*70)
    
    # Step 1: Predict CATEGORIA on test set
    print("\n[Step 1/2] Predicting CATEGORIA on test set...")
    X_test_cat_clean = X_test_cat.copy()
    if 'is_augmented' in X_test_cat_clean.columns:
        X_test_cat_clean = X_test_cat_clean.drop(columns=['is_augmented'])
    
    y_test_pred_categoria = model_categoria.predict(X_test_cat_clean)
    test_report_categoria = classification_report(
        y_test_cat, y_test_pred_categoria, 
        target_names=label_encoders["CATEGORIA"].classes_,
        zero_division=0, output_dict=True
    )
    test_accuracy_categoria = accuracy_score(y_test_cat, y_test_pred_categoria)
    
    print(f"✅ CATEGORIA prediction accuracy: {test_accuracy_categoria:.4f} ({test_accuracy_categoria*100:.2f}%)")
    
    # Check distribution of categories in test set (including altro_X)
    print(f"\n📊 Distribuzione categorie nel test set:")
    test_cat_counts = pd.Series(y_test_cat).value_counts().sort_index()
    test_cat_names = [le_categoria.inverse_transform([int(idx)])[0] for idx in test_cat_counts.index]
    for idx, count in zip(test_cat_counts.index, test_cat_counts.values):
        cat_name = le_categoria.inverse_transform([int(idx)])[0]
        print(f"   {cat_name}: {count} campioni")
    
    # Check predicted categories
    print(f"\n📊 Distribuzione categorie predette nel test set:")
    pred_cat_counts = pd.Series(y_test_pred_categoria).value_counts().sort_index()
    for idx, count in zip(pred_cat_counts.index, pred_cat_counts.values):
        cat_name = le_categoria.inverse_transform([int(idx)])[0]
        print(f"   {cat_name}: {count} campioni")
    
    # Step 2: Add predicted CATEGORIA as feature and predict CLASSE
    # BUT: Skip CLASSE prediction for "A", "B", "C" categories (grouped rare categories)
    print("\n[Step 2/2] Predicting CLASSE using predicted CATEGORIA...")
    print("   Note: CLASSE will NOT be predicted for 'A', 'B', 'C' categories (grouped rare categories)")
    
    # Identify which samples have grouped rare categories ("A", "B", "C")
    altro_mask = np.array([is_altro_category(cat, le_categoria) for cat in y_test_pred_categoria])
    n_altro = altro_mask.sum()
    n_predictable = len(y_test_pred_categoria) - n_altro
    
    print(f"   Samples with grouped rare categories ('A', 'B', 'C'): {n_altro} (will skip CLASSE prediction)")
    print(f"   Samples with regular categories: {n_predictable} (will predict CLASSE)")
    
    # Create test dataframe with predicted CATEGORIA
    df_test_cascade = df_test_classe.copy()
    # Replace CATEGORIA_encoded with predicted CATEGORIA (encoded)
    # Note: y_test_pred_categoria is already encoded (numeric)
    df_test_cascade['CATEGORIA_encoded'] = y_test_pred_categoria
    
    # Prepare X for CLASSE prediction (with predicted CATEGORIA)
    X_test_classe_cascade, _ = prepare_xy(df_test_cascade, "CLASSE", exclude_columns_classe)
    
    # Predict CLASSE only for non-grouped rare category samples
    # For grouped rare categories ("A", "B", "C"), we'll use a special value (-1) to indicate "not predicted"
    y_test_pred_classe = np.full(len(y_test_pred_categoria), -1, dtype=int)
    
    if n_predictable > 0:
        # Only predict for non-grouped rare category samples
        X_test_classe_predictable = X_test_classe_cascade[~altro_mask]
        y_test_pred_classe_predictable = model_classe.predict(X_test_classe_predictable)
        y_test_pred_classe[~altro_mask] = y_test_pred_classe_predictable
    
    # Evaluate CLASSE prediction only on samples that were actually predicted (non-grouped rare categories)
    if n_predictable > 0:
        y_test_classe_predictable = y_test_classe[~altro_mask]
        y_test_pred_classe_predictable = y_test_pred_classe[~altro_mask]
        
        test_report_classe = classification_report(
            y_test_classe_predictable, y_test_pred_classe_predictable, 
            target_names=label_encoders["CLASSE"].classes_,
            zero_division=0, output_dict=True
        )
        test_accuracy_classe = accuracy_score(y_test_classe_predictable, y_test_pred_classe_predictable)
        
        print(f"✅ CLASSE prediction accuracy (only on predictable samples): {test_accuracy_classe:.4f} ({test_accuracy_classe*100:.2f}%)")
        print(f"   Evaluated on {n_predictable} samples (excluded {n_altro} grouped rare category samples)")
        
        # Save confusion matrix for CLASSE model (cascade, using predicted CATEGORIA)
        cm_path_classe_cascade = os.path.join(DATA_DIR, "confusion_matrix_classe_cascade.csv")
        cm_plot_path_classe_cascade = os.path.join(DATA_DIR, "confusion_matrix_classe_cascade_plot.png")
        cm_df_classe_cascade, _, _ = analyze_confusion_matrix(
            y_test_classe_predictable, y_test_pred_classe_predictable, 
            label_encoder=label_encoders["CLASSE"],
            save_path=cm_path_classe_cascade,
            plot_path=cm_plot_path_classe_cascade
        )
    else:
        print(f"⚠️  No samples to evaluate CLASSE (all are grouped rare categories)")
        test_report_classe = None
        test_accuracy_classe = 0.0
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS: Cascade Pipeline")
    print("="*70)
    
    print(f"\n📊 Test Set Performance:")
    print(f"\n   CATEGORIA Model (separate):")
    print(f"      Accuracy: {test_accuracy_categoria:.4f} ({test_accuracy_categoria*100:.2f}%)")
    print(f"      F1-macro:  {test_report_categoria['macro avg']['f1-score']:.4f}")
    
    print(f"\n   CLASSE Model (using TRUE CATEGORIA, not cascade):")
    print(f"      Accuracy: {test_accuracy_classe_true_cat:.4f} ({test_accuracy_classe_true_cat*100:.2f}%)")
    print(f"      F1-macro:  {test_report_classe_true_cat['macro avg']['f1-score']:.4f}")
    
    print(f"\n   CLASSE Model (using predicted CATEGORIA, cascade):")
    if test_report_classe is not None:
        print(f"      Accuracy: {test_accuracy_classe:.4f} ({test_accuracy_classe*100:.2f}%) - evaluated on {n_predictable} samples")
        print(f"      F1-macro:  {test_report_classe['macro avg']['f1-score']:.4f}")
        print(f"      Excluded {n_altro} grouped rare category samples from CLASSE prediction")
    else:
        print(f"      ⚠️  No samples to evaluate (all are grouped rare categories)")
    
    print(f"\n=== CATEGORIA - TEST RESULTS ===")
    print(classification_report(
        y_test_cat, y_test_pred_categoria, 
        target_names=label_encoders["CATEGORIA"].classes_,
        zero_division=0
    ))
    
    print(f"\n=== CLASSE - TEST RESULTS (Cascade, using predicted CATEGORIA) ===")
    # Filter out -1 values (not predicted samples) before classification_report
    valid_mask = (y_test_classe >= 0) & (y_test_pred_classe >= 0)
    if valid_mask.sum() > 0:
        y_test_classe_valid = y_test_classe[valid_mask]
        y_test_pred_classe_valid = y_test_pred_classe[valid_mask]
        # Get unique classes present in valid data to show only relevant classes
        unique_classes = np.unique(np.concatenate([y_test_classe_valid, y_test_pred_classe_valid]))
        unique_classes = unique_classes[unique_classes >= 0]  # Remove -1 if present
        target_names_valid = [label_encoders["CLASSE"].classes_[i] for i in unique_classes]
        print(classification_report(
            y_test_classe_valid, y_test_pred_classe_valid, 
            target_names=target_names_valid,
            labels=unique_classes,
            zero_division=0
        ))
    else:
        print("⚠️  No valid samples to evaluate (all are -1)")
    
    
    cm_path_cat = os.path.join(DATA_DIR, "confusion_matrix_categoria.csv")
    cm_plot_path_cat = os.path.join(DATA_DIR, "confusion_matrix_categoria_plot.png")
    cm_df_cat, _, _ = analyze_confusion_matrix(
        y_test_cat, y_test_pred_categoria, 
        label_encoder=label_encoders["CATEGORIA"],
        save_path=cm_path_cat,
        plot_path=cm_plot_path_cat
    )

    print("\n✅ Cascade pipeline completed successfully!")
    
    # Save models and label encoders
    print("\n💾 Saving models and label encoders...")
    model_categoria_path = os.path.join(DATA_DIR, "model_categoria.pkl")
    model_classe_path = os.path.join(DATA_DIR, "model_classe.pkl")
    label_encoders_path = os.path.join(DATA_DIR, "label_encoders_final.pkl")
    
    with open(model_categoria_path, 'wb') as f:
        pickle.dump(model_categoria, f)
    print(f"   ✅ Model CATEGORIA saved to: {model_categoria_path}")
    
    with open(model_classe_path, 'wb') as f:
        pickle.dump(model_classe, f)
    print(f"   ✅ Model CLASSE saved to: {model_classe_path}")
    
    with open(label_encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"   ✅ Label encoders saved to: {label_encoders_path}")
    
    return model_categoria, model_classe, label_encoders

if __name__ == "__main__":
    model = main()

