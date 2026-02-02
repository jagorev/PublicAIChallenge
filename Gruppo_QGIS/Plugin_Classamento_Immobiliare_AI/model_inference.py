"""
Modulo per l'inferenza con i modelli di classificazione immobiliare (Versione XML).
"""

import os, sys
import pickle
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple, Any

# Configurazione Joblib per evitare crash in QGIS
if sys.platform == "win32":
    os.environ["JOBLIB_START_METHOD"] = "spawn"
else:
    os.environ["JOBLIB_START_METHOD"] = "fork"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelInference:
    
    def __init__(self, model_dir: Optional[str] = None):
        self.model_categoria = None
        self.model_classe = None
        self.label_encoders = None
        self.le_categoria = None
        self.le_classe = None
        self.column_types = None
        self.numeric_medians = None
        self.filtered_column_types = None
        
        if model_dir is None:
            plugin_dir = os.path.dirname(__file__)
            model_dir = os.path.join(plugin_dir, "data")
        
        self.model_dir = os.path.abspath(model_dir)
        self.model_categoria_path = os.path.join(self.model_dir, "model_categoria.pkl")
        self.model_classe_path = os.path.join(self.model_dir, "model_classe.pkl")
        self.label_encoders_path = os.path.join(self.model_dir, "label_encoders_final.pkl")
        self.numeric_medians_path = os.path.join(self.model_dir, "numeric_medians.pkl")
        self.column_types_path = os.path.join(self.model_dir, "column_types.pkl")
        
        self._explainer_categoria = None
    
    def _disable_model_parallelism(self, model):
        """Disabilita il parallelismo interno di sklearn/joblib."""
        if hasattr(model, 'n_jobs'): model.n_jobs = 1
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                if hasattr(estimator, 'n_jobs'): estimator.n_jobs = 1
        
    def load_models(self) -> bool:
        """Carica tutti i modelli e i file di supporto."""
        try:
            with open(self.model_categoria_path, "rb") as f: 
                self.model_categoria = pickle.load(f)
            self._disable_model_parallelism(self.model_categoria)
            
            with open(self.model_classe_path, "rb") as f: 
                self.model_classe = pickle.load(f)
            self._disable_model_parallelism(self.model_classe)
            
            with open(self.label_encoders_path, "rb") as f: 
                self.label_encoders = pickle.load(f)
            self.le_categoria = self.label_encoders.get("CATEGORIA")
            self.le_classe = self.label_encoders.get("CLASSE")

            with open(self.numeric_medians_path, "rb") as f: 
                self.numeric_medians = pickle.load(f)
            
            with open(self.column_types_path, "rb") as f: 
                self.column_types = pickle.load(f)

            # Filtra i tipi di colonna in base alle feature richieste dal modello Categoria
            required_features = set(self.model_categoria.feature_names_in_)
            self.filtered_column_types = {
                col: typ for col, typ in self.column_types.items() 
                if col in required_features
            }
            return True
        except Exception as e:
            return False
    
    def _ensure_models_loaded(self):
        if self.model_categoria is None: 
            self.load_models()

    def parse_xml_input(self, xml_file: str) -> pd.DataFrame:
        """Parsing identico al codice di riferimento."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename = os.path.basename(xml_file)

            elenco_ui = root.find('.//ElencoUI')
            if elenco_ui is None: 
                return pd.DataFrame()

            all_rows = []
            ui_elements = elenco_ui.findall('.//UICostituzione')

            for ui in ui_elements:
                row = {"source_file": filename}
                
                # Identificativo Catastale PM
                idpm = ui.find('.//ElencoIdentificativiCatastaliPM/IdentificativoCatastalePM')
                if idpm is not None:
                    row.update(idpm.attrib)

                # Classamento
                classamento = ui.find('.//Classamento')
                if classamento is not None:
                    row.update(classamento.attrib)

                # Indirizzo
                indir = ui.find('.//ElencoIndirizzi/Indirizzo')
                if indir is not None:
                    row.update(indir.attrib)

                # Piani
                piani = ui.findall('.//ElencoPiani/Piano')
                row["lista_piani"] = ";".join([p.attrib.get("numeroPiano", "") for p in piani])

                # Mod1N-2 (iterazione ricorsiva come da riferimento)
                mod1n2 = ui.find('.//Mod1N-2')
                if mod1n2 is not None:
                    row.update(mod1n2.attrib)
                    for elem in mod1n2.iter():
                        if elem is not mod1n2:
                            row.update(elem.attrib)
                
                all_rows.append(row)
            
            df_xml = pd.DataFrame(all_rows)
            
            # Post-processing interi
            int_cols = ['comuneCatastale', 'foglio', 'numeratore', 'subalterno']
            for col in int_cols:
                if col in df_xml.columns:
                    try: 
                        df_xml[col] = df_xml[col].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
                    except: 
                        pass
            return df_xml

        except Exception as e:
            # Per debug in QGIS meglio non stampare su console ma ritornare vuoto o loggare
            return pd.DataFrame()

    def clean_dataframe(self, df_final: pd.DataFrame) -> pd.DataFrame:
        """Rimuove colonne non necessarie."""
        columns_to_drop = [
            'source_file', 'codiceVia', 'indirizzoIT', 'civico1', 'civico2',
            'civico3', 'foglio', 'numeratore', 'subalterno','comuneCatastale',
            'estAltroDescrizione', 'intAltroDescrizione', 'altroDescrizione',
            'pavimentazioneAltroDescrizione', 'categoriaImmobiliare',
            'sottoCategoriaImmobiliare'
        ]

        return df_final.drop(columns=columns_to_drop, errors='ignore')

    def prepare_row(self, df_xml: pd.DataFrame) -> pd.DataFrame:
        """Prepara una singola riga seguendo la logica del training."""
        self._ensure_models_loaded()
        if df_xml.empty: return pd.DataFrame()
        
        df_prepared = pd.DataFrame(index=[0])
        
        for col, col_type in self.filtered_column_types.items():
            if col in df_xml.columns:
                if col_type == "boolean":
                    df_prepared[col] = [1]
                elif col_type == "categorical":
                    le = self.label_encoders.get(col)
                    val = df_xml[col].iloc[0]
                    # Gestione valori mancanti o stringhe
                    val_str = "MISSING" if pd.isna(val) else str(val)
                    
                    if le and val_str in le.classes_:
                        df_prepared[col] = [le.transform([val_str])[0]]
                    else:
                        # Fallback a "MISSING" se esiste nel LE
                        try:
                            df_prepared[col] = [le.transform(["MISSING"])[0]]
                        except:
                            df_prepared[col] = [0]
                else: # Numeric
                    val = df_xml[col].iloc[0]
                    try:
                        df_prepared[col] = [float(val)]
                    except:
                        # Se fallback, usa la mediana
                        df_prepared[col] = [float(self.numeric_medians.get(col, 0))]
            else:
                # Colonna mancante nel DataFrame XML
                if col_type == "boolean":
                    df_prepared[col] = [0]
                elif col_type == "numeric":
                    df_prepared[col] = [self.numeric_medians.get(col, 0)]
                else: # Categorical missing
                    le = self.label_encoders.get(col)
                    if le is not None:
                        try:
                            df_prepared[col] = [le.transform(["MISSING"])[0]]
                        except:
                            df_prepared[col] = [0]
                    else:
                        df_prepared[col] = [0]
                        
        return df_prepared

    def _run_prediction_pipeline(self, df_p: pd.DataFrame) -> Dict[str, Any]:
        """Esegue predizione Categoria -> Classe."""
        # 1. Categoria
        X_cat = df_p[self.model_categoria.feature_names_in_]
        p_cat = self.model_categoria.predict(X_cat).astype(int)[0]
        prob_cat = self.model_categoria.predict_proba(X_cat)[0]
        cat_dec = self.le_categoria.inverse_transform([p_cat])[0]
        
        # 2. Classe
        df_cl = df_p.copy()
        df_cl['CATEGORIA'] = p_cat
        
        # Prepara X_classe assicurando tutte le feature
        X_cl = pd.DataFrame(index=[0])
        for col in self.model_classe.feature_names_in_:
            if col in df_cl.columns:
                X_cl[col] = df_cl[col]
            elif col in self.numeric_medians:
                X_cl[col] = self.numeric_medians[col]
            else:
                X_cl[col] = 0
            
        p_cl = self.model_classe.predict(X_cl).astype(int)[0]
        prob_cl = self.model_classe.predict_proba(X_cl)[0]
        cl_dec = self.le_classe.inverse_transform([p_cl])[0]

        return {
            'categoria': cat_dec,
            'categoria_encoded': int(p_cat),
            'categoria_proba': float(prob_cat[p_cat]),
            'categoria_all_proba': dict(zip(self.le_categoria.classes_, prob_cat)),
            'classe': str(cl_dec),
            'classe_encoded': int(p_cl),
            'classe_proba': float(prob_cl[p_cl]),
            'classe_all_proba': dict(zip(self.le_classe.classes_, prob_cl)),
            'final_prediction': f"{cat_dec}/{cl_dec}"
        }

    def run_full_analysis(self, xml_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Orchestratore principale chiamato dalla GUI."""
        self._ensure_models_loaded()
        
        # 1. Pipeline
        df_raw = self.parse_xml_input(xml_path)
        if df_raw.empty: 
            return {"error": "XML vuoto o non valido"}
        
        df_clean = self.clean_dataframe(df_raw)
        df_proc = self.prepare_row(df_clean)
        
        if df_proc.empty or df_proc.isnull().values.any():
              # Fallback di sicurezza se prepare_row ha problemi
             pass

        pred_res = self._run_prediction_pipeline(df_proc)
        
        # 2. SHAP
        shap_data = self.compute_shap_explanation(df_proc)
        if shap_data: 
            shap_data['predicted_class'] = pred_res['categoria']
        
        # 3. Formattazione GUI (Top N)
        sorted_cat = sorted(pred_res['categoria_all_proba'].items(), key=lambda x:x[1], reverse=True)
        sorted_cls = sorted(pred_res['classe_all_proba'].items(), key=lambda x:x[1], reverse=True)
        
        formatted_preds = []
        for i in range(min(5, len(sorted_cat))):
            formatted_preds.append({
                "categoria": sorted_cat[i][0],
                "probabilita_cat": sorted_cat[i][1],
                "classe": sorted_cls[i][0] if i < len(sorted_cls) else "N/A",
                "probabilita_classe": sorted_cls[i][1] if i < len(sorted_cls) else 0.0,
                "final_prediction": pred_res['final_prediction']
            })
            
        # 4. Salvataggio CSV (Bridge per report_generator)
        if output_dir:
            self.save_inference_results(pred_res, shap_data, output_dir, sorted_cat, sorted_cls)
            
        return {
            "predizioni": formatted_preds,
            "shap_data": shap_data
        }

    def save_inference_results(self, pred_dict, shap_data, output_dir, sorted_cat, sorted_cls):
        """Salva i CSV per report_generator."""
        os.makedirs(output_dir, exist_ok=True)
        
        row = {'final_prediction': pred_dict['final_prediction']}
        # Salva top 3 per coerenza col reference code
        for i in range(3):
            if i < len(sorted_cat):
                row[f'CATEGORIA_top{i+1}'] = sorted_cat[i][0]
                row[f'CATEGORIA_top{i+1}_conf'] = sorted_cat[i][1]
            if i < len(sorted_cls):
                row[f'CLASSE_top{i+1}'] = sorted_cls[i][0]
                row[f'CLASSE_top{i+1}_conf'] = sorted_cls[i][1]
                
        pd.DataFrame([row]).to_csv(os.path.join(output_dir, "predictions_output.csv"), index=False)
        
        if shap_data:
            pd.DataFrame(shap_data['top_positive']).to_csv(os.path.join(output_dir, "shap_top10_positive.csv"), index=False)
            pd.DataFrame(shap_data['top_negative']).to_csv(os.path.join(output_dir, "shap_top10_negative.csv"), index=False) 
            pd.DataFrame(shap_data['top_30']).to_csv(os.path.join(output_dir, "shap_top30_features.csv"), index=False)

    def compute_shap_explanation(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calcolo SHAP (Feature Importance locale)."""
        if not SHAP_AVAILABLE: return None
        model = self.model_categoria
        X = df_processed[model.feature_names_in_]
        
        if self._explainer_categoria is None: 
            self._explainer_categoria = shap.TreeExplainer(model)
        
        pred = model.predict(X).astype(int)[0]
        vals = self._explainer_categoria.shap_values(X)
        
        # Gestione differenze formato output SHAP (lista vs array)
        if isinstance(vals, list): 
            target_vals = vals[pred][0]
        elif isinstance(vals, np.ndarray) and vals.ndim == 3: 
            target_vals = vals[0, :, pred]
        else: 
            target_vals = vals[0]
            
        sdf = pd.DataFrame({
            'feature': X.columns, 
            'shap_value': target_vals, 
            'feature_value': X.iloc[0].values
        })
        
        return {
            'top_positive': sdf[sdf.shap_value > 0].nlargest(10, 'shap_value').to_dict('records'),
            'top_negative': sdf[sdf.shap_value < 0].nsmallest(10, 'shap_value').to_dict('records'),
            'top_30': sdf.assign(abs_v=sdf.shap_value.abs()).nlargest(30, 'abs_v').to_dict('records')
        }