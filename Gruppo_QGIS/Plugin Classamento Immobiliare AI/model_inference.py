"""
Modulo per l'inferenza con i modelli di classificazione immobiliare.

Pipeline a cascata:
1. Modello CATEGORIA: predice la categoria catastale (A01, A02, B06, C02, ecc.)
2. Modello CLASSE: predice la classe usando CATEGORIA come feature aggiuntiva

Il modulo supporta anche la spiegabilità locale tramite SHAP (TreeExplainer).
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

# SHAP è opzionale - importato solo se disponibile
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelInference:
    """
    Classe per gestire l'inferenza con i modelli di classificazione a cascata.
    
    Carica due modelli pickle:
    - model_categoria.pkl: predice CATEGORIA
    - model_classe.pkl: predice CLASSE (usando CATEGORIA come feature)
    
    E i label encoders:
    - label_encoders_final.pkl: dizionario con encoders per CATEGORIA e CLASSE
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Inizializza il modulo di inferenza.
        
        Args:
            model_dir: Cartella contenente i file del modello.
                       Se None, usa la cartella 'data' nella stessa directory del plugin.
        """
        self.model_categoria = None
        self.model_classe = None
        self.label_encoders = None
        self.le_categoria = None
        self.le_classe = None
        self.df_model_reference = None
        
        if model_dir is None:
            plugin_dir = os.path.dirname(__file__)
            model_dir = os.path.join(plugin_dir, "data")
        
        self.model_dir = os.path.abspath(model_dir)
        self.model_categoria_path = os.path.join(self.model_dir, "model_categoria.pkl")
        self.model_classe_path = os.path.join(self.model_dir, "model_classe.pkl")
        self.label_encoders_path = os.path.join(self.model_dir, "label_encoders_final.pkl")
        self.df_model_reference_path = os.path.join(self.model_dir, "df_model.csv")
        
        self._explainer_categoria = None
        self._explainer_classe = None
        self.last_error = None
        
    def load_models(self) -> bool:
        """Carica tutti i modelli e gli encoders dai file pickle."""
        try:
            for path, name in [
                (self.model_categoria_path, "model_categoria.pkl"),
                (self.model_classe_path, "model_classe.pkl"),
                (self.label_encoders_path, "label_encoders_final.pkl"),
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File non trovato: {path}")
            
            with open(self.model_categoria_path, "rb") as f:
                self.model_categoria = pickle.load(f)
            
            with open(self.model_classe_path, "rb") as f:
                self.model_classe = pickle.load(f)
            
            with open(self.label_encoders_path, "rb") as f:
                self.label_encoders = pickle.load(f)
            
            self.le_categoria = self.label_encoders.get("CATEGORIA")
            self.le_classe = self.label_encoders.get("CLASSE")
            
            if self.le_categoria is None or self.le_classe is None:
                raise ValueError("Label encoders per CATEGORIA o CLASSE non trovati")
            
            if os.path.exists(self.df_model_reference_path):
                self.df_model_reference = pd.read_csv(self.df_model_reference_path, low_memory=False)
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            return False
    
    def _ensure_models_loaded(self):
        """Assicura che i modelli siano caricati."""
        if self.model_categoria is None or self.model_classe is None:
            if not self.load_models():
                raise RuntimeError(f"Impossibile caricare i modelli: {self.last_error}")
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocessa i dati di input per renderli compatibili con i modelli."""
        self._ensure_models_loaded()
        df_processed = data.copy()
        expected_features_cat = self.model_categoria.feature_names_in_
        
        for col in expected_features_cat:
            if col not in df_processed.columns:
                if self.df_model_reference is not None and col in self.df_model_reference.columns:
                    if pd.api.types.is_numeric_dtype(self.df_model_reference[col]):
                        df_processed[col] = self.df_model_reference[col].median()
                    else:
                        mode_val = self.df_model_reference[col].mode()
                        df_processed[col] = mode_val[0] if len(mode_val) > 0 else 0
                else:
                    df_processed[col] = 0
        
        for col in expected_features_cat:
            if self.df_model_reference is not None and col in self.df_model_reference.columns:
                if pd.api.types.is_numeric_dtype(self.df_model_reference[col]):
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    df_processed[col] = df_processed[col].fillna(self.df_model_reference[col].median())
        
        for col, le in self.label_encoders.items():
            if col in expected_features_cat and col not in ['CATEGORIA', 'CLASSE']:
                if col in df_processed.columns:
                    most_frequent = le.classes_[0]
                    df_processed[col] = df_processed[col].fillna(most_frequent).astype(str)
                    df_processed[col] = df_processed[col].apply(
                        lambda x: x if x in le.classes_ else most_frequent
                    )
                    df_processed[col] = le.transform(df_processed[col])
        
        return df_processed
    
    def predict_categoria(self, data: pd.DataFrame, preprocessed: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predice la CATEGORIA per i dati forniti."""
        self._ensure_models_loaded()
        if not preprocessed:
            data = self.preprocess_input(data)
        
        expected_features = self.model_categoria.feature_names_in_
        X = data[expected_features]
        
        pred_encoded = self.model_categoria.predict(X).astype(int)
        pred_decoded = self.le_categoria.inverse_transform(pred_encoded)
        proba = self.model_categoria.predict_proba(X)
        
        return pred_encoded, pred_decoded, proba
    
    def predict_classe(self, data: pd.DataFrame, categoria_encoded: np.ndarray, preprocessed: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predice la CLASSE per i dati forniti, usando CATEGORIA come feature."""
        self._ensure_models_loaded()
        if not preprocessed:
            data = self.preprocess_input(data)
        
        df_with_cat = data.copy()
        df_with_cat['CATEGORIA_encoded'] = categoria_encoded
        
        expected_features = self.model_classe.feature_names_in_
        for col in expected_features:
            if col not in df_with_cat.columns:
                if self.df_model_reference is not None and col in self.df_model_reference.columns:
                    if pd.api.types.is_numeric_dtype(self.df_model_reference[col]):
                        df_with_cat[col] = self.df_model_reference[col].median()
                    else:
                        mode_val = self.df_model_reference[col].mode()
                        df_with_cat[col] = mode_val[0] if len(mode_val) > 0 else 0
                else:
                    df_with_cat[col] = 0
        
        X = df_with_cat[expected_features]
        
        pred_encoded = self.model_classe.predict(X).astype(int)
        pred_decoded = self.le_classe.inverse_transform(pred_encoded)
        proba = self.model_classe.predict_proba(X)
        
        return pred_encoded, pred_decoded, proba
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Esegue la pipeline completa di predizione (CATEGORIA -> CLASSE)."""
        self._ensure_models_loaded()
        df_processed = self.preprocess_input(data)
        
        cat_encoded, cat_decoded, cat_proba = self.predict_categoria(df_processed, preprocessed=True)
        classe_encoded, classe_decoded, classe_proba = self.predict_classe(
            df_processed, cat_encoded, preprocessed=True
        )
        
        idx = 0
        cat_all_proba = {
            self.le_categoria.classes_[i]: float(cat_proba[idx][i])
            for i in range(len(self.le_categoria.classes_))
        }
        classe_all_proba = {
            self.le_classe.classes_[i]: float(classe_proba[idx][i])
            for i in range(len(self.le_classe.classes_))
        }
        
        return {
            'categoria': cat_decoded[idx],
            'categoria_encoded': int(cat_encoded[idx]),
            'categoria_proba': float(cat_proba[idx][cat_encoded[idx]]),
            'categoria_all_proba': cat_all_proba,
            'classe': str(classe_decoded[idx]),
            'classe_encoded': int(classe_encoded[idx]),
            'classe_proba': float(classe_proba[idx][classe_encoded[idx]]),
            'classe_all_proba': classe_all_proba,
            'final_prediction': f"{cat_decoded[idx]}/{classe_decoded[idx]}"
        }
    
    def get_top_predictions(self, data: pd.DataFrame, top_n: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """Restituisce le top N predizioni ordinate per probabilità."""
        result = self.predict(data)
        
        cat_sorted = sorted(
            result['categoria_all_proba'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        classe_sorted = sorted(
            result['classe_all_proba'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return {
            'categoria': [{'nome': nome, 'probabilita': prob} for nome, prob in cat_sorted],
            'classe': [{'nome': str(nome), 'probabilita': prob} for nome, prob in classe_sorted],
            'final_prediction': result['final_prediction']
        }
    
    def get_single_prediction(self, data: pd.DataFrame, top_n: int = 2) -> List[Dict[str, Any]]:
        """
        Metodo di compatibilità con il vecchio plugin.
        Restituisce le predizioni nel formato atteso dal dialog esistente.
        """
        top_preds = self.get_top_predictions(data, top_n)
        
        results = []
        for i in range(min(top_n, len(top_preds['categoria']))):
            cat_pred = top_preds['categoria'][i] if i < len(top_preds['categoria']) else {'nome': 'N/A', 'probabilita': 0}
            classe_pred = top_preds['classe'][i] if i < len(top_preds['classe']) else {'nome': 'N/A', 'probabilita': 0}
            
            results.append({
                "categoria": cat_pred['nome'],
                "classe": classe_pred['nome'],
                "probabilita_cat": cat_pred['probabilita'],
                "probabilita_classe": classe_pred['probabilita'],
                "probabilita": cat_pred['probabilita']
            })
        
        return results
    
    def compute_shap_explanation(self, data: pd.DataFrame, target: str = 'categoria') -> Optional[Dict[str, Any]]:
        """Calcola la spiegazione SHAP per la predizione."""
        if not SHAP_AVAILABLE:
            return None
        
        self._ensure_models_loaded()
        df_processed = self.preprocess_input(data)
        
        if target == 'categoria':
            model = self.model_categoria
            X = df_processed[model.feature_names_in_]
            pred_encoded, pred_decoded, _ = self.predict_categoria(df_processed, preprocessed=True)
            
            if self._explainer_categoria is None:
                self._explainer_categoria = shap.TreeExplainer(model)
            explainer = self._explainer_categoria
            
        elif target == 'classe':
            model = self.model_classe
            cat_encoded, _, _ = self.predict_categoria(df_processed, preprocessed=True)
            df_processed['CATEGORIA_encoded'] = cat_encoded
            
            X = df_processed[model.feature_names_in_]
            pred_encoded, pred_decoded, _ = self.predict_classe(df_processed, cat_encoded, preprocessed=True)
            
            if self._explainer_classe is None:
                self._explainer_classe = shap.TreeExplainer(model)
            explainer = self._explainer_classe
        else:
            raise ValueError(f"target deve essere 'categoria' o 'classe'")
        
        shap_values = explainer.shap_values(X)
        pred_idx = int(pred_encoded[0])
        
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values_pred = shap_values[0, :, pred_idx]
            base_value = explainer.expected_value[pred_idx]
        elif isinstance(shap_values, list):
            shap_values_pred = shap_values[pred_idx][0]
            base_value = explainer.expected_value[pred_idx]
        else:
            shap_values_pred = shap_values[0]
            base_value = explainer.expected_value if not hasattr(explainer.expected_value, '__len__') else explainer.expected_value[0]
        
        feature_names = list(X.columns)
        feature_values = X.iloc[0].values
        
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_pred,
            'feature_value': feature_values,
            'abs_shap': np.abs(shap_values_pred)
        })
        
        top_positive = shap_df[shap_df['shap_value'] > 0].nlargest(10, 'shap_value')
        top_negative = shap_df[shap_df['shap_value'] < 0].nsmallest(10, 'shap_value')
        
        return {
            'features': feature_names,
            'shap_values': shap_values_pred.tolist(),
            'feature_values': feature_values.tolist(),
            'base_value': float(base_value),
            'predicted_class': pred_decoded[0],
            'top_positive': top_positive.to_dict('records'),
            'top_negative': top_negative.to_dict('records'),
            'top_30': shap_df.nlargest(30, 'abs_shap').to_dict('records')
        }
    
    def get_available_categories(self) -> List[str]:
        """Restituisce la lista di tutte le categorie disponibili."""
        self._ensure_models_loaded()
        return list(self.le_categoria.classes_)
    
    def get_available_classes(self) -> List[str]:
        """Restituisce la lista di tutte le classi disponibili."""
        self._ensure_models_loaded()
        return [str(c) for c in self.le_classe.classes_]
    
    def get_expected_features_categoria(self) -> List[str]:
        """Restituisce le feature attese dal modello CATEGORIA."""
        self._ensure_models_loaded()
        return list(self.model_categoria.feature_names_in_)
    
    def get_expected_features_classe(self) -> List[str]:
        """Restituisce le feature attese dal modello CLASSE."""
        self._ensure_models_loaded()
        return list(self.model_classe.feature_names_in_)
    
    @staticmethod
    def load_sample_input(csv_path: str) -> pd.DataFrame:
        """Carica un file CSV con i dati dell'immobile."""
        return pd.read_csv(csv_path)
    
    def save_inference_results(self, data: pd.DataFrame, output_dir: str, top_n: int = 3) -> Dict[str, str]:
        """
        Salva i risultati dell'inferenza in file CSV per la generazione del report.
        
        Args:
            data: DataFrame con i dati dell'immobile
            output_dir: Cartella dove salvare i risultati
            top_n: Numero di predizioni top da salvare
            
        Returns:
            Dizionario con i path dei file salvati
        """
        import os
        
        # Crea la cartella se non esiste
        os.makedirs(output_dir, exist_ok=True)
        
        self._ensure_models_loaded()
        df_processed = self.preprocess_input(data)
        
        # Predizioni CATEGORIA
        cat_encoded, cat_decoded, cat_proba = self.predict_categoria(df_processed, preprocessed=True)
        
        # Predizioni CLASSE
        classe_encoded, classe_decoded, classe_proba = self.predict_classe(
            df_processed, cat_encoded, preprocessed=True
        )
        
        # Top N per CATEGORIA e CLASSE
        top_cat_indices = np.argsort(cat_proba[0])[-top_n:][::-1]
        top_classe_indices = np.argsort(classe_proba[0])[-top_n:][::-1]
        
        # Crea DataFrame predizioni
        pred_data = {
            'final_prediction': [f"{cat_decoded[0]}/{classe_decoded[0]}"],
        }
        
        for i in range(top_n):
            pred_data[f'CATEGORIA_top{i+1}'] = [self.le_categoria.classes_[top_cat_indices[i]]]
            pred_data[f'CATEGORIA_top{i+1}_conf'] = [float(cat_proba[0][top_cat_indices[i]])]
        
        for i in range(top_n):
            pred_data[f'CLASSE_top{i+1}'] = [str(self.le_classe.classes_[top_classe_indices[i]])]
            pred_data[f'CLASSE_top{i+1}_conf'] = [float(classe_proba[0][top_classe_indices[i]])]
        
        risultato = pd.DataFrame(pred_data)
        predictions_path = os.path.join(output_dir, "predictions_output.csv")
        risultato.to_csv(predictions_path, index=False)
        
        # Salva analisi SHAP se disponibile
        shap_paths = {}
        shap_explanation = self.compute_shap_explanation(data, target='categoria')
        
        if shap_explanation:
            # Top 30 features
            shap_top30 = pd.DataFrame(shap_explanation['top_30'])
            shap_top30_path = os.path.join(output_dir, "shap_top30_features.csv")
            shap_top30.to_csv(shap_top30_path, index=False)
            shap_paths['shap_top30'] = shap_top30_path
            
            # Top 10 positive
            shap_positive = pd.DataFrame(shap_explanation['top_positive'])
            shap_pos_path = os.path.join(output_dir, "shap_top10_positive.csv")
            shap_positive.to_csv(shap_pos_path, index=False)
            shap_paths['shap_positive'] = shap_pos_path
            
            # Top 10 negative
            shap_negative = pd.DataFrame(shap_explanation['top_negative'])
            shap_neg_path = os.path.join(output_dir, "shap_top10_negative.csv")
            shap_negative.to_csv(shap_neg_path, index=False)
            shap_paths['shap_negative'] = shap_neg_path
        
        return {
            'predictions': predictions_path,
            'output_dir': output_dir,
            **shap_paths
        }


def run_inference(csv_path: str, model_dir: Optional[str] = None, top_n: int = 2) -> List[Dict[str, Any]]:
    """Funzione di convenienza per eseguire l'inferenza da un file CSV."""
    inference = ModelInference(model_dir=model_dir)
    data = pd.read_csv(csv_path)
    return inference.get_single_prediction(data, top_n)

