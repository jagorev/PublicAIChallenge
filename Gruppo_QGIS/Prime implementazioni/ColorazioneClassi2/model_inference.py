"""
Modulo per l'inferenza con il modello di classificazione immobiliare.

Il modello è un RandomForest addestrato sulle caratteristiche degli immobili
per predire la categoria catastale (A01, A02, B06, C02, ecc.)
"""

import os
import pickle
import pandas as pd
from typing import List, Dict, Optional


class ModelInference:
    """
    Classe per gestire l'inferenza con il modello di classificazione.
    
    Carica il modello pickle salvato e fornisce metodi per:
    - Predire la classe di un immobile
    - Ottenere le probabilità per ogni classe
    - Restituire le top N classi con maggiore confidence
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inizializza il modulo di inferenza.
        
        Args:
            model_path: Percorso al file model.pkl. Se None, usa il percorso di default.
        """
        self.model = None
        self.label_encoder = None
        self.classes = None
        
        if model_path is None:
            # Percorso di default: model.pkl nella stessa cartella del plugin
            plugin_dir = os.path.dirname(__file__)
            model_path = os.path.join(plugin_dir, "model.pkl")
        
        self.model_path = os.path.abspath(model_path)
        
    def load_model(self) -> bool:
        """
        Carica il modello dal file pickle.
        
        Returns:
            True se il modello è stato caricato con successo, False altrimenti.
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"File modello non trovato: {self.model_path}")
            
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
            
            self.model = saved["model"]
            self.label_encoder = saved["label_encoder"]
            self.classes = self.label_encoder.classes_
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Errore nel caricamento del modello: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> List[str]:
        """
        Predice la classe per i dati forniti.
        
        Args:
            data: DataFrame con le features dell'immobile
            
        Returns:
            Lista di classi predette (es: ["A02"])
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError(f"Impossibile caricare il modello: {getattr(self, 'last_error', 'errore sconosciuto')}")
        
        pred = self.model.predict(data)
        decoded = self.label_encoder.inverse_transform(pred)
        return list(decoded)
    
    def predict_proba(self, data: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Predice le probabilità per ogni classe.
        
        Args:
            data: DataFrame con le features dell'immobile
            
        Returns:
            Lista di dizionari con classe e probabilità per ogni sample
            Es: [{"A02": 0.85, "A03": 0.10, "B01": 0.05}]
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Impossibile caricare il modello")
        
        probas = self.model.predict_proba(data)
        
        results = []
        for proba in probas:
            class_probs = {}
            for i, prob in enumerate(proba):
                class_name = self.label_encoder.inverse_transform([i])[0]
                class_probs[class_name] = float(prob)
            results.append(class_probs)
        
        return results
    
    def get_top_predictions(self, data: pd.DataFrame, top_n: int = 2) -> List[List[Dict[str, any]]]:
        """
        Restituisce le top N predizioni ordinate per probabilità.
        
        Args:
            data: DataFrame con le features dell'immobile
            top_n: Numero di classi top da restituire (default: 2)
            
        Returns:
            Lista di liste di dizionari nel formato atteso dal plugin:
            [[{"categoria": "A02", "probabilita": 0.85}, {"categoria": "A03", "probabilita": 0.10}]]
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Impossibile caricare il modello")
        
        all_probas = self.predict_proba(data)
        
        results = []
        for class_probs in all_probas:
            # Ordina per probabilità decrescente
            sorted_probs = sorted(
                class_probs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Prendi solo le top N
            top_predictions = [
                {"categoria": cat, "probabilita": prob}
                for cat, prob in sorted_probs[:top_n]
            ]
            results.append(top_predictions)
        
        return results
    
    def get_single_prediction(self, data: pd.DataFrame, top_n: int = 2) -> List[Dict[str, any]]:
        """
        Restituisce le top N predizioni per un singolo immobile.
        Metodo di convenienza per quando si ha un solo record.
        
        Args:
            data: DataFrame con le features di un singolo immobile
            top_n: Numero di classi top da restituire (default: 2)
            
        Returns:
            Lista di dizionari nel formato atteso dal plugin:
            [{"categoria": "A02", "probabilita": 0.85}, {"categoria": "A03", "probabilita": 0.10}]
        """
        results = self.get_top_predictions(data, top_n)
        if results:
            return results[0]
        return []
    
    def get_available_classes(self) -> List[str]:
        """
        Restituisce la lista di tutte le classi disponibili nel modello.
        
        Returns:
            Lista delle classi (es: ["A01", "A02", "B01", ...])
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Impossibile caricare il modello")
        
        return list(self.classes)
    
    @staticmethod
    def load_sample_input(csv_path: str) -> pd.DataFrame:
        """
        Carica un file CSV con i dati dell'immobile.
        
        Args:
            csv_path: Percorso al file CSV
            
        Returns:
            DataFrame con i dati dell'immobile
        """
        return pd.read_csv(csv_path)
    
    @staticmethod
    def get_required_features() -> List[str]:
        """
        Restituisce la lista delle features richieste dal modello.
        Queste sono le colonne che devono essere presenti nel DataFrame di input.
        
        Returns:
            Lista dei nomi delle features
        """
        # Features basate sul sample_input.csv
        return [
            "lista_piani", "datiMetriciNettiManuali", "tipoRiferimento", "annoRiferimento",
            "spessoreMuri", "superficieLordaMq", "numeroPiano", "superficieMq",
            "giardinoSuperficieLordaMq", "accessoCarrabile", "altezzaMediaLocaliPrincipaliCm",
            "postoAutoScoperto", "altriAccessoriAltro", "denominatore",
            "intPorteInterneMetallo", "intPorteIngressoLegnoTamburato",
            "intPorteInterneLegnoTamburato", "num", "superficieUtileMq", "bagniNum",
            "bagniSuperficieUtileMq", "corridoiNum", "corridoiSuperficieUtileMq",
            "altezzaMediaUtileCm", "superficieMqVaniAventiAltezzaMediaMinore230Cm",
            "riscaldamento", "condizionamento", "videoCitofono", "ascensoreServizio",
            "acquaCalda", "citofonico", "camereParquet", "cucinaBagnoPiastrelleCeramica",
            "intPorteIngressoAltro", "intPorteInterneLegnoMassello", "estFinestreLegnoMassello",
            "estVetroCameraLegnoMassello", "altriAccessoriPiastrelleCeramica",
            "intPorteIngressoMetallo", "estFinestreAltro", "estFinestreMetallo",
            "superficieLordaComunicantiMq", "camerePiastrelleCeramica", "estVetroCameraAltro",
            "intPorteIngressoLegnoMassello", "camereAltro", "cucinaBagnoParquet",
            "altriAccessoriParquet", "superficieLordaNonComunicantiMq", "camereMarmo",
            "cucinaBagnoMarmo", "ascensoriNumero", "camereGomme", "pianiFuoriTerraNum",
            "pianiFuoriTerraMc", "pianiEntroTerraNum", "pianiEntroTerraMc",
            "estVetroCameraMetallo", "cucinaBagnoAltro", "cucinaBagnoPiastrelleScaglie",
            "altriAccessoriPiastrelleScaglie", "ascensoreUsoEsclusivo", "montacarichi",
            "altriAccessoriGomme", "estDoppioInfissoLegnoMassello", "intPorteInterneAltro",
            "altriAccessoriMarmo", "altriAccessoriMoquette", "camereMoquette",
            "estDoppioInfissoAltro", "cucinaBagnoGomme", "camerePiastrelleScaglie",
            "estDoppioInfissoMetallo", "altroSuperficieLordaMq", "cucinaBagnoMoquette",
            "ZONA", "SUPERFICIE", "VALIMIS", "PIANI"
        ]


# Funzione di convenienza per uso rapido
def run_inference(csv_path: str, top_n: int = 2) -> List[Dict[str, any]]:
    """
    Funzione di convenienza per eseguire l'inferenza da un file CSV.
    
    Args:
        csv_path: Percorso al file CSV con i dati dell'immobile
        top_n: Numero di classi top da restituire (default: 2)
        
    Returns:
        Lista di dizionari con categoria e probabilità
    """
    inference = ModelInference()
    data = pd.read_csv(csv_path)
    return inference.get_single_prediction(data, top_n)
