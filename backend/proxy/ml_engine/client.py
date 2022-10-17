import random 
from typing import List 
from dataclasses import dataclass 
import pickle
import pandas as pd 
from sklearn import ensemble, linear_model, neural_network 

MODEL_TYPES = [
    linear_model.LogisticRegression,
    neural_network.MLPClassifier,
    ensemble.RandomForestClassifier,
]
@dataclass
class ModelClassification:
    model_id: str
    date: str 
    features: List[str]
    output: str 
    modelType: str
    
    def asdict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

def train_model(dataset: pd.DataFrame, output: str, inputs: List[str]) -> float:
    X = dataset[inputs]
    y = dataset[output]

    model = random.choice(MODEL_TYPES)()
    modelStr = str(model)
    model.fit(X, y)
    data = pickle.dumps(model)

    return (modelStr, data)


def predict_model(model, hInput):
    tups = model.predict_proba([hInput])
    ans = []
    for tup in tups:
        if tup[0] > tup[1]:
            ans = [True, tup[0]]
        else:
            ans = [False, tup[1]]
    return ans
    