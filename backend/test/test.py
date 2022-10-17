import random
from typing import Dict, List
import codecs
import pandas as pd
from sklearn import ensemble, linear_model, neural_network
import pickle

MODEL_TYPES = [
    linear_model.LogisticRegression,
    neural_network.MLPClassifier,
    ensemble.RandomForestClassifier,
]

def train_model(dataset: pd.DataFrame, output: str, inputs: List[str]) -> float:


    X = dataset[inputs]
    y = dataset[output]

    model = random.choice(MODEL_TYPES)()


    model.fit(X, y)
    data = pickle.dumps(model)

    return data

def predict(model, hInput):
    tups = model.predict_proba([hInput])
    ans = []
    for tup in tups:
        if tup[0] > tup[1]:
            ans.append("True")
        else:
            ans.append("False")
    return ans
    


def train_model_and_make_prediction(
    dataset: pd.DataFrame,
    output: str,
    inputs: List[str],
    hypothetical_input: Dict[str, float],
) -> float:  # probability of output being `True` for hypothetical input
    assert dataset[output].dtype in (bool,)
    assert all(dataset[input].dtype in (float, int) for input in inputs)

    X = dataset[inputs]
    y = dataset[output]

    model = random.choice(MODEL_TYPES)()
    model.fit(X, y)
    data = pickle.dumps(model)

    return model.predict_proba(
        pd.DataFrame({input: [hypothetical_input[input]] for input in inputs})
    )[0, model.classes_.tolist().index(True)]

def main(): 
    df = pd.read_csv("loan_pool_1.csv")
    


    output = ["IsMod"]
    inputs = ["LTV", "FICO", "StateMedianIncome", "PayHistory"]
    hInput = [54, 759, 48328, 43]
    model = train_model(df, output, inputs)

    loadedModel = pickle.loads(model)


    prediction = predict(loadedModel, hInput)
    print("prediction: ", prediction)


if __name__ == "__main__":
    main()