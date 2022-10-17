import random
from typing import Dict, List
import codecs
import pandas as pd
from sklearn import ensemble, linear_model, neural_network
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split  

import pickle

MODEL_TYPES = [
    linear_model.LogisticRegression,
    neural_network.MLPClassifier,
    ensemble.RandomForestClassifier,
]

def train_model(dataset: pd.DataFrame, output: str, inputs: List[str]) -> float:
    X = dataset[inputs]
    y = dataset[output]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4, random_state=0) 
 
    lr_model = MODEL_TYPES[0]().fit(X_train, Y_train)
    nn_model = MODEL_TYPES[1]().fit(X_train, Y_train)
    ensemble_model = MODEL_TYPES[2]().fit(X_train, Y_train)

    lr_auc = accuracy_score(Y_test, lr_model.predict(X_test))
    nn_auc = accuracy_score(Y_test, nn_model.predict(X_test))
    ensemble_auc = accuracy_score(Y_test, ensemble_model.predict(X_test))

    max_acc = max(lr_auc, nn_auc, ensemble_auc)
    chosen_model = None
    if max_acc == lr_auc:
        pred = lr_model.predict(X_test)
        conf_mat = confusion_matrix(Y_test, pred)
        TN, FP, FN, TP = conf_mat.ravel()
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        return {"model_name": "Logistic Regression", "model_bin": pickle.dumps(lr_model),  "auc": lr_auc, "tpr": TPR, "tnr": TNR, "fpr": FPR, "fnr": FNR, "ppv": PPV, "npv": NPV, "fdr": FDR}
    elif max_acc == nn_auc:
        pred = nn_model.predict(X_test)
        conf_mat = confusion_matrix(Y_test, pred)
        TN, FP, FN, TP = conf_mat.ravel()
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        return {"model_name": "MLP Classifier", "model_bin": pickle.dumps(nn_model), "auc": nn_auc, "tpr": TPR, "tnr": TNR, "fpr": FPR, "fnr": FNR, "ppv": PPV, "npv": NPV, "fdr": FDR}
    else:
        pred = ensemble_model.predict(X_test)
        conf_mat = confusion_matrix(Y_test, pred)
        TN, FP, FN, TP = conf_mat.ravel()
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        return {"model_name": "Random Forest Classifier", "model_bin": pickle.dumps(ensemble_model), "auc": ensemble_auc, "tpr": TPR, "tnr": TNR, "fpr": FPR, "fnr": FNR, "ppv": PPV, "npv": NPV, "fdr": FDR}


    

def predict(model, hInput):
    tups = model.predict_proba(hInput)
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
    print(model)

    loadedModel = pickle.loads(model["model_bin"])


    prediction = predict(loadedModel, [hInput])
    print("prediction: ", prediction)


if __name__ == "__main__":
    main()