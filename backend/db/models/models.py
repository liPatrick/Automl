from sqlalchemy import Column, String, PickleType, TIMESTAMP, Numeric
from sqlalchemy.types import Date


from db import ml_db

class MLModels(ml_db.declarative_base):
    __tablename__ = "mlmodels"

    id = Column(String, primary_key=True, index=True)
    date = Column(TIMESTAMP)
    features = Column(String)
    output = Column(String)
    modelType = Column(String)
    modelBin = Column(PickleType)
    auc = Column(Numeric)
    tpr = Column(Numeric)
    tnr = Column(Numeric)
    fpr = Column(Numeric)
    fnr = Column(Numeric)
    ppv = Column(Numeric)
    npv = Column(Numeric)
    fdr = Column(Numeric)
    def __init__(self, id, date, features, output, modelType, modelBin, auc, tpr, tnr, fpr, fnr, ppv, npv, fdr):
        self.id = id 
        self.date = date
        self.features = features
        self.output = output
        self.modelType = modelType
        self.modelBin = modelBin
        self.auc = auc
        self.tpr = tpr
        self.tnr = tnr
        self.fpr = fpr
        self.fnr = fnr
        self.ppv = ppv
        self.npv = npv
        self.fdr = fdr

        
