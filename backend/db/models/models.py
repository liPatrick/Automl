from sqlalchemy import Column, String, PickleType, TIMESTAMP
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

    def __init__(self, id, date, features, output, modelType, modelBin):
        self.id = id
        self.date = date
        self.features = features
        self.output = output
        self.modelType = modelType
        self.modelBin = modelBin
