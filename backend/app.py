# We need this line to make sure os.environ gets set.
from db.models.models import MLModels
import config.config as constants
import pandas as pd
from io import StringIO
from flask import Flask
from db import ml_db
from db.models.models import MLModels
import uuid
from proxy.ml_engine.client import train_model, predict_model
from proxy.ml_engine.client import ModelClassification
import pickle
from flask import request, jsonify

FLASK_APP_PORT = "5001"
FLASK_APP_HOST = "0.0.0.0"

app = Flask(__name__)
app.config.update(
    SQLALCHEMY_DATABASE_URI=constants.SQLALCHEMY_DATABASE_URI,
    SQLALCHEMY_TRACK_MODIFICATIONS=constants.SQLALCHEMY_TRACK_MODIFICATIONS,
    SECRET_KEY=constants.FLASK_SECRET_KEY
)

ml_db.flask_sqlalchemy_db.init_app(app)
with app.app_context():
    ml_db.flask_sqlalchemy_db.create_all()

@app.get("/")
def main():
    return "Hello World"

@app.route("/train", methods=['POST'])
def train(): 
    json = request.get_json()
    dataset = json["dataset"]
    features = json["features"]
    output = json["output"]
    date = json["date"]
    datasetIO = StringIO(dataset)
    df = pd.read_csv(datasetIO, sep=",")

    #format the inputs 
    fDataset = df
    fFeatures = features  
    fOutputs = output
    fDate = date
    uid = str(uuid.uuid4())

    #prediction step 
    m = train_model(fDataset, fOutputs, fFeatures)

    #save model in db 
    newModel = MLModels(uid, fDate, fFeatures, fOutputs, m[0], m[1])
    ml_db.app_session.add(newModel)
    ml_db.app_session.commit()
    result = ml_db.app_session.query(MLModels).filter(MLModels.id == uid).first()
    print(result.id)
    print(result.date)
    print(result.features)
    print(result.output)
    print(result.modelType)
    print(result.modelBin)

    model_output: ModelClassification = ModelClassification(
        model_id=uid,
        date=fDate,
        features=fFeatures,
        output=fOutputs,
        modelType=m[0],
    )

    model_dict = model_output.asdict()
    return jsonify(model_dict)

@app.route("/predict", methods=['POST'])
def predict(): 
    json = request.get_json()
    model_id = json["model_id"]
    featureValues = json["featureValues"]
    print(model_id)
    print(featureValues)
    for i, feat in enumerate(featureValues): 
        feat = int(feat)
        featureValues[i] = feat 
    result = ml_db.app_session.query(MLModels).filter(MLModels.id == model_id).first()
    loadedModel = pickle.loads(result.modelBin)
    prediction = predict_model(loadedModel, featureValues)
    return jsonify(prediction)

@app.route("/models", methods=['GET'])
def models(): 
    result = ml_db.app_session.query(MLModels).order_by(MLModels.date).all()    
    returnVal = []

    
    for val in result: 
        fid = val.id 
        fdate = val.date 
        print(type(fdate))
        fdate = fdate.strftime("%m/%d/%Y, %-I:%M:%S %p")

        ffeatures = val.features[1:-1].split(",")
        foutput = val.output
        fmodelType = val.modelType
        model_output: ModelClassification = ModelClassification(
            model_id=fid,
            date=fdate,
            features=ffeatures,
            output=foutput,
            modelType=fmodelType,
        )
        model_dict = model_output.asdict()
        returnVal.append(model_dict)

    return jsonify(list(reversed(returnVal)))

@app.teardown_appcontext
def shutdown_session(_exception=None):
    ml_db.app_session.remove()    

#app.register_blueprint(automl_blueprint, url_prefix='/b/automl')
app.run(host=FLASK_APP_HOST, port=FLASK_APP_PORT)