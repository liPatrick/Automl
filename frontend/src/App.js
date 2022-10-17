import React, { useEffect, useState } from "react";
import utf8 from "utf8"

function App() {
  const [file, setFile] = useState();
  const [array, setArray] = useState([]);

  const [validFeatures, setValidFeatures] = useState([]);
  const [validOutputs, setValidOutputs] = useState([]);

  const [pastModels, setPastModels] = useState([]);
  const [currModel, setCurrModel] = useState({ "features": [], "output": "" });
  const [currPrediction, setCurrPrediction] = useState([]);

  useEffect(() => {
    fetch("/models", {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }).then((response) => response.json())
      .then((data) => {
        setPastModels(data)
      })


  }, [])


  const fileReader = new FileReader();

  const handleOnChange = (e) => {
    setFile(e.target.files[0]);
  };

  const getValidFeatures = (fNames, fVal) => {
    let map = { "outputs": [], "features": [] }
    for (let i = 0; i < fVal.length; i++) {

      try {
        let b = JSON.parse(fVal[i].toLowerCase())
        if (typeof b == "boolean") {
          map["outputs"].push(fNames[i])
        } else if (typeof b == "number") {
          map["features"].push(fNames[i])
        }

      } catch (exception) {
        console.log("not boolean")
      }



    }
    return map
  }

  const csvFileToArray = string => {
    const csvHeader = string.slice(0, string.indexOf("\n")).split(",");
    const csvRows = string.slice(string.indexOf("\n") + 1).split("\n");

    const firstRow = csvRows[0].split(",")
    let possibleFeatureAndOutputs = getValidFeatures(csvHeader, firstRow)


    let validFeatures = []
    for (let i = 0; i < possibleFeatureAndOutputs["features"].length; i++) {
      validFeatures.push({ "featureName": possibleFeatureAndOutputs["features"][i], "selected": false })
    }
    setValidFeatures(validFeatures)
    let validOutputs = []
    for (let i = 0; i < possibleFeatureAndOutputs["outputs"].length; i++) {
      validOutputs.push({ "outputName": possibleFeatureAndOutputs["outputs"][i], "selected": false })
    }
    setValidOutputs(validOutputs)
    const array = csvRows.map(i => {
      const values = i.split(",");
      const obj = csvHeader.reduce((object, header, index) => {
        object[header] = values[index];
        return object;
      }, {});
      return obj;
    });

    setArray(array);
  };

  const handleOnSubmit = (e) => {
    e.preventDefault();

    if (file) {
      fileReader.onload = function (event) {
        const text = event.target.result;
        csvFileToArray(text);
      };

      fileReader.readAsText(file);
    }
  };

  const handleOnFeatureCheck = (e) => {

    let newValidFeatures = structuredClone(validFeatures)

    for (let i = 0; i < newValidFeatures.length; i++) {
      let key = newValidFeatures[i]["featureName"]
      let val = newValidFeatures[i]["selected"]
      if (key === e.target.id) {
        newValidFeatures[i]["selected"] = !val
      }
    }
    setValidFeatures(newValidFeatures)
  }

  const handleOnOutputCheck = (e) => {
    let newValidOutputs = structuredClone(validOutputs)

    for (let i = 0; i < newValidOutputs.length; i++) {
      let key = newValidOutputs[i]["outputName"]
      if (key === e.target.id) {
        newValidOutputs[i]["selected"] = true
      } else {
        newValidOutputs[i]["selected"] = false
      }
    }
    setValidOutputs(newValidOutputs)

  }


  const handleOnTrain = (e) => {
    let features = []
    let output = []
    for (let i = 0; i < validFeatures.length; i++) {
      if (validFeatures[i]["selected"]) {
        features.push(validFeatures[i]["featureName"])
      }
    }
    for (let i = 0; i < validOutputs.length; i++) {
      if (validOutputs[i]["selected"]) {
        output.push(validOutputs[i]["outputName"])
      }
    }

    if (file && features.length > 0 && output.length > 0) {
      fileReader.onload = function (event) {
        const text = event.target.result;
        let encoded = utf8.encode(text)
        fetch("/train", {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            "dataset": encoded,
            "features": features,
            "output": output,
            "date": new Date().toLocaleString(),

          })
        }).then((response) => response.json())
          .then((data) => {
            /*
            let model_id = data["model_id"]
            let date = data["date"]
            let features = data["features"]
            let output = data["output"]
            let modelType = data["modelType"]
            let modelLiteral = { model_id: model_id, date: date, features: features, output: output, modelType: modelType }
            let pastModelsCopy = structuredClone(pastModels)
            pastModelsCopy.unshift(modelLiteral)
            setPastModels(pastModelsCopy)*/
            fetch("/models", {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
              },
            }).then((response) => response.json())
              .then((data) => {
                setPastModels(data)
              })

          })
      }
      fileReader.readAsText(file);
    }

  }

  const handleOnSelectModel = (e) => {
    for (let i = 0; i < pastModels.length; i++) {
      if (pastModels[i]["model_id"] === e.target.id) {
        let featureAndWeights = []
        for (let j = 0; j < pastModels[i]["features"].length; j++) {
          featureAndWeights.push({ feature: pastModels[i]["features"][j], weight: 0 })
        }
        let myModel = { model_id: e.target.id, features: featureAndWeights, output: pastModels[i]["output"], auc: pastModels[i]["auc"], tfr: pastModels[i]["tfr"], tnr: pastModels[i]["tnr"], tpr: pastModels[i]["tpr"], fnr: pastModels[i]["fnr"], ppv: pastModels[i]["ppv"], npv: pastModels[i]["npv"], fdr: pastModels[i]["fdr"] }
        console.log(myModel)
        setCurrModel(myModel)
      }
    }
  }

  const handleOnChangeParams = (e) => {
    let newCurrModel = structuredClone(currModel)
    for (let i = 0; i < newCurrModel["features"].length; i++) {
      if (newCurrModel["features"][i]["feature"] === e.target.id) {
        newCurrModel["features"][i]["weight"] = e.target.value
      }
    }
    setCurrModel(newCurrModel)
  }

  const handleOnPredict = (e) => {
    let featureValues = []
    for (let i = 0; i < currModel["features"].length; i++) {
      featureValues.push(currModel["features"][i]["weight"])
    }
    console.log(featureValues)
    fetch("/predict", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        "model_id": currModel.model_id,
        "featureValues": featureValues,
      })
    }).then((response) => response.json())
      .then((data) => {
        setCurrPrediction(data)

      })
  }

  function Prediction(props) {
    if (props.predVal.length !== 0) {
      return (
        <div>
          <p>{currModel.output + " result: " + String(props.predVal[0])}</p>
          <p>confidence: {String(props.predVal[1].toFixed() * 100) + "%"} </p>
        </div>
      )
    } else {
      return (<div />)
    }
  }


  const headerKeys = Object.keys(Object.assign({}, ...array));

  return (
    <div className="px-4 sm:px-6 lg:px-8">

      <div className="mt-12 sm:flex sm:items-center">
        <div className="sm:flex-auto">
          <h1 className="text-xl font-semibold text-gray-900">Step 1: Upload Training Set</h1>

        </div>
        <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none">
          <form>
            <input
              type={"file"}
              id={"csvFileInput"}
              accept={".csv"}
              onChange={handleOnChange}
            />
            <button
              type="button"
              className="inline-flex items-center justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 sm:w-auto"

              onClick={(e) => {
                handleOnSubmit(e);
              }}
            >
              IMPORT CSV
            </button>
          </form>
        </div>
      </div>

      <div className="mt-8 sm:lex-auto flex flex-col">
        <div className="-my-2 -mx-4 overflow-x-auto sm:-mx-6 lg:-mx-8">
          <div className="max-h-96 overflow-scroll inline-block min-w-full py-2 align-middle md:px-6 lg:px-8">
            <div className="shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
              <table className="min-w-full divide-y divide-gray-300" style={{ maxHeight: "30rem" }}>
                <thead className="bg-gray-50">
                  <tr key={"header"}>
                    {headerKeys.map((key) => (
                      <th scope="col" className="sticky py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">{key}</th>
                    ))}
                  </tr>
                </thead>

                <tbody className="divide-y divide-gray-200 bg-white">
                  {array.map((item) => (
                    <tr key={item.id}>
                      {Object.values(item).map((val) => (
                        <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">{val}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 sm:flex sm:items-center">

        <div>
          <h1 className="text-xl font-semibold text-gray-900">Step 2: Select Features and Outputs</h1>
          <div className="mt-8 row" style={{ display: 'flex' }}>

            <div className="left-panel box" style={{ flex: 5 }}>
              <fieldset>
                <legend className="text-lg font-medium text-gray-900">Features</legend>
                <div className="mt-4 max-w-xs divide-y divide-gray-200 border-t border-b border-gray-200">
                  {validFeatures.map((feature) => (
                    <div key={feature.featureName} className="relative flex items-start py-4">
                      <div className="min-w-0 flex-1 text-sm">
                        <label htmlFor={`person-${feature.featureName}`} className="select-none font-medium text-gray-700">
                          {feature.featureName}
                        </label>
                      </div>
                      <div className="ml-3 flex h-5 items-center">
                        <input
                          checked={feature.selected}
                          onChange={handleOnFeatureCheck}
                          id={`${feature.featureName}`}
                          name={`person-${feature.featureName}`}
                          type="checkbox"
                          className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </fieldset>
            </div>

            <div className="sm:ml-32 right-panel box" style={{ flex: 5 }}>
              <label className="text-base font-medium text-gray-900">Outputs</label>
              <fieldset className="mt-4">
                <div className="space-y-4">
                  {validOutputs.map((output) => (
                    <div key={output.outputName} className="flex items-center">
                      <input
                        id={output.outputName}
                        checked={output.selected}
                        onChange={handleOnOutputCheck}
                        name="notification-method"
                        type="radio"
                        defaultChecked={output.outputName === 'email'}
                        className="h-4 w-4 border-gray-300 text-indigo-600 focus:ring-indigo-500"
                      />
                      <label htmlFor={output.outputName} className="ml-3 block text-sm font-medium text-gray-700">
                        {output.outputName}
                      </label>
                    </div>
                  ))}
                </div>
              </fieldset>
            </div>

            <div className="mt-4 sm:mt-0 sm:ml-32 sm:flex-none">
              <button
                type="button"
                className="inline-flex items-center justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 sm:w-auto"

                onClick={(e) => {
                  handleOnTrain(e);
                }}
              >
                Train Model
              </button>
            </div>
          </div>
        </div>
      </div>






      <div className="mt-8 sm:flex sm:items-center">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">Step 3: Predict</h1>
          <div className="pb-12 mt-8 row" style={{ display: 'flex' }}>
            <div className="left-panel box" style={{ flex: 5 }}>
              <label className="text-base font-medium text-gray-900">Select your Model</label>
              <fieldset className="mt-4">
                <div className="space-y-4">
                  {pastModels.map((model) => (
                    <div key={model.model_id} className="flex items-center">
                      <input
                        id={model.model_id}
                        name="model-select"
                        type="radio"
                        defaultChecked={model.model_id === currModel}
                        className="h-4 w-4 border-gray-300 text-indigo-600 focus:ring-indigo-500"
                        onChange={handleOnSelectModel}
                      />
                      <label htmlFor={model.model_id} className="ml-3 block text-sm font-medium text-gray-700">
                        {model.modelType + " " + model.date}
                      </label>
                    </div>
                  ))}
                </div>
              </fieldset>
            </div>


            <div className="sm:ml-32" style={{ flex: 5 }}>
              <label className="text-base font-medium text-gray-900">Enter in prediction parameters</label>

              <ul className="divide-y divide-gray-200">
                {currModel.features.map((feature) => (
                  <li key={feature.feature} className="flex py-4">
                    <div className="ml-3">
                      <p className="text-sm font-medium text-gray-900">{feature.feature}</p>
                      <input
                        type="email"
                        name="email"
                        id={feature.feature}
                        className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                        value={feature.weight}
                        onChange={handleOnChangeParams}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mt-4 sm:mt-0 sm:ml-32 sm:flex-none">
              <button
                type="button"
                className="inline-flex items-center justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 sm:w-auto"
                onClick={handleOnPredict}
              >
                Predict
              </button>
              <Prediction predVal={currPrediction} />
              <p>Metrics: </p>
              <p>Accuracy: {currModel.auc}</p>
              <p>True positive rate: {currModel.tpr}</p>
              <p>True negative rate: {currModel.tnr} </p>
              <p>Precision: {currModel.ppv}</p>

            </div>
          </div>
        </div>
      </div>
    </div>
  )
}


export default App;
