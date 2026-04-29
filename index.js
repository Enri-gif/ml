const csvUrl = 'http://127.0.0.1:8080/concrete.csv';
let normalisedFeature;
let normalisedLabel;
let trainingFeatures;
let trainingLabels;
let testingFeatures;
let testingLabels;
let model;
let points;

const modelStatus = document.getElementById("model-status");
const trainButton = document.getElementById("train-button");
const testButton = document.getElementById("test-button");
const loadButton = document.getElementById("load-button");
const saveButton = document.getElementById("save-button");
const predictButton = document.getElementById("predict-button");

trainButton.disabled = true;
testButton.disabled = true;
saveButton.disabled = true;
predictButton.disabled = true;

function normalise(tensor, min = null, max = null) {
    const realMin = min ?? tensor.min();
    const realMax = max ?? tensor.max();

    const normalized = tensor.sub(realMin).div(realMax.sub(realMin));

    return {
        tensor: normalized,
        min: realMin,
        max: realMax
    };
}

function denormalise(tensor, min, max) {
    const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];
    if (featureDimensions && featureDimensions > 1) {
        // More than one feature
        // Split into separate tensors
        const features = tf.split(tensor, featureDimensions, 1);
        // Denormalise
        const denormalised = features.map((featureTensor, i) => denormalise(featureTensor, min[i], max[i]));
        const returnTensor = tf.concat(denormalised, 1);
        return returnTensor;
    }
    else {
        const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
        return denormalisedTensor;
    }
}
async function run() {
    await tf.ready();

    const concreteDataset = tf.data.csv(csvUrl);

    const pointsDataset = concreteDataset.map(record => ({
        age: record.age,
        strength: record.strength
    }));
    points = await pointsDataset.toArray();

    const featureValues = points.map(p => [p.age]);

    const labelValues = points.map(p => p.strength);

    const featureTensor = tf.tensor2d(featureValues);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    // Use numbers for min/max
    const featureMin = (await featureTensor.min().data())[0];
    const featureMax = (await featureTensor.max().data())[0];

    const labelMin = (await labelTensor.min().data())[0];
    const labelMax = (await labelTensor.max().data())[0];

    normalisedFeature = normalise(
        featureTensor,
        tf.scalar(featureMin),
        tf.scalar(featureMax)
    );

    normalisedLabel = normalise(
        labelTensor,
        tf.scalar(labelMin),
        tf.scalar(labelMax)
    );

    tf.dispose([featureTensor, labelTensor]);

    // Split into training/testing sets
    const numExamples = normalisedFeature.tensor.shape[0];
    const trainSize = Math.floor(numExamples * 0.8);
    const testSize = numExamples - trainSize;

    trainingFeatures = normalisedFeature.tensor.slice([0, 0], [trainSize, 1]);
    testingFeatures = normalisedFeature.tensor.slice([trainSize, 0], [testSize, 1]);
    trainingLabels = normalisedLabel.tensor.slice([0, 0], [trainSize, 1]);
    testingLabels = normalisedLabel.tensor.slice([trainSize, 0], [testSize, 1]);

    return { trainingFeatures, trainingLabels };
}

async function plot(pointsArray, featureName, classKey, size = 400, equalizeClassSizes) {
    const allSeries = {};

    pointsArray.forEach(p =>{
        const seriesName = classKey;
        let series = allSeries[seriesName];
        if(!series){
            series = [];
            allSeries[seriesName] = series;
        }
        series.push(p);
    });

    if(equalizeClassSizes){
        let maxLength = null;
        Object.values(allSeries).forEach(series =>{
            if(maxLength === null || series.length < maxLength && series.length >= 100){
                maxLength = series.length
            }
        });

        Object.keys(allSeries).forEach(keyName =>{
            allSeries[keyName] = allSeries[keyName].slice(0, maxLength);
            if(allSeries[keyName].length < 100){
                delete allSeries[keyName];
            }
        })
    };

    tfvis.render.scatterplot(
        { name: `${featureName} vs Concrete Strength` },
        { 
            values: Object.values(allSeries),
            series: Object.keys(allSeries)
        },
        { xLabel: featureName, yLabel: "Concrete Strength" }
    );
}

async function plotPredictionLine() {
    const min = normalisedFeature.min;
    const max = normalisedFeature.max;

    const minVal = normalisedFeature.min.dataSync()[0];
    const maxVal = normalisedFeature.max.dataSync()[0];

    const xsTensor = tf.linspace(minVal, maxVal, 100).reshape([100, 1]);

    const normalisedXs = normalise(xsTensor, min, max);

    const preds = model.predict(normalisedXs.tensor);

    const ysTensor = denormalise(
        preds,
        normalisedLabel.min,
        normalisedLabel.max
    );

    const xs = await xsTensor.data();
    const ys = await ysTensor.data();

    console.log("XS sample:", xs.slice(0, 10));
    console.log("YS sample:", ys.slice(0, 10));

    // 🔥 HARD FILTER invalid values
    const predictedPoints = [];

    for (let i = 0; i < xs.length; i++) {
        const x = xs[i];
        const y = ys[i];

        if (
            Number.isFinite(x) &&
            Number.isFinite(y)
        ) {
            predictedPoints.push({ x, y });
        }
    }

    console.log("Valid points:", predictedPoints.length);

    tfvis.render.scatterplot(
        { name: "Age vs Concrete Strength" },
        {
            values: [
                points.map(p => ({
                    x: p.age,
                    y: p.strength
                })),
                predictedPoints
            ],
            series: ["data", "prediction"]
        },
        {
            xLabel: "age",
            yLabel: "Concrete Strength"
        }
    );
}

function createModel() {
    model = tf.sequential();

    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [1]
    }));

    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 1
    }));

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError',
        metrics: ['mse']
    });

    return model;
}

async function init() {
    const result = await run();
    trainingFeatures = result.trainingFeatures;
    trainingLabels = result.trainingLabels;

    trainButton.disabled = false;
    modelStatus.textContent = "Not yet trained";
}

init();
document.getElementById("toggle-button").addEventListener("click", () => tfvis.visor().toggle());

async function plotParams(weight, bias) {
    model.getLayer(null, 0).setWeights([
        tf.tensor2d([[weight]]),
        tf.tensor1d([bias]),
    ]);
    await plotPredictionLine();
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer 1" }, layer);
}

async function trainModel(trainingFeatures, trainingLabels) {
    model = await createModel();
    const modelStatus = document.getElementById("model-status");
    const { onEpochEnd } = tfvis.show.fitCallbacks({ name: 'Training Performance' }, ['loss']);

    await model.fit(trainingFeatures, trainingLabels, {
        epochs: 20,
        shuffle: true,
        callbacks: {
            onEpochBegin: async() => {
                tfvis.show.layer({ name: 'Layer 1' }, model.getLayer(undefined, 0));
                tfvis.show.layer({ name: 'Layer 2' }, model.getLayer(undefined, 1));
                // tfvis.show.layer({ name: 'Layer 3' }, model.getLayer(undefined, 2));
                // tfvis.show.layer({ name: 'Layer 4' }, model.getLayer(undefined, 2));
            },
            onEpochEnd: async (epoch, logs) => {
                await onEpochEnd(epoch, logs);
                await plotPredictionLine();
                // await plot(points, "age", "strength");
                modelStatus.textContent = `Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(8)}`;
            }
        }
    });

    tf.dispose([trainingFeatures, trainingLabels]);
    return model;
}

async function testModel(testingFeatures, testingLabels) {
    const result = model.evaluate(testingFeatures, testingLabels);

    const lossTensor = result[0];

    const loss = await lossTensor.data();

    console.log("Loss:", loss[0]);

    return loss[0];
}

async function saveModel() { await model.save('localstorage://my-model-1'); }
async function loadModel() { model = await tf.loadLayersModel('localstorage://my-model-1'); }

async function predict() {
    const predictionInputOne = parseFloat(
        document.getElementById("prediction-input-1").value
    );

    if (isNaN(predictionInputOne)) {
        alert("Please enter a valid number");
        return;
    }

    const outputValue = tf.tidy(() => {
        const inputTensor = tf.tensor2d([[predictionInputOne]]);

        const normalisedInput = normalise(
            inputTensor,
            normalisedFeature.min,
            normalisedFeature.max
        );

        const prediction = model.predict(normalisedInput.tensor);

        const outputTensor = denormalise(
            prediction,
            normalisedLabel.min,
            normalisedLabel.max
        );

        return outputTensor.dataSync()[0];
    });

    document.getElementById("prediction-output").innerHTML =
        `Predicted concrete strength: ${outputValue.toFixed(2)} MPa`;
}

trainButton.addEventListener("click", async () => {
    if (!trainingFeatures || !trainingLabels) return console.error("Training data not ready yet.");

    trainButton.disabled = true;
    loadButton.disabled = true;
    await trainModel(trainingFeatures, trainingLabels);

    modelStatus.textContent += "\nTraining complete";
    await plotPredictionLine();

    loadButton.disabled = false;
    trainButton.disabled = false;
    testButton.disabled = false;
    saveButton.disabled = false;
    predictButton.disabled = false;
});

testButton.addEventListener("click", async () => {
    if (!model || !testingFeatures || !testingLabels) return console.error("Model/test data not ready.");

    const loss = await testModel(testingFeatures, testingLabels);
    document.getElementById("testing-status").textContent = `Loss: ${loss.toFixed(8)}`;
});

loadButton.addEventListener("click", async () => {
    await loadModel();
    await plotPredictionLine();
    predictButton.disabled = false;
});

saveButton.addEventListener("click", async () => { await saveModel(); });
predictButton.addEventListener("click", () => { predict(); });

