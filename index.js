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

    const pointsDataset = concreteDataset.map(record => record);
    points = await pointsDataset.toArray();

    const featureValues = points.map(p => [
        p.cement,
        p.slag,
        p.ash,
        p.water,
        p.superplastic,
        p.coarseagg,
        p.fineagg,
        p.age,
        p.water_cement_ratio,
        p.total_binder,
        p.aggregate_to_cement,
        p.cement_water_interaction,
        p.age_strength_proxy
    ]);
    const labelValues = points.map(p => p.strength);

    // [ N samples, 13 features ]
    const featureTensor = tf.tensor2d(featureValues);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    const featureMin = featureTensor.min(0);
    const featureMax = featureTensor.max(0);

    const labelMin = (await labelTensor.min().data())[0];
    const labelMax = (await labelTensor.max().data())[0];

    normalisedFeature = normalise(featureTensor, featureMin, featureMax);

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

    trainingFeatures = normalisedFeature.tensor.slice([0, 0], [trainSize]);
    testingFeatures = normalisedFeature.tensor.slice([trainSize, 0], [testSize]);
    trainingLabels = normalisedLabel.tensor.slice([0, 0], [trainSize]);
    testingLabels = normalisedLabel.tensor.slice([trainSize, 0], [testSize]);

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
    const featureMin = normalisedFeature.min;
    const featureMax = normalisedFeature.max;

    const featureIndex = 11;

    const base = [
        141.3, 212.0, 0.0, 203.5, 0.0,
        971.8, 748.5,
        28,
        1.44, 353.3, 12.17,
        0,
        5.29
    ];

    const xsArray = [];

    for (let i = 0; i <= 100000; i += 1) {
        const row = [...base];
        row[featureIndex] = i;
        xsArray.push(row);
    }

    const xsTensor = tf.tensor2d(xsArray);

    // ---- normalize input
    const normXs = normalise(xsTensor, featureMin, featureMax);

    const preds = model.predict(normXs.tensor);

    const ysTensor = denormalise(
        preds,
        normalisedLabel.min,
        normalisedLabel.max
    );

    const xsData = xsTensor.arraySync();
    const ysData = await ysTensor.data();

    const predictedPoints = [];

    for (let i = 0; i < xsData.length; i++) {
        predictedPoints.push({
            x: xsData[i][featureIndex],
            y: ysData[i]
        });
    }

    tfvis.render.scatterplot(
        { name: "cement_water_interaction vs Concrete Strength (fixed mix)" },
        {
            values: [
                points.map(p => ({
                    x: p.cement_water_interaction,
                    y: p.strength
                })),
                predictedPoints
            ],
            series: ["real data", "prediction"]
        },
        {
            xLabel: "cement_water_interaction",
            yLabel: "strength"
        }
    );

    tf.dispose([xsTensor, normXs.tensor, preds, ysTensor]);
}

function createModel() {
    model = tf.sequential();

    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [13]
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
    const inputValue = parseFloat(
        document.getElementById("prediction-input-1").value
    );

    if (isNaN(inputValue)) {
        alert("Please enter a valid number");
        return;
    }

    const featureIndex = 11;

    // ---- IMPORTANT: use RAW dataset mean, not normalized
    const rawFeatures = points.map(p => [
        p.cement,
        p.slag,
        p.ash,
        p.water,
        p.superplastic,
        p.coarseagg,
        p.fineagg,
        p.age,
        p.water_cement_ratio,
        p.total_binder,
        p.aggregate_to_cement,
        p.cement_water_interaction,
        p.age_strength_proxy
    ]);

    const base = tf.tensor2d(rawFeatures).mean(0).arraySync();

    base[featureIndex] = inputValue;

    const inputTensor = tf.tensor2d([base]);

    const normInput = normalise(
        inputTensor,
        normalisedFeature.min,
        normalisedFeature.max
    );

    const prediction = model.predict(normInput.tensor);

    const outputTensor = denormalise(
        prediction,
        normalisedLabel.min,
        normalisedLabel.max
    );

    const result = (await outputTensor.data())[0];

    document.getElementById("prediction-output").innerHTML =
        `Predicted concrete strength: ${result.toFixed(2)} MPa`;

    tf.dispose([inputTensor, normInput.tensor, prediction, outputTensor]);
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

