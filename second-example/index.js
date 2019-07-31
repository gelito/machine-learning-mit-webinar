import * as tf from '@tensorflow/tfjs';
import * as MobilenetFactory from '@tensorflow-models/mobilenet';
import {IMAGENET_CLASSES} from './imagenet_classes';

const IMAGE_SIZE = 224;

const imgHTML = document.getElementById('img');
const preparingHtml = document.getElementById('preparing');
const fileHtml = document.getElementById('file');
const modelFromLayersHtml = document.getElementById('modelFromLayers');
const modelFromMobilenetv1Html = document.getElementById('modelFromMobilenetv1');
const modelFromMobilenetv2Html = document.getElementById('modelFromMobilenetv2');

let modelFromLayers;
let modelFromMobilenetv1;
let modelFromMobilenetv2;

let predictionsFromLayers;
let predictionsFromMobilenetv1;
let predictionsFromMobilenetv2;


async function prepareModels() {

    const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
    modelFromLayers = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    
    modelFromMobilenetv1 = await MobilenetFactory.load({ version:1, alpha:1 });
    modelFromMobilenetv2 = await MobilenetFactory.load({ version:2, alpha:1 });
}

async function makePredictions(img) {

    // Classify the image using Mobilenet layer.
    const startTimeLayers = performance.now();
    predictionsFromLayers = await getPredictionFromLayers(img);
    printPrediction('from Layers', predictionsFromLayers, startTimeLayers, modelFromLayersHtml);

    // Classify the image Mobilenetv1.
    const startTimeMobilenetv1 = performance.now();
    predictionsFromMobilenetv1 = await modelFromMobilenetv1.classify(img);
    printPrediction('from Mobilenet V1', predictionsFromMobilenetv1, startTimeMobilenetv1, modelFromMobilenetv1Html);


    // Classify the image Mobilenetv2.
    const startTimeMobilenetv2 = performance.now();
    predictionsFromMobilenetv2 = await modelFromMobilenetv2.classify(img);
    printPrediction('from Mobilenet V2', predictionsFromMobilenetv2, startTimeMobilenetv2, modelFromMobilenetv2Html);
}

async function getPredictionFromLayers(imgElement) {

    // tidy Executes the provided function fn and after it is executed, cleans up all intermediate tensors
    const logits = tf.tidy(() => {
        // tf.browser.fromPixels() returns a Tensor from an image element.
        const img = tf.browser.fromPixels(imgElement).toFloat();
    
        const offset = tf.scalar(127.5);
        // Normalize the image from [0, 255] to [-1, 1].
        const normalized = img.sub(offset).div(offset);
    
        // Reshape to a single-element batch so we can pass it to predict.
        const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    
        // Make a prediction through mobilenet.
        return modelFromLayers.predict(batched);
      });  
      
    return getTopKClasses(logits, 3);
}


/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopKClasses(logits, topK) {
    const values = await logits.data();

    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
        valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
        return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
        topkValues[i] = valuesAndIndices[i].value;
        topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
        topClassesAndProbs.push({
        className: IMAGENET_CLASSES[topkIndices[i]],
        probability: topkValues[i]
        })
    }
    return topClassesAndProbs;
}


async function printPrediction(title, data, start, html) {
    console.log ("here", title, html);
    let total = performance.now() - start;
    total = (total / 1000).toFixed(4);

    html.innerHTML = `<h4>Result for ${title} (${total}s)</h4><ul>`;

    for (let i = 0; i < data.length; i++) {
        const element = data[i];
        html.innerHTML += `<li>${element.className}: ${element.probability.toFixed(4)}</li>`;
    }
    
    html.innerHTML += '</ul>';
}

// Watch for new images updates
fileHtml.addEventListener('change', evt => {
  const file = evt.target.files[0];
  let reader = new FileReader();

  imgHTML.src = '';
  reader.onload = e => {
    // Fill the image & call predict.
    img.src = e.target.result;
    img.width = IMAGE_SIZE;
    img.height = IMAGE_SIZE;
    img.onload = () => makePredictions(img);
  };

  // Read in the image file as a data URL.
  reader.readAsDataURL(file);
});


async function run() {

    const startPreparing = performance.now();
    await prepareModels();
    let totalPreparing = performance.now() - startPreparing;
    totalPreparing = (totalPreparing / 1000).toFixed(4);

    preparingHtml.innerHTML = `Models ready in ${totalPreparing}s`;

    fileHtml.disabled = false;
}

run();


