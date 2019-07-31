const trainingHTML = document.getElementById("training");
const valueToTestHTML = document.getElementById("value-to-test");
const expectedHTML = document.getElementById("expected");
const predictionHTML = document.getElementById("prediction");
const valueNewEpochsHTML = document.getElementById("new-epochs");
let model;

// Training the model with 250 epochs
run(250);


async function run(epochs) {
  trainingHTML.innerHTML = `Training the model with ${epochs} epochs...`;
  valueToTestHTML.disabled = true
  valueToTestHTML.value = '';
  valueNewEpochsHTML.disabled = true
  valueNewEpochsHTML.value = '';

  await createAndTrainModel(epochs);

  trainingHTML.innerHTML = `Trained model (${epochs} epochs), you can ask for predictions now`;
  valueToTestHTML.disabled = false
  valueNewEpochsHTML.disabled = false
}


async function createAndTrainModel(epochs = 250) {
    // Create a simple model.
    model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
  
    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
  
    // Generate some synthetic data for training. (y = 2x - 1)
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);
    
    // Train the model using the data.
    await model.fit(xs, ys, {epochs});
    
}



async function makePredition() {
  const value = Number(this.value);
  const x = tf.tensor2d([value], [1, 1]);
  const predictionY = model.predict(x).dataSync();

  predictionHTML.innerHTML = predictionY;
  expectedHTML.innerHTML = (value * 2) - 1 ;

}


valueToTestHTML.addEventListener('change', makePredition);
valueNewEpochsHTML.addEventListener('change', function(){
  const epochs = Number(this.value);
  run(epochs)
});
