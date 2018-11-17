/* start node http-server in terminal in same folder to serve files */
var model;
const n = new npyjs();

var x_test_3d = null;
async function predict(model) {
  model.summary();
  //var prediction = model.predict(x_test_3d);
  x_test_3d.print()
  console.log(x_test_3d.shape)
  const output = await model.predict(x_test_3d);
  const axis = -1;
  const classValues = tf.argMax(output, 1);
  output.print(true);
  classValues.print(true);
}

fetch("http://127.0.0.1:8080/data.txt")
  .then(response => response.json())
  .then(json => createTensorFromJson(json));


function createTensorFromJson(json) {
  console.log(json);
  const x_test_2d = tf.tensor2d(json, [42, 175]);
  x_test_3d = x_test_2d.reshape([42, 175, 1]);
  tf.loadModel('http://127.0.0.1:8080/model/model.json').then(model => {
    predict(model);
  });
}

/*n.load('http://127.0.0.1:8080/ts_features.npy', (array, shape) => {
    // `array` is a one-dimensional array of the raw data
    // `shape` is a one-dimensional array that holds a numpy-style shape.
    //x_test_3d = tf.tensor3d(restored_array, shape);
    const x_test_2d = tf.tensor2d(array, [42, 175]);
    x_test_3d = x_test_2d.reshape([42, 175, 1]);

    tf.loadModel('http://127.0.0.1:8080/model/model.json').then(model => {
      predict(model);
    });
});*/


