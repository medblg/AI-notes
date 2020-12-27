TensorFlow notes
===

Refs:
---
- https://www.tensorflow.org/tutorials/
- https://www.youtube.com/watch?v=tPYj3fFJGjk
- https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ/playlists - tensorflow channel

- https://www.youtube.com/watch?v=5ECD8J3dvDQ&list=PLQY2H8rRoyvxcmHHRftsuiO1GyinVAwUg&index=3
- https://www.youtube.com/watch?v=KNAWp2S3w94

- https://www.youtube.com/watch?v=KNAWp2S3w94&list=PLZKsYDC2S5rM6yKBs5ParXS6RWda6iAnK


# instalation :
`pip install tensorflow`
- https://medium.com/@cran2367/install-and-setup-tensorflow-2-0-2c4914b9a265
- install virtualenv
`sudo pip3 install -U virtualenv`
## steps:
1. create virt env :
- virtualenv(  
  - isolated env for python projects
  - indep set of packages(dependencies)
  - keep same version of package -> avoid conflicting
)
- instantiate virtual env:
- `python3 -m venv --system-site-packages ./venv`
- `virtualenv --system-site-packages -p python3 tf_2`
  - `--system-site-packages` -> pjcts within virtenv (name tf_2)can access global site-packages
  - `-p python3 tf_2` -> set python3 as interpreter for our tf_2 virtualenv.
  - `tf_2` is created as physical dir at location of virtualenv.
	- can be skipped `-p python3` if virtualenv -> installed with python3.

2. activate virtual environment

- cmd : `source tf_2/bin/activate`
- terminal change to (tf_2) $

3. install tensorflow 2.0
(tf_2) $ pip install --upgrade pip
- update pip: `pip3 install --upgrade pip`
- install : `pip3 install --upgrade tensorflow==2.0.0-rc1`
- packages installed within the venv : `pip list`

4. test installation:

- (tf) $
```
python3 -c "import tensorflow as tf; x = [[2.]]; print('tensorflow version', tf.__version__); print('hello, {}'.format(tf.matmul(x, x)))"
```
```
python3 -c "import matplotlib.pyplot as plt; import tensorflow as tf; layers=tf.keras.layers; import numpy as np; print(tf.__version__)"
```
- exemples: test with MNIST (fashion_mnist) image class examples.

5. deactivate the virtualenv

- before closing : `deactivate`


# quick start
- use keras :
  - build neural network classifies images
  - train neural netw
  - evaluate the accuracy of the model
(
- google colab notebook ? = https://colab.research.google.com/
write and execute python in browser

https://colab.research.google.com/notebooks/markdown_guide.ipynb

- https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)
  - anaconda -> free open srce distr for python and R -
  - for scientific computing(data science, machine learning..)
)

- using tensorflow2 in google colab :
  - https://colab.research.google.com/notebooks/tensorflow_version.ipynb

```
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

- see keras api on tensorflow `https://www.tensorflow.org/api_docs/python/tf/keras/`

# basic ML tasks with keras:
- `https://www.tensorflow.org/guide/keras/overview`
- to build train models
- tf.keras.version -> tf.keras might be diff keras from pypi
- saving model's weigths ->
  - tf.keras defaults to checkpoint format.
  - pass `save_format='h5'` -> to use hdf5 | filename ends with .h5

## build simple model
1. Sequential model:
- Keras -> assemble layers to build models.
- model -> graph of layers.
  - most common -> stack of layers(tensorflow.keras.layers) -> `tf.keras.Sequential`
  - ex: multi-layer perceptron
  - see more `https://www.tensorflow.org/guide/keras/functional`
```
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
# adds density-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# add another layer
model.add(layers.Dense(64, activation='relu'))
# add an output layer with 10 output units
model.add(layers.Dense(10))
```
2. configure layers
- many `tf.keras.layers`
- common arguments:
  - **activation**: set activ functi. && by default no activation is set.
  - **kernel_initializer** and **bias_initializer**:
    - creates layer's weights(kernel && bias)
    - `kernel` -> defaults -> `Glorot uniform`
    - `bias` -> defaults -> `zeros`
  - **kernel_regularizer** and **bias_regularizer**:
    - applies layer's wigths
    - ex: L1 | L2 -> regularization. -> by default -> no regularization.

- instantiate `tf.keras.layers.Dense` layers:
  - relu layer:
`layers.Dense(64, activation='relu')`
  - linear layer with L1 regularization factor 0.01
`layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))`
  - linear layer with L2 regularization factor 0.0.1
`layers.Dense(64, kernel_regularizers=tf.keras.regularizers.l2(0.0.1))`
  - linear layer with kernel initialized to random orthogonal matrix:
`layers.Dense(64, kernel_initializer='orthogonal')`
  - linear layer with bias vector initialized to 2.0s
`layers.Dense(64, bias_initializer=tf.keras.initiliazers.Constant(2.0))`

3. train and evaluate:
- training : calling the `compile` method.
  - takes 3 argmts:
    - `optimizer` : training procedure. ex:
      - `tf.keras.optimizers.Adam` or `tf.keras.optimizers.SGD`
      - for use default parameters -> use `'adam'`, `'sgd'`
    - `loss` : func to minimize during optimization.
      - common ->
        - `mse(mean square error)`
        - `categorical_crossentropy`
        - `binary_crossentropy`
    - `metrics` -> monitor training.
    - `run_eagerly=True` -> addition -> to make it train and evaluates eagerly.
```
model = tf.keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(32,)),
layers.Dense(64, activation='relu'),
layers.Dense(10)
])
model.compile(
	optimizer=tf.keras.optimizers.Adam(0.01),
	loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
	metrics=['accuracy']
)
```

- train from numpy data:

  - numpy -> to train and evaluate model -> model is fit to trainning data.
  - `tf.keras.Model.fit()` -> 3 imp args:
    - **epochs** -> training strctred into epochs.
      - epoch -> one iteration over entire input data. (smaller batches)
    - **batch_size** ->  specify size of batches
    - **validation_data** -> monitor performnce
      - tuple of inputs & labels -> display loss & metrics
```
import numpy as np
data = np.random.random(1000, 32)
labels = np.random.random(1000, 10)

model.fit(data, labels, epochs=10, batch_size=32)
```
**using validation_data**:
```
import numpy as np
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))
model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
```

- train from tf.data datasets:
- `https://www.tensorflow.org/guide/data`



# lexique machine learning:
https://developers.google.com/machine-learning/glossary#logits
