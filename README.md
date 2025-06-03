<h1 align="center">Numpy Neural Networks</h1>

<h2>Description:</h2>

<p>
To test my understanding of Feedforward and Convolutional Neural Networks, I decided to implement both from scratch without using any specialized machine learning libraries like PyTorch. Instead, this project primarily uses NumPy to build the necessary machine learning components like backpropagation, optimizers, and loss functions.
</p>


<h2>Usage:</h2>
<ol>
  <li>Activate a virtual environment.</li>
  <li>Run <code>pip install -r requirements.txt</code> to install the dependencies.</li>
  <li>Run <code>main.py</code> to train a model. (Inference is done in the <code>inference.ipynb</code> Jupyter notebook)</li>
</ol>


<h2>Hyperparameters:</h2>
<p>All hyperparameters are defined in <code>main.py</code>.</p>
<ul>
  <li><code>DATASET</code> (str): Specifies the dataset to use. Either <code>MNIST</code> and <code>Fashion_MNIST</code>.</li>
  <li><code>ONE_DIM</code> (bool): Determines whether to flatten the image data into vectors (<code>True</code>, for feedforward neural networks) or keep them as matrices (<code>False</code>, for convolutional neural networks).</li>
  <li><code>MODEL</code> (src.nn.models): The model to train:
    <ul>
      <li><strong>CNN:</strong> Pass a sequence of integers representing the number of channels in each layer. Also requires <code>input_dims</code> (tuple) and <code>output_dim</code> (int), which should already be defined. The number of hidden layers is <code>len(args) - 1</code>.  
        <br>For Example: <code>CNN(1, 64, 32, input_dims, output_dim)</code> creates two hidden layers: one with 1 input and 64 output channels, and another with 64 input and 32 output channels.
      </li>
      <li><strong>FFNN:</strong> Pass a sequence of integers representing the number of neurons in each layer. The first and last values (input/output dimensions) should already be defined.  
        <br>Example: <code>FFNN(128, 64, 32)</code> creates two hidden layers: one with 128 input and 64 output neurons, and another with 64 input and 32 output neurons.
      </li>
    </ul>
  </li>
  <li><code>LEARNING_RATE</code> (float): The learning rate used by the optimizer.</li>
  <li><code>OPTIMIZER</code> (src.nn.optimizers): The optimization algorithm to use. Currently supports Stochastic Gradient Descent and Mini-Batch Gradient Descent.</li>
  <li><code>EPOCHS</code> (int): The number of training epochs.</li>
  <li><code>BATCH_SIZE</code> (int): The number of samples per batch for training.</li>
  <li><code>MODEL_NAME</code> (str): Name of the model to be saved or loaded.</li>
</ul>

<h2>Results:</h2>
<p>
  This project went much better than I expected. I successfully implemented an FFNN and a CNN, along with backpropagation for both, optimizers, loss functions, activation functions, and the saving/loading of parameters.
</p>
<p>
<strong>Below are the results of the FFNN and CNN I trained:</strong>
<ul>
  <li>
    </br>The CNN trained on the Fashion_MNIST dataset achieved the following average loss and accuracy on the test data, which I am happy with:

![image](https://github.com/user-attachments/assets/5c6cf495-f649-455c-908e-9df35f220126)
    </br>Here are a few of its predictions:

![clipboard2](https://github.com/user-attachments/assets/ed5fc002-7e97-4e4d-92f7-469407db686c)
  </li>
  <li>
    </br>The FFNN trained on the MNIST dataset achieved the following average loss and accuracy on the test data, which I am also happy with:

![image](https://github.com/user-attachments/assets/b639ba66-312b-487c-8451-2a7b30411db9)
    </br>Here are a few of its predictions:

![clipboard5](https://github.com/user-attachments/assets/b7446d5c-3935-4ded-8440-570ea5a9fba5)
  </li>
</ul>
</p>

</ul>
</p>





