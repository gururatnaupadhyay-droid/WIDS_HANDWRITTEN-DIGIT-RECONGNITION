Dataset for the MNIST Pixel values is from kaggle: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

The dataset has 60000 training data and 10000 testing data

The main Project is implemented via Tensorflow which is a library that is used to make Convolutional Neural Network learning much more user friendly and straightforward.


1. Data Preparation: Setting the Stage

Before the model can learn, the raw data needs pre processing. The script loads the training and testing sets using Pandas, then moves into three critical preprocessing steps:

Normalization: Pixel values are scaled from their original range (0–255) down to 0–1. This helps the model converge much faster.
Reshaping: The flat 784-pixel rows are reshaped into 28x28x1 tensors. This 3D structure is vital because it preserves the spatial relationships between pixels—something a flat list of numbers can't do.
One-Hot Encoding: The labels (0–9) are converted into categorical vectors. Instead of the digit "3," the model sees `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.


2. The Architecture: Designing the Brain

The model follows a classic CNN architecture, which is specifically designed to mimic how the human visual cortex processes information.

| Layer Type | Purpose | Key Parameters |
| **Conv2D** | Feature Extraction | 32 filters, 3x3 kernel |
| **MaxPooling** | Data Compression | 2x2 pool size |
| **Conv2D** | Advanced Feature Extraction | 64 filters, 3x3 kernel |
| **Flatten** | Reshaping for Output | Converts 2D to 1D |
| **Dense** | Decision Making | 128 neurons, ReLU |
| **Dropout** | Overfitting Prevention | 50% neuron deactivation |
| **Softmax** | Final Classification | 10 output classes |

The inclusion of **Dropout (0.5)** is important. It randomly "turns off" half the neurons during training, forcing the model to find multiple paths to the right answer rather than relying on a few "heavy-lifter" neurons.

---

3. The "Smart Stop" Callback

Perhaps the most practical part of this code is the `StopAtAccuracy` class. In many machine learning projects, we waste time and computing power running a model through 50 or 100 epochs when it has already mastered the task by epoch 5.

This custom **Keras Callback** monitors the validation accuracy. As soon as the model hits the **99% accuracy** threshold, the script triggers an early exit. As seen in the output, the model hit the target by **Epoch 3**, saving significant time and preventing the model from over-tuning to the training data.

---

4. Results: 
Validation Accuracy: 99.00%
Test Accuracy: 98.94%



