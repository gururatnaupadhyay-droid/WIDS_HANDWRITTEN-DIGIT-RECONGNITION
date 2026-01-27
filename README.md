Dataset for the MNIST Pixel values is from kaggle: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

The dataset has 60000 training data and 10000 testing data

The main Project is implemented via Tensorflow which is a library that is used to make Convolutional Neural Network learning much more user friendly and straightforward.
---

1. Data Preparation: Setting the Stage

Before the model can learn, the raw data needs pre processing. The script loads the training and testing sets using Pandas, then moves into three critical preprocessing steps:

Normalization: Pixel values are scaled from their original range (0–255) down to 0–1. This helps the model converge much faster.
Reshaping: The flat 784-pixel rows are reshaped into 28x28x1 tensors. This 3D structure is vital because it preserves the spatial relationships between pixels—something a flat list of numbers can't do.
One-Hot Encoding: The labels (0–9) are converted into categorical vectors. Instead of the digit "3," the model sees `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.

---
2. The Architecture: Designing the Brain

The model follows a classic CNN architecture, which is specifically designed to mimic how the human visual cortex processes information.

The Visual Processing Stage (Convolutional Layers)
This stage mimics the human eye by scanning the image for patterns.

First Convolutional Layer (Conv2D): This is the entry point. It uses 32 different filters to scan the 28x28 pixel image. It looks for basic features like edges, vertical lines, and horizontal strokes.

First Max Pooling Layer: To make the model efficient, this layer shrinks the image dimensions. By looking at 2x2 pixel squares and keeping only the brightest (most important) pixel, it reduces the amount of data the computer has to process without losing the key features.

Second Convolutional Layer (Conv2D): Now that the basics are found, this layer uses 64 filters to look for more complex "features of features"—such as the specific curves that distinguish a "6" from an "8."

Second Max Pooling Layer: Another round of compression occurs here, ensuring the model remains fast and focuses only on the most dominant visual signals.

The Translation Stage (Flattening)
Flatten Layer: At this point, the data is still in a 2D "map" format. The Flatten layer unrolls this 2D grid into a single, long 1D list of numbers. Think of it like taking a folded map and stretching it out into one long line so the next layer can read it from start to finish.

The Decision-Making Stage (Dense Layers)
This is the "brain" of the model where the actual classification happens.

Hidden Dense Layer: This layer has 128 neurons that weigh the importance of all the features found earlier. It uses the ReLU activation function, which acts as a gate—it allows important signals to pass through while blocking irrelevant ones (turning negative values to zero).

Dropout Layer: This is a crucial "safety" step. It randomly shuts off 50% of the neurons during each training step. This forces the model to not rely too heavily on any single pixel or feature, making it much better at recognizing handwritten digits it has never seen before.

Output Dense Layer: The final layer has 10 neurons, representing digits 0 through 9. It uses the Softmax function to turn the model's internal logic into a probability (e.g., "I am 98% sure this is a 4"). 

The inclusion of **Dropout (0.5)** is important. It randomly "turns off" half the neurons during training, forcing the model to find multiple paths to the right answer rather than relying on a few "heavy-lifter" neurons.

---

3. The "Smart Stop" Callback

Perhaps the most practical part of this code is the `StopAtAccuracy` class. In many machine learning projects, we waste time and computing power running a model through 50 or 100 epochs when it has already mastered the task by epoch 5.

This custom **Keras Callback** monitors the validation accuracy. As soon as the model hits the **99% accuracy** threshold, the script triggers an early exit. As seen in the output, the model hit the target by **Epoch 3**, saving significant time and preventing the model from over-tuning to the training data.

---

4. Results: 
Validation Accuracy: 99.00%
Test Accuracy: 98.94%



