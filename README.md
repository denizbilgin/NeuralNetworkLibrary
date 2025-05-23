# NeuralNetworkLibrary

This repository contains a neural network library built from scratch using only Python and NumPy. It aims to provide a fundamental understanding of neural network mechanics without relying on high-level deep learning frameworks.

---

## Features

- **Pure Python and NumPy Implementation**  
  Designed for clarity and educational purposes, demonstrating how neural networks work at a low level.

- **Modular Design**  
  Separate modules for key components of a neural network.

### Core Components

- **Activation Functions**  
  Includes common activation functions used in neural networks.

- **Loss Functions**  
  Provides various loss functions for different types of machine learning tasks.

- **Metrics**  
  Offers metrics to evaluate model performance.

- **Neural Layers**  
  Implements different types of neural layers, allowing for flexible network architectures.

- **Neural Network**  
  The main class for building, training, and evaluating neural networks.

### Classification Examples

- **Binary Classification**  
  An example script (`MainBinaryClassification.py`) to demonstrate binary classification tasks.

- **Multiclass Classification**  
  An example script (`MainMulticlassClassification.py`) to showcase multiclass classification tasks.

- **Utility Functions**  
  A `Utils.py` module for helper functions.

---

## Getting Started

### Prerequisites

- Python 3.12 or higher  
- NumPy

### Installation

No specific installation is required. Simply clone the repository:

```bash
git clone https://github.com/denizbilgin/NeuralNetworkLibrary.git
cd NeuralNetworkLibrary
```

---

## Example Usage

You can easily define your own neural network model and train it using the provided classes:

```python
# Determine layers
hidden1 = NeuralLayer(20, ReLU())
hidden2 = NeuralLayer(7, ReLU())
hidden3 = NeuralLayer(5, ReLU())
output_layer = NeuralLayer(1, Sigmoid())

# Create neural network
my_network = NeuralNetwork(
    train_x, train_y,
    [hidden1, hidden2, hidden3, output_layer],
    BinaryCrossEntropy(),
    [accuracy],
    0.001
)

# Train the network
costs = my_network.fit(2000)

# Make predictions with the trained model
predictions = my_network.predict(test_x)

# Evaluate the test set
my_network.evaluate(test_x, test_y)
```

You are free to configure different architectures, activation functions, and loss metrics according to your task.

---

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

---

## Contact

For any questions or feedback, feel free to reach out.
