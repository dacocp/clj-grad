# Clj-Grad: A Simple Clojure Neural Network Library

Clj-Grad is a lightweight neural network library implemented in Clojure. The library allows you to create simple neural network architectures using multi-layer perceptrons (MLPs) and provides an implementation of the backpropagation algorithm for gradient computation.

## Features

- Simple implementation of multi-layer perceptrons (MLPs)
- Basic operations for manipulating `Value` instances
- A set of basic activation functions: tanh, ReLU
- Supports addition, multiplication, subtraction, division, and power operations
- A small set of neural network components: Neuron, Layer, and MLP
- Protocol for defining custom neural network components


## Usage

### Importing the library
    
    ```clojure
    (ns your-namespace
      (:require [clj-grad.engine :as e]
                [clj-grad.nn :as nn]))
    ```

### Creating a simple neural network
    
    ```clojure
    ;; Create a neural network with 2 input neurons, 1 hidden layer of 4 neurons, and 1 output neuron
    (def net (nn/mlp 2 [4 1]))
    ```

### Forward pass

    ```clojure
    ;; Perform a forward pass with the neural network
    (def input [1 2])
    (def output (net input))
    ```
### Backward pass

    The backward pass is not yet implemented, but you can contribute by implementing the backward function in the clj-grad.engine namespace.

    ```clojure
    ;; TODO: Perform backpropagation on the computation graph rooted at the given Value
    ;; (e/backward output)
    ```


