# ⚡ Custom CUDA/CuPy Neural Network Engine from Scratch

A high-performance, GPU-accelerated deep learning framework built from scratch in Python using **CuPy** for hardware acceleration and manual matrix calculus for backpropagation.

Designed to train on the **MNIST Handwritten Digit Dataset**, reaching **~96%+ test accuracy** without relying on PyTorch, TensorFlow, or Autograd.

---

## 🚀 Features

* **Zero High-Level ML Frameworks:** Built entirely using raw linear algebra operations.
* **GPU-Accelerated:** Written using `CuPy` to leverage NVIDIA CUDA cores for high-throughput matrix operations.
* **He (Kaiming) Weight Initialization:** Prevents vanishing/exploding gradients in hidden layers.
* **Activations & Derivatives:** Implemented custom vectorized **Leaky ReLU** and stable **Softmax** loss routines.
* **Vectorized Backprop Engine:** Hand-derived chain rule gradients for weights, biases, and activation layers across mini-batches.

---

## 📐 Architecture

| Layer Type | Input Dim | Output Dim | Activation / Function |
| :--- | :--- | :--- | :--- |
| **Linear Layer 1** | 784 | 128 | Leaky ReLU ($\alpha=0.001$) |
| **Linear Layer 2** | 128 | 10 | Numerically Stable Softmax |

* **Loss Function:** Categorical Cross-Entropy Loss
* **Optimizer:** Mini-Batch Stochastic Gradient Descent (SGD)

---

## 🛠️ Requirements & Setup

### Prerequisites
* NVIDIA GPU with CUDA drivers installed
* Python 3.8+

