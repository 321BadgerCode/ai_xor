# XOR Neural Network in C (4-bit Weight Precision)

This project implements a fully functional **feedforward neural network** in **pure C** from scratch to solve the classic **XOR classification problem**. All network weights use **4-bit fixed-point quantization (Q2.2 format)**, emulating the constraints of embedded or low-resource systems.

---

## ✨ Features

- ✅ Fully-connected feedforward network: **2–2–1 architecture**
- ✅ 4-bit quantized weights (`-2.0` to `+1.75`, step: `0.25`)
- ✅ Floating-point "shadow weights" for smooth learning updates
- ✅ Fixed-point quantization at every weight update step
- ✅ Sigmoid activation with clamping to prevent saturation
- ✅ Backpropagation with mean squared error (MSE) loss
- ✅ Trains to solve XOR within 1,000 epochs

---

## 🧠 Network Topology

```
Input Layer (2 neurons)
        ↓
Hidden Layer (2 neurons, sigmoid)
        ↓
Output Layer (1 neuron, sigmoid)
```

This topology is the smallest feedforward neural network capable of solving the XOR problem — a non-linearly separable binary classification task.

---

## 🧮 Fixed-Point Arithmetic

Weights are quantized to **Q2.2 fixed-point** format:

- Precision: 2 fractional bits (step size = `0.25`)
- Range: `-2.00` to `+1.75`
- Represented using **4-bit signed integers**

This simulates environments such as microcontrollers or neuromorphic chips where full 32-bit floats are unavailable or undesirable.

---

## 🛠️ Build and Run

```bash
git clone https://github.com/321BadgerCode/ai_xor.git
cd ./ai_xor/
gcc ./main.c -o ./ai_xor -lm
./ai_xor
```

Sample output:
```
--- TRAINING ---
Epoch     0 | Loss: 0.494306
Epoch   100 | Loss: 0.472542
Epoch   200 | Loss: 0.469250
Epoch   300 | Loss: 0.447136
Epoch   400 | Loss: 0.380697
Epoch   500 | Loss: 0.268273
Epoch   600 | Loss: 0.143832
Epoch   700 | Loss: 0.072933
Epoch   800 | Loss: 0.042877
Epoch   900 | Loss: 0.028768
Epoch  1000 | Loss: 0.021094

--- XOR TEST ---
Input: 0 0 => Output: 0.0866 (0)
Input: 0 1 => Output: 0.9018 (1)
Input: 1 0 => Output: 0.9007 (1)
Input: 1 1 => Output: 0.1253 (0)
```

---

## 📂 File Overview

| File              | Description                                      |
|-------------------|--------------------------------------------------|
| `main.c`          | Main source code with training and testing logic |
| `README.md`       | This documentation                               |

---

## 📚 How It Works

- The network is initialized with random 4-bit fixed-point weights.
- Forward propagation computes neuron activations using sigmoid.
- Backpropagation updates shadow float weights using gradient descent.
- Weights are re-quantized to 4-bit values after each update.
- The loss (MSE) is monitored every 100 epochs.
- After training, the model classifies all 4 XOR inputs.

---

## 🔬 Why XOR?

The XOR function:
```
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0
```

It is **not linearly separable**, so a perceptron (no hidden layer) cannot solve it. A hidden layer introduces non-linearity, enabling correct classification.

---

## 📈 Loss Convergence (Expected)

- Initial loss ~ 0.5
- Final loss after 1k epochs < 0.1
- Output values converge to:
  - Close to 0 for class 0
  - Close to 1 for class 1

---

## 📌 Notes

- The network uses `float` for internal computation (e.g., sigmoid) and training.
- Inference uses quantized weights, simulating constrained environments.
- Ideal for embedded AI, fixed-point DSPs, and quantization experiments.

---

## 🧑‍💻 Author

> Handcrafted in C with bit-level control over neural computations.  
> If you like neural networks, compilers, or bare-metal ML, you might like this project.

---

## 📜 License

[MIT License](LICENSE)