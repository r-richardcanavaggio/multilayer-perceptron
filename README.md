# Multilayer Perceptron (MLP) From Scratch

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Type Checked](https://img.shields.io/badge/Mypy-Strict%20Verified-blue?style=flat)](http://mypy-lang.org/)
[![School](https://img.shields.io/badge/42-AI%20Specialization-000000?style=flat&logo=42&logoColor=white)](https://42.fr/)

A fully vectorized, production-grade Multilayer Perceptron implemented from first mathematical principles using pure NumPy. Features a modular PyTorch-like architecture, analytical backpropagation engine, extensible optimizer strategy pattern (SGD & Adam), and 3D loss surface topology analysis via SVD parameter projection.

---

## 📌 Academic & Constraints Context

This project is part of the **42 School AI / Data Science curriculum**. The objective is to construct a deep learning classification system from scratch to diagnose breast cancer tumors using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset (569 instances, 30 continuous real-valued features per cell nucleus).

### Non-Negotiable Engineering Constraints
- **Zero High-Level Deep Learning Frameworks**: Complete ban on PyTorch, TensorFlow, Keras, Scikit-learn, and Autograd.
- **Pure NumPy Vectorization**: All linear transformations, chain-rule gradient backpropagations, batch scheduling, and optimization passes must be implemented purely through matrix calculus.
- **Strict Data Integrity**: Elimination of data leakage by ensuring normalization statistics ($\mu, \sigma$) are derived solely from the training split.
- **Static Typing**: 100% strict type safety enforced across all modules via `mypy --strict`.

---

## 🏗️ Architecture & Design Patterns

The repository is organized following modern object-oriented deep learning framework design principles (analogous to PyTorch Core), cleanly separating data streaming, tensor computation, layer state, loss evaluation, and parameter updates.

### Forward / Backward Data Flow

```text
[ Input Features X ]
         │
         ▼
┌──────────────────┐
│   DataLoader     │ ◄── Shuffled Mini-batches (B samples)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      Forward Pass: Z = X·W + b, A = σ(Z)
│    DenseLayer    ├────────────────────────────────────────────┐
└────────▲─────────┘                                            │
         │                Backward Pass: dX = dZ·Wᵀ             ▼
         │             ◄───────────────────────────      ┌──────────────┐
         │                                               │ DenseLayer   │
         │                                               │ (Output)     │
         │                                               └──────┬───────┘
         │                                                      │ A_out
         │                                                      ▼
┌────────┴─────────┐                                     ┌──────────────┐
│  Optimizer (ABC) │ ◄── Updates: W -= α·ΔW, b -= α·Δb   │ Trainer      │
│  (SGD / Adam)    │                                     │ (Loss / BCE) │
└──────────────────┘                                     └──────────────┘
```

### Component Breakdown

| Module | Core Role & Engineering Design |
|:---|:---|
| `src/Dataset.py` | Encapsulates input matrices and targets; handles indexing and dimensional validation. |
| `src/DataLoader.py` | Iterable mini-batch streaming generator with random epoch shuffling (`__iter__`, `yield`). |
| `src/DenseLayer.py` | Encapsulates layer weights $W$, biases $b$, analytical activation derivatives, and parameter updates. |
| `src/NeuralNetwork.py` | Sequential execution container orchestrating forward propagation, reverse backpropagation, and `.npz` serialization. |
| `src/Optimizer.py` | **Strategy Pattern** with abstract `Optimizer` base class, `SGDOptimizer`, `AdamOptimizer`, and `OptimizerFactory`. |
| `src/Trainer.py` | Decoupled training engine orchestrating the epoch loop, mini-batch updates, validation metrics, and early stopping. |
| `src/History.py` | Strongly-typed recorder (`HistoryDict`) capturing loss, accuracy, and parameter vectors ($\theta_t$) across epochs. |
| `src/EarlyStopping.py` | Regularization callback monitoring validation divergence with configurable patience. |
| `src/LossLandscape.py` | Dimensionality reduction engine (SVD / PCA) generating 3D loss surface topologies around optimal weights. |
| `src/Math.py` | Numerically stable activation functions (Sigmoid, Softmax), Glorot uniform initialization, and BCE loss. |

---

## 🧠 Engineering Highlights

### 1. Vectorized Backpropagation & Combined Analytical Derivative

Rather than calculating individual scalar derivatives, all operations are vectorized across batch dimension $B$.

#### Output Layer Simplification

Computing the derivative of Softmax paired with Binary Cross-Entropy loss analytically simplifies the gradient calculation. The partial derivative with respect to the output pre-activation $Z^{[L]}$ reduces directly to:

$$
\frac{\partial \mathcal{L}}{\partial Z^{[L]}} = \hat{Y} - Y
$$

This eliminates the need to compute or allocate the intermediate $K \times K$ Softmax Jacobian matrix.

#### Hidden Layer Gradient Contraction

Backpropagation proceeds upstream through hidden layers using pure matrix products:

$$
\frac{\partial \mathcal{L}}{\partial Z^{[l]}} = \frac{\partial \mathcal{L}}{\partial A^{[l]}} \odot \sigma'\left(Z^{[l]}\right)
$$

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{B} (A^{[l-1]})^T \frac{\partial \mathcal{L}}{\partial Z^{[l]}}, \quad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{B} \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial Z^{[l]}}
$$

$$
\frac{\partial \mathcal{L}}{\partial A^{[l-1]}} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} (W^{[l]})^T
$$

### 2. Adaptive Moment Estimation (Adam) From Scratch
Implemented the Adam optimizer with bias correction to accelerate convergence in non-convex loss surfaces:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_{t} = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

The layer remains agnostic of the optimization mechanics: `DenseLayer` delegates weight updates to `Optimizer.update()`, enabling zero-cost runtime swapping of optimization algorithms via `OptimizerFactory`.

### 3. Numerical Stability & Quality Engineering
- **Glorot / Xavier Uniform Initialization**: Layer weights are initialized within bounds $\pm \frac{1}{\sqrt{n_{\text{in}} \cdot n_{\text{out}}}}$ to avoid vanishing/exploding gradients in initial passes.
- **Log-Domain Precision**: Binary Cross-Entropy evaluation is safe against numerical underflow and zero-division artifacts.
- **Strict Typing Verification**: The entire codebase strictly conforms to `mypy --strict` with zero typing errors across all 17 source files:

```text
$ mypy --strict src/ train.py predict.py split.py
Success: no issues found in 17 source files
```

### 4. 3D Loss Landscape Topology (Bonus)
To inspect the optimization trajectory and verify convergence towards a stable minimum:
1. Flattens network parameters at each epoch $t$ into a high-dimensional vector $\theta_t \in \mathbb{R}^P$.
2. Centers the trajectory matrix around the final converged parameter vector $\theta^*$.
3. Applies **Singular Value Decomposition (SVD)** to extract the two leading principal variation directions $d_1, d_2$.
4. Computes a 2D mesh grid over $(u, v)$ coordinates, reconstructing temporary parameter sets $\theta(u, v) = \theta^* + u \cdot d_1 + v \cdot d_2$.
5. Evaluates full dataset loss across the surface and overlays the optimization path in 3D space.

---

## 🚀 Pipeline & Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/multilayer_perceptron.git
cd multilayer_perceptron

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Codebase Integrity

```bash
mypy --strict src/ train.py predict.py split.py
```

### 3. Data Preprocessing & Splitting

Splits raw data (80% train, 20% test), maps diagnoses (`M` $\to 1$, `B` $\to 0$), and applies z-score standardization derived strictly from the training partition:

```bash
python3 split.py
```

### 4. Model Training

Train the network with configurable topologies, activation functions, optimizers, learning rates, and early stopping patience:

```bash
# Train using Adam optimizer with Early Stopping
python3 train.py -o Adam -l 24 24 24 -e 50 -s 10 -lR 0.01

# Train using Stochastic Gradient Descent (SGD)
python3 train.py -o SGD -l 24 24 24 -e 50 -lR 0.1
```

### 5. Prediction & Evaluation

Loads saved weights (`data/model_weights.npz`) and evaluates performance on unseen test data:

```bash
python3 predict.py
```

---

## 💻 Showcase & Real Terminal Outputs

### Model Training Execution (`train.py`)

```text
$ python3 train.py -o Adam -e 50 -s 10 -lR 0.01
Successfully loaded data in data/training_data.csv
Successfully loaded data in data/testing_data.csv
Epoch 1: Training loss = 0.53218571 | Validation loss = 0.15417870
Epoch 2: Training loss = 0.12117479 | Validation loss = 0.06284457
Epoch 3: Training loss = 0.08656392 | Validation loss = 0.09342783
Epoch 4: Training loss = 0.07894161 | Validation loss = 0.07438113
Epoch 5: Training loss = 0.07239458 | Validation loss = 0.05860561
Epoch 6: Training loss = 0.06764356 | Validation loss = 0.05787028
Epoch 7: Training loss = 0.07538802 | Validation loss = 0.04441908
Epoch 8: Training loss = 0.07494354 | Validation loss = 0.06011735
Epoch 9: Training loss = 0.06347635 | Validation loss = 0.25850142
Epoch 10: Training loss = 0.06515142 | Validation loss = 0.19916024
Epoch 11: Training loss = 0.06039511 | Validation loss = 0.08689131
Epoch 12: Training loss = 0.07120848 | Validation loss = 0.04860856
Epoch 13: Training loss = 0.07917582 | Validation loss = 0.21246300
Epoch 14: Training loss = 0.07266126 | Validation loss = 0.20677901
Epoch 15: Training loss = 0.05801347 | Validation loss = 0.09039113
Epoch 16: Training loss = 0.05927256 | Validation loss = 0.18160378
Epoch 17: Training loss = 0.05709562 | Validation loss = 0.14260090
Epoch 18: Training loss = 0.05863500 | Validation loss = 0.20152522
Model saved succesfully in data/model_weights.npz
```
*Note: Training terminated automatically at epoch 18 due to Early Stopping regularization after validation loss reached its optimum at epoch 7.*

### Evaluation & Confusion Matrix (`predict.py`)

```text
$ python3 predict.py
Binary cross-entropy loss: 0.11147211665439706

                  Predicted Positives  Predicted Negatives
Actual Positives                   26                    0
Actual Negatives                    4                   84

Accuracy: 96.49%
Precision: 86.67%
Recall: 100.00%
F1 Score: 92.86%
```

> **Clinical Relevance**: In medical diagnostics (cancer detection), **Recall is the critical metric**. A Recall of **100.00%** indicates zero false negatives (all malignant cases correctly identified).

---

## 📊 Training Dynamics & Visualizations

| Training & Validation Loss | Training & Validation Accuracy |
|:---:|:---:|
| ![Loss Curve](plots/loss.png) | ![Accuracy Curve](plots/accuracy.png) |

### 3D Loss Surface Topology (SVD Trajectory Projection)

![3D Loss Landscape](plots/loss_landscape.png)
