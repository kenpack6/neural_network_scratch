# Neural Network From Scratch

A hand-rolled feedforward neural network, built to learn how backpropagation actually works instead of just calling `.backward()`. The repo has two parallel implementations trained on MNIST digit classification:

1. **From scratch (NumPy only)** — dense layers, ReLU, softmax, and cross-entropy, with forward pass, backpropagation, and gradient descent all implemented manually.
2. **PyTorch version** — the same architecture idea, built with `torch.nn`, used as a reference/sanity check for the scratch implementation.

## Why this project

Frameworks like PyTorch make it easy to train a network without ever computing a gradient by hand. The goal here was the opposite: implement forward propagation, backpropagation, and gradient descent from first principles with just NumPy, then validate the math and results against an equivalent PyTorch model.

Every gradient used in this implementation was hand-derived on paper before being translated into code — see [Derivations](#derivations) below for the worked math behind the `backward()` methods.

## Project Structure

```
neural_network_scratch/
├── dataset/
│   └── mnist.pkl.gz                   # MNIST data (train/validation/test)
├── src/neural_network_no_pytorch/
│   ├── layer_class.py                 # Layer_Dense, ReLU, SoftMax, CrossEntropy
│   ├── network_class.py               # Network: forward pass, backprop, update
│   ├── main.py                        # Training script for the scratch model
│   ├── practice_files/                # Scratch space for testing ideas
│   └── neural_network_pytorch/
│       ├── neural_network_class.py    # Equivalent model built with torch.nn
│       ├── main.py                    # Train / test the PyTorch model
│       ├── visualize_mnist.py         # Plot sample MNIST digits and predictions
│       └── mnist_model_*.pth          # Saved PyTorch model checkpoints
├── pyproject.toml
└── uv.lock
└── nn_derivations.pdf                 # Handwritten derivations and notes during implementation
```

## Status

**Complete.** Both the from-scratch NumPy implementation and the PyTorch reference model are fully built, trained, and evaluated on MNIST.

| Component | Scratch (NumPy) | PyTorch |
|---|---|---|
| Forward pass | ✅ Done | ✅ Done |
| Backpropagation | ✅ Done | ✅ Done (autograd) |
| Weight updates (SGD) | ✅ Done | ✅ Done (Adam) |
| Training loop | ✅ Done | ✅ Done |
| Evaluation / inference | ✅ Done | ✅ Done |

The network architecture (scratch version) is `784 → 15 → ReLU → 15 → ReLU → 10`, trained on flattened 28x28 MNIST images.

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires Python 3.14+.

```bash
# Clone the repo
git clone https://github.com/kenpack6/neural_network_scratch.git
cd neural_network_scratch

# Install dependencies
uv sync
```

**Note:** the `dataset/` folder is git-ignored. Download the pickled MNIST dataset (`mnist.pkl.gz`) separately and place it in `dataset/` before running any training scripts.

## Usage

Train the from-scratch model:
```bash
uv run python -m neural_network_no_pytorch.main
```

Train the PyTorch reference model:
```bash
uv run python -m neural_network_no_pytorch.neural_network_pytorch.main
```

Visualize sample MNIST digits:
```bash
uv run python -m neural_network_no_pytorch.neural_network_pytorch.visualize_mnist
```

## What I Learned

- Deriving and implementing backpropagation manually for dense layers, ReLU, softmax, and cross-entropy loss
- How the chain rule composes across layers — each layer only needs the gradient from the layer ahead of it, plus its own local Jacobian
- Why softmax's Jacobian is a full matrix (not just a diagonal one), since every output probability depends on every input logit
- Why fusing softmax with cross-entropy collapses an ugly Jacobian-vector product down to a single clean subtraction, `p - y`
- Numerical stability tricks (e.g., shifting inputs before softmax, clipping before log/division)
- Using a PyTorch implementation as a correctness check for a from-scratch one

## Derivations

All derivatives implemented in `layer_class.py` were worked out by hand before being coded. The core results are summarized below.

### Softmax Jacobian

For $p = \text{softmax}(z)$, the Jacobian entry for output $i$ with respect to input $j$ is:

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

where $\delta_{ij}$ is the Kronecker delta. In matrix form, across all $i,j$ at once:

$$J = \text{diag}(p) - pp^\top$$

### Softmax + Cross-Entropy (fused gradient)

For cross-entropy loss $L = -y^\top \log(p)$ against one-hot label $y$, chaining the loss gradient through the softmax Jacobian:

$$\frac{\partial L}{\partial z} = J^\top \frac{\partial L}{\partial p} = \left(\text{diag}(p) - pp^\top\right)(-y \oslash p)$$

which simplifies — via cancellation of the $1/p_i$ terms and the one-hot identity $\sum_i y_i = 1$ — to:

$$\frac{\partial L}{\partial z} = p - y$$

This is why the combined softmax + cross-entropy backward pass is so cheap: no explicit Jacobian is ever built, and the result is a single vector subtraction.

### Standalone Softmax backward (generic, layer-decoupled)

When softmax is implemented as its own layer (not fused with cross-entropy), it must apply the full Jacobian-vector product to whatever gradient `dout` flows in from upstream:

$$\frac{\partial L}{\partial z} = \left(\text{diag}(p) - pp^\top\right) g, \qquad g = \texttt{dout}$$

Per-entry, this reduces to:

$$\left(\frac{\partial L}{\partial z}\right)_j = p_j\left(g_j - \sum_i p_i g_i\right)$$

implemented without ever materializing the full $n\times n$ Jacobian matrix, by computing the dot product $\sum_i p_i g_i$ directly.

### Linear (dense) layer backward

For $y = xW + b$, with incoming gradient $g = \frac{\partial L}{\partial y}$:

$$\frac{\partial L}{\partial x} = gW^\top \qquad \frac{\partial L}{\partial W} = x^\top g \qquad \frac{\partial L}{\partial b} = \sum_n g_{n,:}$$

The pattern: whichever matrix is on a given side of the forward-pass multiplication gets swapped and transposed on that side of the backward-pass product.

### Gradient descent update rule

Derived from a first-order Taylor expansion of $L$ around the current parameters, choosing the update direction that guarantees a decrease in loss for small enough step size $\eta$:

$$W \leftarrow W - \eta \frac{\partial L}{\partial W} \qquad b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

## Roadmap / Possible Extensions

- [ ] Add accuracy/loss curve plots over training epochs
- [ ] Compare scratch model accuracy against the PyTorch model on the same architecture
- [ ] Experiment with additional optimizers (momentum, Adam) in the scratch implementation
- [ ] Extend to a configurable number of layers/hidden units
