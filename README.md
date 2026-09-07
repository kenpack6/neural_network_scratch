# Neural Network From Scratch

A hand-rolled feedforward neural network, built to learn how backpropagation actually works instead of just calling `.backward()`. The repo has two parallel implementations trained on MNIST digit classification:

1. **From scratch (NumPy only)** — dense layers, ReLU, softmax, and cross-entropy, all implemented manually, including forward pass and (in progress) backprop.
2. **PyTorch version** — the same architecture idea, built with `torch.nn`, used as a reference/sanity check for the scratch implementation.

## Why this project

Frameworks like PyTorch make it easy to train a network without ever computing a gradient by hand. The goal here was the opposite: implement forward propagation, backpropagation, and gradient descent from first principles with just NumPy, then compare it against an equivalent PyTorch model to validate the math and results.

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
```

## Status

This is an active work in progress, built for learning rather than production use.

| Component | Scratch (NumPy) | PyTorch |
|---|---|---|
| Forward pass | ✅ Done | ✅ Done |
| Backpropagation | 🚧 In progress | ✅ Done (autograd) |
| Weight updates (SGD) | 🚧 In progress | ✅ Done (Adam) |
| Training loop | 🚧 In progress | ✅ Done |
| Evaluation / inference | ⬜ Planned | ✅ Done |

The current network architecture (scratch version) is `784 → 15 → ReLU → 15 → ReLU → 10`, trained on flattened 28x28 MNIST images.

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

## What I'm Learning

- Deriving and implementing backpropagation manually for dense layers, ReLU, softmax, and cross-entropy loss
- How the chain rule composes across layers (each layer only needs the gradient from the layer ahead of it)
- Numerical stability tricks (e.g., shifting inputs before softmax, clipping before log/division)
- Using a PyTorch implementation as a correctness check for a from-scratch one

## Roadmap

- [ ] Finish `backward()` for `Layer_Dense` (weight/bias gradients)
- [ ] Finish `update()` for gradient descent
- [ ] Wire up `Network.back_prop()` to chain gradients through all layers
- [ ] Add accuracy tracking and a validation loop
- [ ] Compare scratch model accuracy against the PyTorch model on the same architecture
