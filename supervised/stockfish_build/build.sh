#!/bin/bash

# Clone the repository using gh cli
gh repo clone official-stockfish/Stockfish -- --depth 1 --branch sf_14
cd Stockfish/src

# Display available make targets
make help

# Build the neural network file
make net

# Build Stockfish binary with the specified architecture
make build ARCH=x86-64-modern