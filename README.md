# My VGON Implementation

This project is my Python implementation of the VGON (Variational Quantum Optimization with Generative Networks) algorithm, as described in the paper [Variational Optimization for Quantum Problems using Deep Generative Networks](https://arxiv.org/abs/2203.04316). [The author also implemented](https://github.com/zhangjianjianzz/VGON).

VGON is a hybrid quantum-classical algorithm that uses a generative model (a Variational Autoencoder) to find the optimal parameters for a quantum circuit. This approach can be used to solve a variety of quantum optimization problems, such as finding the ground state of a molecule or solving combinatorial optimization problems.

## Installation

To install the necessary dependencies, you can use [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Alternatively, you can install the dependencies manually using pip:

```bash
pip install tensorflow-cpu tqdm matplotlib
```

## Usage

The main entry point for running the VGON algorithm is the `vgon.simulator.simulate_with_circuit` function. This function takes a quantum circuit, the number of parameters in the circuit, and the hyperparameters for the generative model as input. It then trains the generative model to find the optimal parameters for the circuit.

Here is a simple example of how to use the `vgon` library:

```python
import tensorflow as tf
from vgon.simulator import simulate_with_circuit
from vgon.graph import plot

# Define a simple quantum circuit
def my_circuit(parameters):
    # In a real application, this would be a quantum circuit
    return tf.reduce_sum(tf.sin(parameters))

# Set the hyperparameters
num_parameters = 10
encoder_layers = [20, 10]
decoder_layers = [10, 20]
latent_dim = 2
epoch_num = 1000
batch_size = 2

# Run the VGON algorithm
loss_list, q_grad_list, model = simulate_with_circuit(
    num_parameters,
    encoder_layers,
    decoder_layers,
    latent_dim,
    epoch_num,
    batch_size,
    my_circuit
)

# Plot the results
plot(loss_list, q_grad_list)
```

## Project Structure

The project is organized as follows:

- `vgon/vgon.py`: Contains the implementation of the Variational Autoencoder (VAE) used as the generative model.
- `vgon/quantum_circuit.py`: Contains functions for creating and simulating quantum circuits (currently commented out, but can be adapted to use libraries like Cirq or TensorFlow Quantum).
- `vgon/loss_function.py`: Defines the loss function for the VGON algorithm.
- `vgon/trainer.py`: Contains the training step function for the generative model.
- `vgon/simulator.py`: Contains the main simulation loop for the VGON algorithm.
- `vgon/sampler.py`: Contains functions for sampling from the trained generative model.
- `vgon/planner.py`: An example script that shows how to use the VGON algorithm.
- `vgon/graph.py`: A simple utility for plotting the results of the simulation.

## References

- [Variational Optimization for Quantum Problems using Deep Generative Networks](https.arxiv.org/abs/2203.04316)
- [Original implementation](https://github.com/zhangjianjianzz/VGON)
