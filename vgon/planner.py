from simulator import simulate_with_parameters
from graph import plot

### 実装
# parameters
number_of_qubits = 4
number_of_layers = 1
number_of_parameters = number_of_qubits * number_of_layers

input_dim = number_of_parameters
output_dim = number_of_parameters
encoder_layers = [20,10]
decoder_layers = [10,20]
latent_dim = 2
epoch_num = 1000
batch_size = 2 # バッチサイズを設定

loss_list, q_grad_list, model, cirq_circuit, pauli_sums, parameter_symbols, qubits = simulate_with_parameters(
    number_of_qubits,
    number_of_layers,
    encoder_layers,
    decoder_layers,
    latent_dim,
    epoch_num,
    batch_size
)

plot(loss_list,q_grad_list)
print(f"final loss {loss_list[len(loss_list)-1]}")