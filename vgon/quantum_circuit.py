# import cirq, sympy, numpy as np, tensorflow as tf, tensorflow_quantum as tfq

# # 回路用関数
# def random_pauli():
#     # np.random.seed(1)
#     rand_v = 3 *  np.random.random()
#     if rand_v < 1:
#         return cirq.rx
#     elif rand_v < 2:
#         return cirq.ry
#     else:
#         return cirq.rz

# def make_czlayer(qubits):
#     czlayer = []
#     for i in range(len(qubits)):
#         if i % 2 != 0:
#             continue
#         if i + 1 < len(qubits):
#             czlayer.append(cirq.CZ(qubits[i],qubits[i+1]))
#     for i in range(len(qubits)):
#         if i % 2 != 1:
#             continue
#         if i + 1 < len(qubits):
#             czlayer.append(cirq.CZ(qubits[i],qubits[i+1]))
#     return czlayer

# def make_cirq_circuit(number_of_qubits, number_of_layers):
#     number_of_parameters = number_of_qubits * number_of_layers
#     # 1量子ビットのRZゲートと計算基底での測定を含むPQCを作成
#     qubits = cirq.GridQubit.rect(1,number_of_qubits)
#     parameter_symbols = ['theta_' + str(i) for i in range(number_of_parameters)]
#     sympy_param_symbols = list(map(sympy.Symbol, parameter_symbols))

#     gates = []
#     rotate_gates = [cirq.ry(np.pi /4)(qubit) for qubit in qubits]
#     gates.extend(rotate_gates)

#     for i in range(number_of_layers):
#         random_gates = [random_pauli()(param)(qubit) for param, qubit in zip(sympy_param_symbols[i * number_of_qubits:(i+1) * number_of_qubits],qubits)]
#         czgates = make_czlayer(qubits)
#         gates.extend(random_gates + czgates)

#     circuit = cirq.Circuit(*gates)
#     # circuit = cirq.Circuit(*(random_gates))
#     print(circuit)
#     paulis = list(map(cirq.Z,qubits))
#     pauli_sums = tfq.convert_to_tensor([paulis])

#     # symbol_values = tf.convert_to_tensor([[np.pi / 4] * number_of_parameters], dtype=tf.dtypes.float32)
#     return circuit, pauli_sums, parameter_symbols, qubits

# # 量子回路を作成
# def make_circuit_mps(circuit, pauli_sums, parameter_symbols,):
#     # circuit, pauli_sums, parameter_symbols, _ = make_cirq_circuit(number_of_qubits,number_of_layers)
#     # 回路をテンソルに変換
#     program = tfq.convert_to_tensor([circuit])
#     # シンボルと値を定義
#     symbol_names = tf.convert_to_tensor(parameter_symbols, dtype=tf.dtypes.string)
#     print(f"symbol_names {symbol_names}")
#     # MPSでシミュレーション
#     return lambda param_tesnor : tfq.math.mps_1d_expectation(program, symbol_names, param_tesnor, pauli_sums)