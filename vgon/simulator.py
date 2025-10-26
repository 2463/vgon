from .vgon import VGON
# from quantum_circuit import make_cirq_circuit, make_circuit_mps
from .loss_function import loss_function_maker, multi_circuit_loss_function_maker,Circuit
from .trainer import train_step_maker
import tensorflow as tf
from tqdm import tqdm

#test2.ipynb を参照
# def simulate_with_parameters(
#         number_of_qubits,
#         number_of_layers,
#         encoder_layers,
#         decoder_layers,
#         latent_dim,
#         epoch_num,
#         batch_size
# ):
#     number_of_parameters = number_of_qubits * number_of_layers
#     input_dim = number_of_parameters
#     output_dim = number_of_parameters
#         ### 古典の準備
#     # モデルのインスタンス化
#     model = VGON(
#         input_dim,
#         output_dim,
#         encoder_layers,
#         decoder_layers,
#         latent_dim
#     )
#     # 最適化アルゴリズムの選択
#     optimizer = tf.optimizers.Adam()
#     # ランダム input の準備
#     def random_input(length,input_dimension):
#         return tf.random.uniform(shape=(length,input_dimension))
#     # トレーニングデータの準備
#     x_train = random_input(epoch_num * batch_size, input_dim)  # トレーニングデータをここにロード
#     # データセットのバッチ化
#     train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
#     ### 量子の準備
#     # circuit
#     cirq_circuit, pauli_sums, parameter_symbols, qubits = make_cirq_circuit(number_of_qubits,number_of_layers)
#     circuit = make_circuit_mps(cirq_circuit, pauli_sums, parameter_symbols,)
#     # loss_function
#     loss_function = loss_function_maker(circuit)
#     # training_step
#     train_step_mps = train_step_mps_maker(model=model,loss_function=loss_function,optimizer=optimizer)
#     # training_cycle
#     loss_list = []
#     q_grad_list = []
#     # トレーニングの実行
#     for x_batch in tqdm(train_dataset):
#         loss, q_grad = train_step_mps(x_batch)
#         loss_list.append(loss)
#         q_grad_list.append(tf.reduce_max(tf.abs(q_grad)))
#     return loss_list, q_grad_list, model, cirq_circuit, pauli_sums, parameter_symbols, qubits


# 自前で ciracuit を用意するタイプ
def simulate_with_circuit(
        number_of_circuit_parameters,
        encoder_layers,
        decoder_layers,
        latent_dim,
        epoch_num,
        batch_size,
        circuit
):
    """
    Args:
        circuit : 形状が [batch_size, n_params] の tf.Tensor を parameter として引数に取り，測定結果の期待値の値を返す回路関数
    """
    input_dim = number_of_circuit_parameters
    output_dim = number_of_circuit_parameters
        ### 古典の準備
    # モデルのインスタンス化
    model = VGON(
        input_dim,
        output_dim,
        encoder_layers,
        decoder_layers,
        latent_dim
    )

    # 最適化アルゴリズムの選択
    optimizer = tf.optimizers.Adam()

    # ランダム input の準備
    def random_input(length,input_dimension):
        return tf.random.uniform(shape=(length,input_dimension))

    # トレーニングデータの準備
    x_train = random_input(epoch_num * batch_size, input_dim)  # トレーニングデータをここにロード
    # データセットのバッチ化
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)

    # loss_function
    loss_function = loss_function_maker(circuit)

    # training_step
    train_step = train_step_maker(model=model,loss_function=loss_function,optimizer=optimizer)

    # training_cycle
    loss_list = []
    q_grad_list = []

    # トレーニングの実行
    for x_batch in tqdm(train_dataset,disable=False,desc="Training"):
        loss, q_grad = train_step(x_batch)
        loss_list.append(loss)
        q_grad_list.append(tf.reduce_max(tf.abs(q_grad)))

    return loss_list, q_grad_list, model

# 自前で ciracuit を用意するタイプ
def train_model(
        model,
        input_dim,
        epoch_num,
        batch_size,
        circuit
):
    """
    Args:
        circuit : 形状が [batch_size, n_params] の tf.Tensor を parameter として引数に取り，測定結果の期待値の値を返す回路関数
    """
        ### 古典の準備

    # 最適化アルゴリズムの選択
    optimizer = tf.optimizers.Adam()

    # ランダム input の準備
    def random_input(length,input_dimension):
        return tf.random.uniform(shape=(length,input_dimension))

    # トレーニングデータの準備
    x_train = random_input(epoch_num * batch_size, input_dim)  # トレーニングデータをここにロード
    # データセットのバッチ化
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)

    # loss_function
    loss_function = loss_function_maker(circuit)

    # training_step
    train_step = train_step_maker(model=model,loss_function=loss_function,optimizer=optimizer)

    # training_cycle
    loss_list = []
    q_grad_list = []

    # トレーニングの実行
    for x_batch in tqdm(train_dataset,disable=False,desc="Training"):
        loss, q_grad = train_step(x_batch)
        loss_list.append(loss)
        q_grad_list.append(tf.reduce_max(tf.abs(q_grad)))

    return loss_list, q_grad_list, model

# 複数回路
def multi_circuit_learning(
        model:VGON,
        input_dim:int,
        epoch_num:int,
        batch_size:int,
        circuits:list[Circuit]
)->tuple[list[float],list[float],VGON]:
    """
    Args:
        circuit (list[Circuit]): 形状が [batch_size, n_params] の tf.Tensor を parameter として引数に取り，測定結果の期待値の値を返す回路関数
    Returns:
        tuple[list[float],list[float],VGON]: loss list, quantum circuit gradient list, model の組
    """
    ### 古典の準備
    # 最適化アルゴリズムの選択
    optimizer = tf.optimizers.Adam()

    # ランダム input の準備
    def random_input(length,input_dimension):
        return tf.random.uniform(shape=(length,input_dimension))

    # トレーニングデータの準備
    x_train = random_input(epoch_num * batch_size, input_dim)  # トレーニングデータをここにロード
    # データセットのバッチ化
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)

    # loss_function
    loss_function = multi_circuit_loss_function_maker(circuits)

    # training_step
    train_step = train_step_maker(model=model,loss_function=loss_function,optimizer=optimizer)

    # training_cycle
    loss_list = []
    q_grad_list = []

    # トレーニングの実行
    for x_batch in tqdm(train_dataset,disable=False,desc="Training"):
        loss, q_grad = train_step(x_batch)
        loss_list.append(loss)
        q_grad_list.append(tf.reduce_max(tf.abs(q_grad)))

    return loss_list, q_grad_list, model