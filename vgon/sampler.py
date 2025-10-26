import tensorflow as tf, numpy as np
from tqdm import tqdm

# ランダム input の準備
def random_input(length,input_dimension):
    return tf.random.uniform(shape=(length,input_dimension))

# sample_number = 1000

def mps_sampling(model,loss_function,input_dim,sample_number):
    # サンプル用のランダムを用意
    x_train = random_input(sample_number, input_dim)

    samplied_list = []
    output_list = []

    for sample in tqdm(x_train):
        output, _, _ = model(tf.expand_dims(sample,axis=0))
        loss = loss_function(output * 2 * np.pi)
        if loss < -1.0000:
            continue
        output_list.append(output)
        samplied_list.append(loss)

    minimum = min(samplied_list)
    minimum_parameter = output_list[samplied_list.index(minimum)]

    print(f"minimum value : {minimum}\nparameter :\n{minimum_parameter}")