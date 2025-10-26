import numpy as np, tensorflow as tf

def train_step_maker(model,loss_function,optimizer):
    # トレーニングステップの定義
    @tf.function
    def train_step(x):
        with tf.GradientTape(persistent=True) as tape:
            output, mu, sigma = model(x)
            # バッチに対して目的関数を計算
            # ここで出力が 0~1 に正規化されているので， 2 * np.pi をかける
            output = tf.cast(output,dtype=tf.float64)
            loss = loss_function(output * 2 * np.pi)
            # print("\n##model(x)##\n{}".format(output))
            # print("\n##loss##\n{}".format(loss))
            # print("\n##model.trainable_variables##\n{}\n".format(model.trainable_variables))
        q_grad = tape.gradient(loss, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, q_grad
    
    return train_step