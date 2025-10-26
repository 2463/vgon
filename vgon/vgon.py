import tensorflow as tf

# エンコーダネットワークの定義
class Encoder(tf.keras.layers.Layer):
    def __init__(self,input_dim,neuron_num_list,latent_dim):
        super().__init__()
        # エンコーダの層を定義
        self.dense_layers = [tf.keras.layers.Dense(neuron_num_list[0], activation='relu',input_dim = input_dim)]
        for neuron_num in neuron_num_list[1:]:
            self.dense_layers.append(tf.keras.layers.Dense(neuron_num, activation='relu'))
        self.mu = tf.keras.layers.Dense(latent_dim)
        self.log_sigma = tf.keras.layers.Dense(latent_dim)

    def call(self, input):
        x = input
        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        mu = self.mu(x)
        sigma = tf.exp(self.log_sigma(x))
        return mu, sigma

# デコーダネットワークの定義
class Decoder(tf.keras.layers.Layer):
    def __init__(self,output_dimension,neuron_num_list,latent_dim):
        super().__init__()
        # デコーダの層を定義
        self.dense_layers = [tf.keras.layers.Dense(neuron_num_list[0], activation='relu',input_dim = latent_dim)]
        for neuron_num in neuron_num_list[1:]:
            self.dense_layers.append(tf.keras.layers.Dense(neuron_num, activation='relu'))

        self.dense_layers = [tf.keras.layers.Dense(neuron_num, activation='relu') for neuron_num in neuron_num_list]
        self.out = tf.keras.layers.Dense(output_dimension, activation='sigmoid')

    def call(self, input):
        z = input
        for dense_layer in self.dense_layers:
            z = dense_layer(z)
        return self.out(z)

# VAEモデルの定義
class VGON(tf.keras.Model):
    def __init__(self, input_dim,output_dim,encoder_layers, decoder_layers, latent_dim):
        super(VGON, self).__init__()
        self.encoder = Encoder(input_dim,encoder_layers,latent_dim)
        self.decoder = Decoder(output_dim,decoder_layers,latent_dim)

    def call(self, x):
        mu, sigma = self.encoder(x)
        epsilon = tf.random.normal(tf.shape(mu))
        z = mu + sigma * epsilon
        output_of_network = self.decoder(z)
        return output_of_network, mu, sigma