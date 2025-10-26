import numpy as np, tensorflow as tf

def create_partial_h(target,total,h):
    result = tf.one_hot(target,total,h,dtype=tf.float64)
    result = tf.expand_dims(result,axis=0)
    return result

# loss function を作成
def loss_function_maker(circuit):
    @tf.custom_gradient
    def loss_function(tensor):
        batch_size = tensor.shape[0]
        batch_tensors = tf.unstack(tensor)
        loss = 0.
        for batch_tensor in batch_tensors:
            # parameter_tensor = tf.expand_dims(batch_tensor, axis=0)
            result = circuit(batch_tensor)
            # loss += result[0,0] * result[0,1]
            loss += result[0,0]
        loss = loss / batch_size

        def grad(upstream):
            h = 1.0e-6  # 有限差分法のステップサイズ
            size = tensor.shape[1]
            grad = []
            for i in range(size):
                partial_h = create_partial_h(i,size,h)
                # print(f"##partial_h##\n{partial_h}\n")
                # tf.print("param_tensor")
                # tf.print(tensor)
                # tf.print("partial_h")
                # tf.print(partial_h)
                # tf.print("partial_h + param_tensor")
                # tf.print(tensor + partial_h)
                # tf.print("loss func")
                # tf.print(loss_function)
                # plus = loss_function(tensor + partial_h)
                # minus = loss_function(tensor - partial_h)
                # tf.print(plus)
                # tf.print(minus)
                partial_diff = (loss - loss_function(tensor - partial_h)) / h
                # tf.print("partial_diff")
                # tf.print(partial_diff)
                grad.append(partial_diff)
                # print(partial_diff)
            # print(grad)
            return tf.stack(grad * batch_size)

        return loss, grad        
    return loss_function

def multi_circuit_loss_function_maker(circuits: list['Circuit']):
    """
    Arg
        circuits: 同梱の Circuit class の list
    """
    def loss_function(tensor):
        # print("test in loss_func")
        loss = 0.
        for circuit in circuits:
            batched_result = tf.map_fn(circuit.circuit,tensor,fn_output_signature=tf.float64)
            result = tf.reduce_mean(batched_result)
            loss += result
        return loss

    return loss_function

class Circuit:
    def __init__(self,gsim,orders):
        self.gsim = gsim
        self.orders = orders
    
    # @tf.function
    def circuit(self,parameters:tf.Tensor):
        parameters = tf.cast(parameters,dtype=tf.float64) # type: ignore
        result = self.gsim.simulate(parameters,self.orders)
        return result
