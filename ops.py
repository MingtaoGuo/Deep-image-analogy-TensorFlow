import tensorflow as tf
import tensorflow.contrib as contrib
from PatchMatch import *


def conv(inputs, w, b, strides=1):
    return tf.nn.conv2d(inputs, w, [1, strides, strides, 1], "SAME") + b

def max_pooling(inputs, ksize=2, strides=2):
    return tf.nn.max_pool(inputs, [1, ksize, ksize, 1], [1, strides, strides, 1], "SAME")

def relu(inputs):
    return tf.nn.relu(inputs)

def vggnet(inputs):
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])
    para = np.load("./vgg19//vgg19.npy", encoding="latin1").item()
    F = {}
    inputs = relu(conv(inputs, para["conv1_1"][0], para["conv1_1"][1]))
    F["conv1_1"] = inputs
    inputs = relu(conv(inputs, para["conv1_2"][0], para["conv1_2"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv2_1"][0], para["conv2_1"][1]))
    F["conv2_1"] = inputs
    inputs = relu(conv(inputs, para["conv2_2"][0], para["conv2_2"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv3_1"][0], para["conv3_1"][1]))
    F["conv3_1"] = inputs
    inputs = relu(conv(inputs, para["conv3_2"][0], para["conv3_2"][1]))
    inputs = relu(conv(inputs, para["conv3_3"][0], para["conv3_3"][1]))
    inputs = relu(conv(inputs, para["conv3_4"][0], para["conv3_4"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv4_1"][0], para["conv4_1"][1]))
    F["conv4_1"] = inputs
    inputs = relu(conv(inputs, para["conv4_2"][0], para["conv4_2"][1]))
    inputs = relu(conv(inputs, para["conv4_3"][0], para["conv4_3"][1]))
    inputs = relu(conv(inputs, para["conv4_4"][0], para["conv4_4"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv(inputs, para["conv5_1"][0], para["conv5_1"][1]))
    F["conv5_1"] = inputs
    return F

def vgg_block(layer, inputs):
    para = np.load("./vgg19//vgg19.npy", encoding="latin1").item()
    if layer == 1:
        inputs = relu(conv(inputs, para["conv1_1"][0], para["conv1_1"][1]))
        inputs_L = inputs * 1.0
        inputs = relu(conv(inputs, para["conv1_2"][0], para["conv1_2"][1]))
        inputs = max_pooling(inputs)
        inputs = relu(conv(inputs, para["conv2_1"][0], para["conv2_1"][1]))
        return inputs, inputs_L
    if layer == 2:
        inputs = relu(conv(inputs, para["conv2_1"][0], para["conv2_1"][1]))
        inputs_L = inputs * 1.0
        inputs = relu(conv(inputs, para["conv2_2"][0], para["conv2_2"][1]))
        inputs = max_pooling(inputs)
        inputs = relu(conv(inputs, para["conv3_1"][0], para["conv3_1"][1]))
        return inputs, inputs_L
    if layer == 3:
        inputs = relu(conv(inputs, para["conv3_1"][0], para["conv3_1"][1]))
        inputs_L = inputs * 1.0
        inputs = relu(conv(inputs, para["conv3_2"][0], para["conv3_2"][1]))
        inputs = relu(conv(inputs, para["conv3_3"][0], para["conv3_3"][1]))
        inputs = relu(conv(inputs, para["conv3_4"][0], para["conv3_4"][1]))
        inputs = max_pooling(inputs)
        inputs = relu(conv(inputs, para["conv4_1"][0], para["conv4_1"][1]))
        return inputs, inputs_L
    if layer == 4:
        inputs = relu(conv(inputs, para["conv4_1"][0], para["conv4_1"][1]))
        inputs_L = inputs * 1.0
        inputs = relu(conv(inputs, para["conv4_2"][0], para["conv4_2"][1]))
        inputs = relu(conv(inputs, para["conv4_3"][0], para["conv4_3"][1]))
        inputs = relu(conv(inputs, para["conv4_4"][0], para["conv4_4"][1]))
        inputs = max_pooling(inputs)
        inputs = relu(conv(inputs, para["conv5_1"][0], para["conv5_1"][1]))
        return inputs, inputs_L


def preprocessing(A, B_prime):
    F_A = vggnet(A)
    F_B_prime = vggnet(B_prime)
    return F_A, F_B_prime

def Deconvolve(sess, name, layer, warped_F_B_prime_L):
    H = np.size(warped_F_B_prime_L, 0)
    W = np.size(warped_F_B_prime_L, 1)
    if layer == 1:
        C = 3
    elif layer == 2:
        C = 64
    elif layer == 3:
        C = 128
    else:
        C = 256
    var_name = name + "R_L_1"
    temp = tf.get_variable(var_name, [1, 2 * H, 2 * W, C], initializer=tf.truncated_normal_initializer(stddev=0.02))
    inputs, R_L_1 = vgg_block(layer, temp)
    Loss = tf.reduce_sum(tf.square(inputs[0, :, :, :] - warped_F_B_prime_L))
    sess.run(tf.global_variables_initializer())
    print("Before L-BFGS, Loss: %f" % (sess.run(Loss)))
    opt = contrib.opt.ScipyOptimizerInterface(Loss, method="L-BFGS-B", options={"maxiter": 50}, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_name))
    opt.minimize(sess)
    print("After L-BFGS, Loss: %f" % (sess.run(Loss)))
    R_L_1 = sess.run(R_L_1)
    return R_L_1


def get_W_L_1(F_A_L_1, alpha_L_1, k=300, tau=0.05):
    temp = np.sum(np.square(F_A_L_1), axis=-1, keepdims=True)
    temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
    W_L_1 = np.float32(temp > tau) * alpha_L_1
    # W_L_1 = alpha_L_1 / (1 + np.exp(-k * (np.square(F_A_L_1) - tau)))
    return W_L_1


