from ops import *
import scipy.misc as misc
from PIL import Image


def DeepImageAnalogy(img, ref, alpha_L, patch_size, search_radius, itr):
    #Preprocessing
    A = tf.placeholder(tf.float32, [None, None, None, 3])
    B_prime = tf.placeholder(tf.float32, [None, None, None, 3])
    F_A, F_B_prime = preprocessing(A, B_prime)
    sess = tf.Session()
    [F_A, F_B_prime] = sess.run([F_A, F_B_prime], feed_dict={A: img, B_prime: ref})
    F_A_L = F_A["conv5_1"]
    F_A_prime_L = F_A_L * 1.
    F_B_prime_L = F_B_prime["conv5_1"]
    F_B_L = F_B_prime_L * 1.
    #Iteration
    layers = [ "conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    phi_a2b_L = None
    phi_b2a_L = None
    for layer in range(layers.__len__()-1, -1, -1):
        print("Layer: conv%d_1" % (layer + 1))
        #NNF search
        if layer == 4:
            phi_a2b_L, nnd_a2b_L = initialise_nnf(F_A_L[0, :, :, :], F_A_prime_L[0, :, :, :], F_B_L[0, :, :, :], F_B_prime_L[0, :, :, :], phi_a2b_L, patch_size[layer], is_init=False)
            phi_b2a_L, nnd_b2a_L = initialise_nnf(F_B_L[0, :, :, :], F_B_prime_L[0, :, :, :], F_A_L[0, :, :, :], F_A_prime_L[0, :, :, :], phi_b2a_L, patch_size[layer], is_init=False)
            phi_a2b_L, nnd_a2b_L = propagate(F_A_L[0, :, :, :], F_A_prime_L[0, :, :, :], F_B_L[0, :, :, :], F_B_prime_L[0, :, :, :], phi_a2b_L, nnd_a2b_L, iters=itr, rand_search_radius=search_radius[layer], patch_size=patch_size[layer])
            phi_b2a_L, nnd_b2a_L = propagate(F_B_L[0, :, :, :], F_B_prime_L[0, :, :, :], F_A_L[0, :, :, :], F_A_prime_L[0, :, :, :], phi_b2a_L, nnd_b2a_L, iters=itr, rand_search_radius=search_radius[layer], patch_size=patch_size[layer])
        else:
            _, nnd_a2b_L = initialise_nnf(F_A_L[0, :, :, :], F_A_prime_L[0, :, :, :], F_B_L[0, :, :, :], F_B_prime_L[0, :, :, :], phi_a2b_L, patch_size[layer], is_init=True)
            _, nnd_b2a_L = initialise_nnf(F_B_L[0, :, :, :], F_B_prime_L[0, :, :, :], F_A_L[0, :, :, :], F_A_prime_L[0, :, :, :], phi_b2a_L, patch_size[layer], is_init=True)
            phi_a2b_L, nnd_a2b_L = propagate(F_A_L[0, :, :, :], F_A_prime_L[0, :, :, :], F_B_L[0, :, :, :], F_B_prime_L[0, :, :, :], phi_a2b_L, nnd_a2b_L, iters=itr, rand_search_radius=search_radius[layer], patch_size=patch_size[layer])
            phi_b2a_L, nnd_b2a_L = propagate(F_B_L[0, :, :, :], F_B_prime_L[0, :, :, :], F_A_L[0, :, :, :], F_A_prime_L[0, :, :, :], phi_b2a_L, nnd_b2a_L, iters=itr, rand_search_radius=search_radius[layer], patch_size=patch_size[layer])
        if layer > 0:
            #Reconstruction
            Warped_F_B_prime_L = reconstruct_image(F_B_prime_L[0, :, :, :], phi_a2b_L)
            R_L_1 = Deconvolve(sess, str(layer) + "a", layer, Warped_F_B_prime_L)
            W_L_1 = get_W_L_1(F_A[layers[layer - 1]], alpha_L[layer-1])
            F_A_prime_L_1 = F_A[layers[layer - 1]] * W_L_1 + R_L_1 * (1 - W_L_1)
            Warped_F_A_L = reconstruct_image(F_A_L[0, :, :, :], phi_b2a_L)
            R_L_1 = Deconvolve(sess, str(layer) + "b", layer, Warped_F_A_L)
            W_L_1 = get_W_L_1(F_B_prime[layers[layer - 1]], alpha_L[layer - 1])
            F_B_L_1 = F_B_prime[layers[layer - 1]] * W_L_1 + R_L_1 * (1 - W_L_1)
            F_A_prime_L = F_A_prime_L_1
            F_B_L = F_B_L_1
            F_A_L = F_A[layers[layer-1]]
            F_B_prime_L = F_B_prime[layers[layer-1]]
            #NNF upsampling
            phi_a2b_L = upsample_nnf(phi_a2b_L)
            phi_b2a_L = upsample_nnf(phi_b2a_L)
    return phi_a2b_L, phi_b2a_L


if __name__ == "__main__":
    A = misc.imresize(np.array(Image.open("./IMAGES/girl_A.jpg")), [224, 224])[np.newaxis, :, :, :]
    B_prime = misc.imresize(np.array(Image.open("./IMAGES/girl_B_prime.jpg")), [224, 224])[np.newaxis, :, :, :]
    # B_prime = np.dstack((B_prime, B_prime, B_prime))[np.newaxis, :, :, :]
    alpha_L = [0.1, 0.6, 0.7, 0.8]  # start: [0.1, 0.6, 0.7, 0.8] +-2
    patch_size = [5, 5, 3, 3, 3]
    search_radius = [4, 4, 6, 6, 6]
    itrs = 5
    phi_a2b, phi_b2a = DeepImageAnalogy(A, B_prime, alpha_L, patch_size, search_radius, itrs)
    Image.fromarray(np.uint8(reconstruct_image(B_prime[0, :, :, :], phi_a2b))).show()#save('./a2b.jpg')
    Image.fromarray(np.uint8(reconstruct_image(A[0, :, :, :], phi_b2a))).show()#save('./b2a.jpg')
    Image.fromarray(np.uint8(reconstruct_image(B_prime[0, :, :, :], phi_a2b))).save('./a2b.jpg')
    Image.fromarray(np.uint8(reconstruct_image(A[0, :, :, :], phi_b2a))).save('./b2a.jpg')
    Image.fromarray(np.uint8(A[0, :, :, :])).save('./A.jpg')
    Image.fromarray(np.uint8(B_prime[0, :, :, :])).save('./B_prime.jpg')
