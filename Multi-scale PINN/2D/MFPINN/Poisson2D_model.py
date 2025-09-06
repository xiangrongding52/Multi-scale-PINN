# import tensorflow as tf
import os.path

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import timeit


class Sampler:  # https://blog.csdn.net/aiwanghuan5017/article/details/102147825
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):  # 初始化
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):  # 样本的定义
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y


class Poisson2D:
    def __init__(self, dir,layers, operator, bcs_sampler, res_sampler, u_sampler, model, stiff_ratio,u_star,x_star):
        # Normalization constants
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir = dir
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x1, self.sigma_x1 = self.mu_X[0], self.sigma_X[0]
        self.mu_x2, self.sigma_x2 = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler
        self.u_sampler = u_sampler
        self.u_star = u_star
        self.x_star = x_star
        # Helmoholtz constant
        

        # Mode
        self.model = model

        # Record stiff ratio
        self.stiff_ratio = stiff_ratio

        # Adaptive constant
        self.beta = 0.9
        self.adaptive_constant_val = np.array(1.0)  # 自适应常数变量

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        if model in ['M4']:
            # Initialize encoder weights and biases
            self.encoder_weights_1 = self.xavier_init([2*layers[0], layers[1]])
            self.encoder_biases_1 = self.xavier_init([1, layers[1]])

            self.encoder_weights_2 = self.xavier_init([2*layers[0], layers[1]])
            self.encoder_biases_2 = self.xavier_init([1, layers[1]])

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph，tensorflow的基本组件
        self.x1_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc3_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_bc4_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_bc4_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc4_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x1_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Define placeholder for adaptive constant
        self.adaptive_constant_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_val.shape)

        # Evaluate predictions
        self.u_bc1_pred = self.net_u(self.x1_bc1_tf, self.x2_bc1_tf)
        self.u_bc2_pred = self.net_u(self.x1_bc2_tf, self.x2_bc2_tf)
        self.u_bc3_pred = self.net_u(self.x1_bc3_tf, self.x2_bc3_tf)
        self.u_bc4_pred = self.net_u(self.x1_bc4_tf, self.x2_bc4_tf)

        self.u_pred = self.net_u(self.x1_u_tf, self.x2_u_tf)
        self.r_pred = self.net_r(self.x1_r_tf, self.x2_r_tf)

        # Boundary loss
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_tf - self.u_bc1_pred))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_tf - self.u_bc2_pred))
        self.loss_bc3 = tf.reduce_mean(tf.square(self.u_bc3_tf - self.u_bc3_pred))
        self.loss_bc4 = tf.reduce_mean(tf.square(self.u_bc4_tf - self.u_bc4_pred))
        self.loss_bcs = self.adaptive_constant_tf * (self.loss_bc1 + self.loss_bc2 + self.loss_bc3 + self.loss_bc4)

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_tf - self.r_pred))

        # Total loss
        self.loss = self.loss_res + self.loss_bcs

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Logger
        self.loss_bcs_log = []
        self.loss_res_log = []
        self.loss_int_log = []
        self.loss_log = []
        self.err_l2r_log = []
        self.step_log = []
        self.saver = tf.train.Saver()

        # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict(self.layers)
        self.dict_gradients_bcs_layers = self.generate_grad_dict(self.layers)

        # Gradients Storage
        self.grad_res = []
        self.grad_bcs = []
        for i in range(len(self.layers) - 1):
            self.grad_res.append(tf.gradients(self.loss_res, self.weights[i])[0])
            self.grad_bcs.append(tf.gradients(self.loss_bcs, self.weights[i])[0])

        # Compute and store the adaptive constant
        self.adpative_constant_log = []
        self.adaptive_constant_list = []

        self.max_grad_res_list = []
        self.mean_grad_res_list = []
        self.mean_grad_bcs_list = []

        for i in range(len(self.layers) - 1):
            self.max_grad_res_list.append(tf.reduce_max(tf.abs(self.grad_res[i])))
            self.mean_grad_bcs_list.append(tf.reduce_mean(tf.abs(self.grad_bcs[i])))

        self.max_grad_res = tf.reduce_max(tf.stack(self.max_grad_res_list))
        self.mean_grad_bcs = tf.reduce_mean(tf.stack(self.mean_grad_bcs_list))
        self.adaptive_constant = self.max_grad_res / self.mean_grad_bcs

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def relu2(self,x):
        return tf.nn.relu(x) ** 2

    def relu3(self,x):
        return tf.nn.relu(x) ** 3

    def srelu(self,x):
        return tf.nn.relu(1 - x) * tf.nn.relu(x)
    # Create dictionary to store gradients
    def generate_grad_dict(self, layers):
        num = len(layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict

    # Save gradients
    def save_gradients(self, tf_dict):
        num_layers = len(self.layers)
        for i in range(num_layers - 1):
            grad_res_value, grad_bcs_value = self.sess.run([self.grad_res[i], self.grad_bcs[i]], feed_dict=tf_dict)

            # save gradients of loss_res and loss_bcs
            self.dict_gradients_res_layers['layer_' + str(i + 1)].append(grad_res_value.flatten())
            self.dict_gradients_bcs_layers['layer_' + str(i + 1)].append(grad_bcs_value.flatten())
        return None

    # Compute the Hessian
    def flatten(self, vectors):
        return tf.concat([tf.reshape(v, [-1]) for v in vectors], axis=0)

    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)
    
    def initialize_NN(self, layers):
        if self.model in ['M1']:
            weights = []
            biases = []
            num_layers = len(layers)
            for l in range(0, num_layers - 1):
                W = self.xavier_init(size=[layers[l], layers[l + 1]])
                b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
                weights.append(W)
                biases.append(b)
            return weights, biases
        
        if self.model in ['M2','M3','M4']:
            in_size = layers[0]
            hidden_layers = layers[1:-1]  # 提取中间部分作为隐藏层
            out_size = layers[-1]
            n_hiddens = len(hidden_layers)
            weights = []
            biases = []

            # 初始化输入层的权重和偏置
            W = self.xavier_init(size=[in_size, hidden_layers[0]])
            B = tf.Variable(tf.zeros([1, hidden_layers[0]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(B)

            # 初始化隐藏层
            for i_layer in range(n_hiddens - 1):
                if i_layer == 0:
                    W = self.xavier_init(size=[hidden_layers[i_layer] * 2, hidden_layers[i_layer + 1]])
                    B = tf.Variable(tf.zeros([1, hidden_layers[i_layer + 1]], dtype=tf.float32), dtype=tf.float32)
                else:
                    W = self.xavier_init(size=[hidden_layers[i_layer], hidden_layers[i_layer + 1]])
                    B = tf.Variable(tf.zeros([1, hidden_layers[i_layer + 1]], dtype=tf.float32), dtype=tf.float32)
                weights.append(W)
                biases.append(B)

            # 初始化输出层的权重和偏置
            W = self.xavier_init(size=[hidden_layers[-1], out_size])
            B = tf.Variable(tf.zeros([1, out_size], dtype=tf.float32), dtype=tf.float32)  # 修正为 out_size
            weights.append(W)
            biases.append(B)
            return weights, biases
    
        return weights, biases

    def forward_pass(self, H):  # 接下来的神经网络为典型的基于物理信息的神经网络


        if self.model in ['M1']:
            num_layers = len(self.layers)
            for l in range(0, num_layers - 2):
                W = self.weights[l]
                b = self.biases[l]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
            W = self.weights[-1]
            b = self.biases[-1]
            H = tf.add(tf.matmul(H, W), b)
            return H

        if self.model in ['M2']:
            freq_frag = [20, 21, 22, 23, 24, 26, 27, 28, 29, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25]
            #freq_frag =  np.arange(1, 121)
            layers=self.layers
            hiddens = layers[1:-1] 
            num_layers = len(hiddens)+1
            Unit_num = int(self.layers[1] / len(freq_frag))
            mixcoe = np.repeat(freq_frag, Unit_num)
            mixcoe = np.concatenate((mixcoe, np.ones([self.layers[1] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
            mixcoe = mixcoe.astype(np.float32)
            W = self.weights[0]
            B = self.biases[0]
            H = tf.matmul(H, W) * mixcoe
            H = tf.concat([tf.cos(H), tf.sin(H)], axis=-1)
            hiddens_record = hiddens[0]
            for k in range(num_layers-2):
                H_pre = H
                W = self.weights[k+1]
                b = self.biases[k+1]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
                if (hiddens[k+1] == hiddens_record) and (k != 0):
                    H = H + H_pre
                hiddens_record = hiddens[k+1]

            W_out= self.weights[-1]
            b_out = self.biases[-1]
            H = tf.add(tf.matmul(H, W_out), b_out)
            return H
        if self.model in ['M3']:
            #freq_frag = [20, 21, 22, 23, 24, 26, 27, 28, 29, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25]
            freq_frag =  np.arange(1, 121)
            layers=self.layers
            hiddens = layers[1:-1] 
            num_layers = len(hiddens)+1
            Unit_num = int(self.layers[1] / len(freq_frag))
            mixcoe = np.repeat(freq_frag, Unit_num)
            mixcoe = np.concatenate((mixcoe, np.ones([self.layers[1] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
            mixcoe = mixcoe.astype(np.float32)
            W = self.weights[0]
            B = self.biases[0]
            H = tf.matmul(H, W) * mixcoe
            H = tf.concat([tf.cos(H), tf.sin(H)], axis=-1)
            hiddens_record = hiddens[0]
            for k in range(num_layers-2):
                H_pre = H
                W = self.weights[k+1]
                b = self.biases[k+1]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
                if (hiddens[k+1] == hiddens_record) and (k != 0):
                    H = H + H_pre
                hiddens_record = hiddens[k+1]

            W_out= self.weights[-1]
            b_out = self.biases[-1]
            H = tf.add(tf.matmul(H, W_out), b_out)
            return H
        if self.model in [ 'M4']:

            freq_frag =  np.arange(1, 121)
            layers=self.layers
            hiddens = layers[1:-1] 
            num_layers = len(hiddens)+1
            Unit_num = int(self.layers[1] / len(freq_frag))
            mixcoe = np.repeat(freq_frag, Unit_num)
            mixcoe = np.concatenate((mixcoe, np.ones([self.layers[1] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
            mixcoe = mixcoe.astype(np.float32)
            W = self.weights[0]
            B = self.biases[0]
            H = tf.matmul(H, W) * mixcoe
            H = tf.concat([tf.cos(H), tf.sin(H)], axis=-1)
            
            encoder_1 = tf.tanh(tf.add(tf.matmul(H, self.encoder_weights_1), self.encoder_biases_1))
            encoder_2 = tf.tanh(tf.add(tf.matmul(H, self.encoder_weights_2), self.encoder_biases_2))

            for l in range(0, num_layers - 2):
                W = self.weights[l+1]
                b = self.biases[l+1]
                H = tf.math.multiply(tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_1) + \
                    tf.math.multiply(1 - tf.tanh(tf.add(tf.matmul(H, W), b)), encoder_2)

            W = self.weights[-1]
            b = self.biases[-1]
            H = tf.add(tf.matmul(H, W), b)
            return H

    def net_u(self, x1, x2):
        u = self.forward_pass(tf.concat([x1, x2], 1))
        return u

    # Forward pass for residual
    def net_r(self, x1, x2):
        u = self.net_u(x1, x2)
        residual = self.operator(u, x1, x2,
                                 self.sigma_x1,
                                 self.sigma_x2)
        return residual

    # Feed minibatch
    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X  # 使得其初始数据服从标准正态分布
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=1):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
            X_bc3_batch, u_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], batch_size)
            X_bc4_batch, u_bc4_batch = self.fetch_minibatch(self.bcs_sampler[3], batch_size)

            # Fetch residual mini-batch
            X_res_batch, f_res_batch = self.fetch_minibatch(self.res_sampler, batch_size)
            X_u_batch, u_batch = self.fetch_minibatch(self.u_sampler, batch_size)
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x1_bc1_tf: X_bc1_batch[:, 0:1], self.x2_bc1_tf: X_bc1_batch[:, 1:2],
                       self.u_bc1_tf: u_bc1_batch,
                       self.x1_bc2_tf: X_bc2_batch[:, 0:1], self.x2_bc2_tf: X_bc2_batch[:, 1:2],
                       self.u_bc2_tf: u_bc2_batch,
                       self.x1_bc3_tf: X_bc3_batch[:, 0:1], self.x2_bc3_tf: X_bc3_batch[:, 1:2],
                       self.u_bc3_tf: u_bc3_batch,
                       self.x1_bc4_tf: X_bc4_batch[:, 0:1], self.x2_bc4_tf: X_bc4_batch[:, 1:2],
                       self.u_bc4_tf: u_bc4_batch,
                       self.x1_u_tf: X_u_batch[:, 0:1], self.x2_u_tf: X_u_batch[:, 1:2],
                       self.u_tf: u_batch,
                       self.x1_r_tf: X_res_batch[:, 0:1], self.x2_r_tf: X_res_batch[:, 1:2], self.r_tf: f_res_batch,
                       self.adaptive_constant_tf: self.adaptive_constant_val
                       }

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Compute the eigenvalues of the Hessian of losses
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value, loss_res_value = self.sess.run([self.loss_bcs, self.loss_res], tf_dict)

                self.loss_bcs_log.append(loss_bcs_value / self.adaptive_constant_val)
                self.loss_res_log.append(loss_res_value)

                # Compute and Print adaptive weights during training
                if self.model in ['M2']:
                    adaptive_constant_value = self.sess.run(self.adaptive_constant, tf_dict)
                    self.adaptive_constant_val = adaptive_constant_value * (1.0 - self.beta) \
                                                 + self.beta * self.adaptive_constant_val

                self.adpative_constant_log.append(self.adaptive_constant_val)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_log.append(loss_value)
                self.step_log.append(it)
                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_res: %.3e, Adaptive_Constant: %.2f ,Time: %.2f' %
                      (it, loss_value, loss_bcs_value, loss_res_value, self.adaptive_constant_val, elapsed))
                start_time = timeit.default_timer()

                u_pred = self.predict_u(self.x_star)
                error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                self.err_l2r_log.append(error_u)
                print('Relative L2 error_u: {:.2e}'.format(error_u))
            # Store gradients
            if it % 10000 == 0:
                self.save_gradients(tf_dict)
                print("Gradients information stored ...")

                # Evaluates predictions at test points

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x1_u_tf: X_star[:, 0:1], self.x2_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x1_r_tf: X_star[:, 0:1], self.x2_r_tf: X_star[:, 1:2]}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star



