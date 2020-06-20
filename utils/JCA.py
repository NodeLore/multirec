"""
Ziwei Zhu
Computer Science and Engineering Department, Texas A&M University
zhuziwei@tamu.edu
""" 


class JCA:

    def __init__(self, sess, args, train_R, vali_R, metric_path, date, data_name,
                 result_path=None):

        if args.f_act == "Sigmoid":
            f_act = tf.nn.sigmoid
        elif args.f_act == "Relu":
            f_act = tf.nn.relu
        elif args.f_act == "Tanh":
            f_act = tf.nn.tanh
        elif args.f_act == "Identity":
            f_act = tf.identity
        elif args.f_act == "Elu":
            f_act = tf.nn.elu
        else:
            raise NotImplementedError("ERROR")

        if args.g_act == "Sigmoid":
            g_act = tf.nn.sigmoid
        elif args.g_act == "Relu":
            g_act = tf.nn.relu
        elif args.g_act == "Tanh":
            g_act = tf.nn.tanh
        elif args.g_act == "Identity":
            g_act = tf.identity
        elif args.g_act == "Elu":
            g_act = tf.nn.elu
        else:
            raise NotImplementedError("ERROR")

        self.sess = sess
        self.args = args

        self.base = args.base

        self.num_rows = train_R.shape[0]
        self.num_cols = train_R.shape[1]
        self.U_hidden_neuron = args.U_hidden_neuron
        self.I_hidden_neuron = args.I_hidden_neuron

        self.train_R = train_R
        self.vali_R = vali_R
        self.num_test_ratings = np.sum(vali_R)

        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch_U = int(self.num_rows / float(self.batch_size)) + 1
        self.num_batch_I = int(self.num_cols / float(self.batch_size)) + 1

        self.lr = args.lr  # learning rate
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step
        self.margin = args.margin

        self.f_act = f_act  # the activation function for the output layer
        self.g_act = g_act  # the activation function for the hidden layer

        self.global_step = tf.Variable(0, trainable=False)

        self.lambda_value = args.lambda_value  # regularization term trade-off

        self.result_path = result_path
        self.metric_path = metric_path
        self.date = date  # today's date
        self.data_name = data_name

        self.neg_sample_rate = args.neg_sample_rate
        self.U_OH_mat = np.eye(self.num_rows, dtype=float)
        self.I_OH_mat = np.eye(self.num_cols, dtype=float)

        print('**********JCA**********')
        self.prepare_model()

    def run(self, train_R, vali_R):
        self.train_R = train_R
        self.vali_R = vali_R
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.train_epoch):
            self.train_model(epoch_itr)
            if epoch_itr % 1 == 0:
                self.test_model(epoch_itr)
        tf.train.Saver().save(self.sess, 'model/model.ckpt')
        print('Save the training process to model folder')
        return self.make_records()

    def prepare_model(self):

        # input rating vector
        self.input_R_U = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_cols], name="input_R_U")
        self.input_R_I = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_rows, None], name="input_R_I")
        self.input_OH_I = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_cols], name="input_OH_I")
        self.input_P_cor = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 2], name="input_P_cor")
        self.input_N_cor = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 2], name="input_N_cor")

        # input indicator vector indicator
        self.row_idx = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 1], name="row_idx")
        self.col_idx = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 1], name="col_idx")

        # user component
        # first layer weights
        UV = tf.get_variable(name="UV", initializer=tf.truncated_normal(shape=[self.num_cols, self.U_hidden_neuron],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # second layer weights
        UW = tf.get_variable(name="UW", initializer=tf.truncated_normal(shape=[self.U_hidden_neuron, self.num_cols],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # first layer bias
        Ub1 = tf.get_variable(name="Ub1", initializer=tf.truncated_normal(shape=[1, self.U_hidden_neuron],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # second layer bias
        Ub2 = tf.get_variable(name="Ub2", initializer=tf.truncated_normal(shape=[1, self.num_cols],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # item component
        # first layer weights
        IV = tf.get_variable(name="IV", initializer=tf.truncated_normal(shape=[self.num_rows, self.I_hidden_neuron],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # second layer weights
        IW = tf.get_variable(name="IW", initializer=tf.truncated_normal(shape=[self.I_hidden_neuron, self.num_rows],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # first layer bias
        Ib1 = tf.get_variable(name="Ib1", initializer=tf.truncated_normal(shape=[1, self.I_hidden_neuron],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        # second layer bias
        Ib2 = tf.get_variable(name="Ib2", initializer=tf.truncated_normal(shape=[1, self.num_rows],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)


        I_factor_vector = tf.get_variable(name="I_factor_vector", initializer=tf.random_uniform(shape=[1, self.num_cols]),
                                          dtype=tf.float32)

        # user component
        U_pre_Encoder = tf.matmul(self.input_R_U, UV) + Ub1  # input to the hidden layer
        self.U_Encoder = self.g_act(U_pre_Encoder)  # output of the hidden layer
        U_pre_Decoder = tf.matmul(self.U_Encoder, UW) + Ub2  # input to the output layer
        self.U_Decoder = self.f_act(U_pre_Decoder)  # output of the output layer

        # item component
        I_pre_mul = tf.transpose(tf.matmul(I_factor_vector, tf.transpose(self.input_OH_I)))
        I_pre_Encoder = tf.matmul(tf.transpose(self.input_R_I), IV) + Ib1  # input to the hidden layer
        self.I_Encoder = self.g_act(I_pre_Encoder * I_pre_mul)  # output of the hidden layer
        I_pre_Decoder = tf.matmul(self.I_Encoder, IW) + Ib2  # input to the output layer
        self.I_Decoder = self.f_act(I_pre_Decoder)  # output of the output layer

        # final output
        self.Decoder = ((tf.transpose(tf.gather_nd(tf.transpose(self.U_Decoder), self.col_idx)))
                        + tf.gather_nd(tf.transpose(self.I_Decoder), self.row_idx)) / 2.0

        tf.add_to_collection('network_output', self.Decoder)

        pos_data = tf.gather_nd(self.Decoder, self.input_P_cor)
        neg_data = tf.gather_nd(self.Decoder, self.input_N_cor)

        pre_cost1 = tf.maximum(neg_data - pos_data + self.margin,
                               tf.zeros(tf.shape(neg_data)[0]))
        cost1 = tf.reduce_sum(pre_cost1)  # prediction squared error
        pre_cost2 = tf.square(self.l2_norm(UW)) + tf.square(self.l2_norm(UV)) \
                    + tf.square(self.l2_norm(IW)) + tf.square(self.l2_norm(IV))\
                    + tf.square(self.l2_norm(Ib1)) + tf.square(self.l2_norm(Ib2))\
                    + tf.square(self.l2_norm(Ub1)) + tf.square(self.l2_norm(Ub2))
        cost2 = self.lambda_value * 0.5 * pre_cost2  # regularization term

        self.cost = cost1 + cost2  # the loss function
        tf.add_to_collection('network_cost', self.cost)
        
        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
        else:
            raise ValueError("Optimizer Key ERROR")

        gvs = optimizer.compute_gradients(self.cost)
        self.optimizer = optimizer.apply_gradients(gvs, global_step=self.global_step)

    def restoreModel(self, path):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.import_meta_graph(path + '.meta')
        saver.restore(self.sess, path)
        self.cost = tf.get_collection('network_cost')[0]
        self.Decoder = tf.get_collection('network_output')[0]

    def predict(self, testR):  # calculate the cost and rmse of testing set in each epoch
        start_time = time.time()

        _, Decoder = self.sess.run([self.cost, self.Decoder],
                                   feed_dict={
                                        self.input_R_U: testR,
                                        self.input_R_I: testR,
                                        self.input_OH_I: self.I_OH_mat,
                                        self.input_P_cor: [[0, 0]],
                                        self.input_N_cor: [[0, 0]],
                                        self.row_idx: np.reshape(range(self.num_rows), (self.num_rows, 1)),
                                        self.col_idx: np.reshape(range(self.num_cols), (self.num_cols, 1))})
        
        #np.save('m.npy', Decoder)
        return Decoder[0]
        



    @staticmethod
    def l2_norm(tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))
