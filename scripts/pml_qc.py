# hello there
import numpy as np
import os
from time import time
from glob import glob
import tensorflow as tf
#from IPython.core.debugger import set_trace

class PML_QC(object):

    def build_model(self):

        nconditions = 1
        self.conditions_placeholder = tf.placeholder( tf.float32, shape = [None, FLAGS.Ngrid1, FLAGS.Ngrid2, FLAGS.Ngrid3, nconditions], name='conditions')
        self.deltarhos_placeholder = tf.placeholder( tf.float32, shape = [None, FLAGS.Ngrid1, FLAGS.Ngrid2, FLAGS.Ngrid3, nconditions], name='ground_truth_delta_rhos')
        self.energies_placeholder = tf.placeholder( tf.float32, shape= [None, 1], name='ground_truth_energies')

        self.dnn_output = self.DNN( self.conditions_placeholder )
        self.loss_rho = tf.norm(   tf.subtract( self.dnn_output[0], self.deltarhos_placeholder )   , ord=1)
        self.loss_e = tf.norm(   tf.subtract( self.dnn_output[1], tf.reshape(self.energies_placeholder,[-1]) )   , ord=1)
        self.loss = self.loss_rho + FLAGS.weight_E * self.loss_e

        self.all_trainable_variables = tf.trainable_variables()
        # regularization is performed only for the part of the DNN computing energies after the bifurcation,
        #   and only for kernel variables (biases are not included into regularization, which is a common practice)
        self.variables_wEregularization = [var for var in self.all_trainable_variables if ('h13_conv' in var.name or 'h14_conv' in var.name) and 'kernel' in var.name]
        # variables for transfer learning for energy or rho predictions:
        self.variables_in_E_prediction = [var for var in self.all_trainable_variables if ('h13_conv' in var.name or 'h14_conv' in var.name or 'h12_deconv' in var.name)]
        self.variables_in_rho_prediction = [var for var in self.all_trainable_variables if ('h12_conv' in var.name or 'predicted_deltarho' in var.name or 'h12_deconv' in var.name)]

        self.wEregularization = FLAGS.weight_wE * tf.add_n( [tf.norm(self.variables_wEregularization[i], ord=2)  for i in range(len(self.variables_wEregularization))])
        self.loss_with_wEregularization = self.loss + self.wEregularization

        self.saver = tf.train.Saver(max_to_keep=3)

    def DNN(self, conditions):

        # smoothing/saturating input
        h0 = tf.tanh( tf.scalar_mul(FLAGS.prefactor_tanh_smoothing_rho, conditions) )

        # encoder
        # 64^3 x 1 -> 32^3 x 128
        h1conv = tf.layers.conv3d( inputs=h0, filters=128, kernel_size=3, padding="same", activation=tf.nn.relu, name='h1_conv')
        h1pool = tf.layers.max_pooling3d( inputs=h1conv, pool_size=2, strides=2, name='h1_pool')
        # 32^3 x 128 -> 16^3 x 256
        h2conv = tf.layers.conv3d( inputs=h1pool, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, name='h2_conv')
        h2pool = tf.layers.max_pooling3d( inputs=h2conv, pool_size=2, strides=2, name='h2_pool')
        # 16^3 x 256 -> 8^3 x 256
        h3conv = tf.layers.conv3d( inputs=h2pool, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, name='h3_conv')
        h3pool = tf.layers.max_pooling3d( inputs=h3conv, pool_size=2, strides=2, name='h3_pool')
        # 8^3 x 256 -> 4^3 x 256
        h4conv = tf.layers.conv3d( inputs=h3pool, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, name='h4_conv')
        h4pool = tf.layers.max_pooling3d( inputs=h4conv, pool_size=2, strides=2, name='h4_pool')
        # 4^3 x 256 -> 2^3 x 256
        h5conv = tf.layers.conv3d( inputs=h4pool, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, name='h5_conv')
        h5pool = tf.layers.max_pooling3d( inputs=h5conv, pool_size=2, strides=2, name='h5_pool')

        # decoder
        # 2^3 x 256 -> 4^3 x 256; concat w prev 4^3 x 256; conv to 4^3 x 256
        h8deconv = tf.layers.conv3d_transpose( inputs=h5pool, filters=256, kernel_size=2, strides=2, activation=None, use_bias=False, name='h8_deconv')
        h8concat = tf.concat( [h8deconv, h4pool], 4, name='h8_concat')
        h8conv = tf.layers.conv3d( inputs=h8concat, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, name='h8_conv')
        # 4^3 x 256 -> 8^3 x 256 ; concat w prev 8^3 x 256; conv to 8^3 x 256
        h9deconv = tf.layers.conv3d_transpose( inputs=h8conv, filters=256, kernel_size=2, strides=2, activation=None, use_bias=False, name='h9_deconv')
        h9concat = tf.concat( [h9deconv, h3pool], 4, name='h9_concat')
        h9conv = tf.layers.conv3d( inputs=h9concat, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, name='h9_conv')
        # 8^3 x 256 -> 16^3 x 256; concat w prev 16^3 x 256; conv to 16^3 x 128
        h10deconv = tf.layers.conv3d_transpose( inputs=h9conv, filters=256, kernel_size=2, strides=2, activation=None, use_bias=False, name='h10_deconv')
        h10concat = tf.concat( [h10deconv, h2pool], 4, name='h10_concat')
        h10conv = tf.layers.conv3d( inputs=h10concat, filters=128, kernel_size=3, padding="same", activation=tf.nn.relu, name='h10_conv')
        # 16^3 x 128 -> 32^3 x 128; concat w prev 32^3 x 128; conv to 32^3 x 64
        h11deconv = tf.layers.conv3d_transpose( inputs=h10conv, filters=128, kernel_size=2, strides=2, activation=None, use_bias=False, name='h11_deconv')
        h11concat = tf.concat( [h11deconv, h1pool], 4, name='h11_concat')
        h11conv = tf.layers.conv3d( inputs=h11concat, filters=64, kernel_size=3, padding="same", activation=tf.nn.relu, name='h11_conv')
        # 32^3 x 64 -> 64^3 x 64; concat w prev 64^3 x 1; conv to 64^3 x 32; conv wo activation to 64^3 x 1
        h12deconv = tf.layers.conv3d_transpose( inputs=h11conv, filters=64, kernel_size=2, strides=2, activation=None, use_bias=False, name='h12_deconv')
        h12concat = tf.concat( [h12deconv, conditions], 4, name='h12_concat')
        # computations bifurcate here
        # here deltarho is computed
        h12conv = tf.layers.conv3d( inputs=h12concat, filters=32, kernel_size=3, padding="same", activation=tf.nn.relu, name='h12_conv')
        predicted_deltarho = tf.layers.conv3d( inputs=h12conv, filters=1, kernel_size=3, padding="same", activation=None, use_bias=False, name='predicted_deltarho')
        # and here energy is computed
        h13conv = tf.layers.conv3d( inputs=h12concat, filters=32, kernel_size=3, padding="same", activation=tf.nn.relu, name='h13_conv')
        h14convP = tf.layers.conv3d( inputs=h13conv, filters=1, kernel_size=3, padding="same", activation=tf.nn.relu, name='h14_convP')
        h14convN = tf.layers.conv3d( inputs=h13conv, filters=1, kernel_size=3, padding="same", activation=tf.nn.relu, name='h14_convN')
        predicted_E = tf.reshape(   tf.layers.average_pooling3d( inputs=h14convP, pool_size=64, strides=1, padding='valid', name='convP_pooling')  -
                              tf.layers.average_pooling3d( inputs=h14convN, pool_size=64, strides=1, padding='valid', name='convN_pooling')
                     , [-1])

        return predicted_deltarho, predicted_E

    def which_subset(self, qm9_i, split_method):

        if split_method == 'based_on_3rd_digit_9ha':
            if qm9_i < 21989:
                return 'testing2'
            else:
                residue = qm9_i % 1000 // 100
                if residue > 1:
                    return 'training'
                elif residue == 1:
                    return 'validation'
                else:
                    return 'testing1'

        elif split_method == 'based_on_3rd_digit':
            # training set includes molecules with QM9 indices ???[2-9]??; validation set with indices ???1??; test set with indices ???0??
            # motivation: ???0?? includes qm9 entries 1-99, for which we have high theory level QM solutions; ???[2-9]?? is max away from it
            residue = qm9_i % 1000 // 100
            if residue > 1:
                return 'training'
            elif residue == 1:
                return 'validation'
            else:
                return 'testing1'

        elif split_method == 'no_validation':
            # train on ???[1-9]??, test on ???0??
            residue = qm9_i % 1000 // 100
            if residue > 0:
                return 'training'
            else:
                return 'testing1'

        elif split_method == 'train_on_all_data':
            return 'training'

        elif split_method == 'train_on_molecules_w_3or4_heavy_atoms':
            if 8 < qm9_i < 49:
                return 'training'
            else:
                return 'testing1'

        else:
            print("Error: in 'which_subset': It has never been like this, and now it is exactly the same again.")
            exit()

    def generate_minibatches(self):

        list_of_QM9_indices_qm_computations_not_converged = [5, 1460, 4095, 4316, 4611, 4925, 5005, 9414, 10927, 11370, 21421, 23990, 24513, 25366, 25430, 30547, 32045, 32310, 33702, 34164, 35927, 36818, 42728, 51838, 59708, 60790, 66644, 67359, 71466, 72032, 72121, 74781, 75823, 75829, 77229, 77359, 77712, 78703, 78784, 78802, 78822, 78823, 78824, 78847, 78848, 79189, 93407, 98794, 98954, 112340, 115484, 124157, 124833, 129394, 132932]
        list_of_QM9_indices_unreliable_qm_computations = [48259, 17522, 112542, 132186, 125495, 95418, 1616, 2091, 95104, 96194, 122276, 118622, 45658, 131034, 73693, 26304, 132519, 61269, 54252, 130876, 21352, 42678, 25433, 19132, 71584, 46334, 45823, 20335, 2624, 3299, 70104, 33387, 15629, 71817, 132571, 99031, 25453, 94097, 130806, 64139, 15707, 31446, 20351, 53667, 49797, 3825, 2993, 132463, 21389, 75050]
        list_of_excluded_QM9_indices = list_of_QM9_indices_qm_computations_not_converged + list_of_QM9_indices_unreliable_qm_computations

        # load energies -- here and below, by energies I mean (E - sum(E_atoms) - (deltaE_HF-sum(E_HF_atoms) - linear_correction),
        #   where linear_correction = c_0 + sum{atom in H, C, N, O, F} c_atom*n_atom, where c_0 and {c_atom} are found from DFT energies
        #   even if I'm doing transfer learning on CCSD(T) energies: I have to use the same linear_correction as in the training on DFT to minimize changes in the network due to retraining
        qm9_indices = np.load( FLAGS.file_w_QM9_indices )
        energy_for_given_qm9_i = np.zeros( 133885 + 1, dtype = np.float64)
        try:
            energies = np.load( FLAGS.file_w_corrections_to_energies )
            for i in range(len(qm9_indices)):
                energy_for_given_qm9_i[ qm9_indices[i] ] = energies[i]
            energies_have_been_loaded = True
        except:
            energies_have_been_loaded = False

        # split data into training, validation and up to two testing sets
        subset = {}
        subset['training'], subset['validation'], subset['testing1'], subset['testing2'] = [], [], [], []
        assert FLAGS.split_method in ['based_on_3rd_digit_9ha', 'based_on_3rd_digit', 'no_validation', 'train_on_all_data', 'train_on_molecules_w_3or4_heavy_atoms'], \
            "Error: unrecognized option for split_method '" + FLAGS.split_method + "', must be 'based_on_3rd_digit_9ha' or 'based_on_3rd_digit' or 'no_validation' or 'train_on_all_data' or 'train_on_molecules_w_3or4_heavy_atoms'"
        for i in range(len(qm9_indices)):
            qm9_i = qm9_indices[i]
            if qm9_i not in list_of_excluded_QM9_indices:
                subset[ self.which_subset(qm9_i, FLAGS.split_method) ].append(qm9_i)
        np.random.shuffle(subset['training'])
        print('Found {} molecules: {} for training, {} for validation, {} for testing1 and {} for testing2'
                  .format(
                         sum( len(subset[subset_name]) for subset_name in ['training', 'validation', 'testing1', 'testing2'] ), 
                         len(subset['training']), len(subset['validation']), len(subset['testing1']), len(subset['testing2']) 
                  ), flush=True)
        print('')

        # generate minibatches
        if not os.path.exists( FLAGS.dataset ):
            os.makedirs( FLAGS.dataset )
        for subset_name in ['training', 'validation', 'testing1', 'testing2']:
            subset_size = len(subset[subset_name])
            n_batches = subset_size // FLAGS.batch_size
            size_of_incomplete_batch = subset_size - n_batches * FLAGS.batch_size
            if size_of_incomplete_batch != 0:
                print('Warning: the size of the {} set ({}) is not a multiple of batch_size ({}); the last batch will contain only {} molecules'
                    .format(subset_name, subset_size, FLAGS.batch_size, size_of_incomplete_batch), flush=True )
                n_batches += 1
            for i_batch in range(n_batches):
                deltarhos_in_this_batch, conditions_in_this_batch, energies_in_this_batch = [], [], []
                if i_batch == n_batches-1:
                    qm9_indices = subset[subset_name][ i_batch*FLAGS.batch_size : subset_size ]
                else:
                    qm9_indices = subset[subset_name][ i_batch*FLAGS.batch_size : (i_batch+1)*FLAGS.batch_size ]
                for qm9_i in qm9_indices:
                    condition_filename = FLAGS.directory_w_densities + '/' + FLAGS.beginning_of_condition_density_filename + str(qm9_i).zfill(6) + FLAGS.end_of_condition_density_filename
                    rho_or_deltarho_filename = FLAGS.directory_w_densities + '/' + FLAGS.beginning_of_rho_or_deltarho_filename + str(qm9_i).zfill(6) + FLAGS.end_of_rho_or_deltarho_filename
                    try:
                        conditions = np.load(condition_filename)
                        if FLAGS.use_deltarho_npy_files:
                            deltarho = np.load(rho_or_deltarho_filename)
                        else:
                            rho = np.load(rho_or_deltarho_filename)
                            deltarho = rho - conditions
                        deltarhos_in_this_batch.append(deltarho)
                        conditions_in_this_batch.append(conditions)
                        if energies_have_been_loaded:
                            energies_in_this_batch.append( energy_for_given_qm9_i[ qm9_i ] )
                    except:
                        pass
                if len(deltarhos_in_this_batch) < FLAGS.batch_size or len(conditions_in_this_batch) < FLAGS.batch_size or len(energies_in_this_batch) < FLAGS.batch_size:
                    print('Warning: in batch [{}/{}] for {}, there are only {}, {} and {} molecules in deltarho, conditions and energy datasets, respectively, instead of {}'
                        .format(i_batch, n_batches, subset_name, len(deltarhos_in_this_batch), len(conditions_in_this_batch), len(energies_in_this_batch), FLAGS.batch_size), flush=True )
                conditions_in_this_batch = np.reshape( conditions_in_this_batch, [-1, FLAGS.Ngrid1, FLAGS.Ngrid2, FLAGS.Ngrid3, 1] )
                deltarhos_in_this_batch = np.reshape( deltarhos_in_this_batch, [-1, FLAGS.Ngrid1, FLAGS.Ngrid2, FLAGS.Ngrid3, 1] )
                np.save( FLAGS.dataset + '/' + subset_name + '_conditions_batch' + str(i_batch) + '.npy', conditions_in_this_batch )
                np.save( FLAGS.dataset + '/' + subset_name + '_deltarhos_batch' + str(i_batch) + '.npy', deltarhos_in_this_batch )
                if energies_have_been_loaded:
                    energies_in_this_batch = np.reshape( energies_in_this_batch, [-1, 1] )
                    np.save( FLAGS.dataset + '/' + subset_name + '_energies_batch' + str(i_batch) + '.npy', energies_in_this_batch )
                print('Saved minibatch {} / {} for {}, with molecules with QM9 indices {}'
                        .format(i_batch, n_batches, subset_name, qm9_indices), flush=True )
                # help! they keep me in a cubicle! I'm coding for food!

    def save(self, checkpoint_dir, step):

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, "PML_QC"), global_step=step)

    def load(self, checkpoint_dir):

        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt_state and ckpt_state.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt_state.model_checkpoint_path)
            self.initial_step = int(os.path.basename(ckpt_state.model_checkpoint_path).split('-')[1])
            return True
        else:
            self.initial_step = 0
            return False

    def train(self):

        self.build_model()
        optimizer = tf.train.AdamOptimizer( FLAGS.learning_rate, beta1 = FLAGS.beta1, epsilon = FLAGS.epsilon).minimize( self.loss_with_wEregularization, 
            var_list = self.all_trainable_variables )
        n_batches = {}
        for subset_name in ['training', 'validation']:
            n_batches[subset_name] = len( glob(FLAGS.dataset + '/' + subset_name + '_deltarhos_batch*.npy') )

        with tf.Session() as sess:
            self.sess = sess
            tf.global_variables_initializer().run()
            if self.load(FLAGS.checkpoint_dir):
                print( 'Loaded model from checkpoint directory ' + FLAGS.checkpoint_dir)
            else:
                print( 'No checkpoint directory ' + FLAGS.checkpoint_dir + ' found, initialised a new model with random coefficients')
            t0_train = time()
            print('Found {} minibatches for training and {} minibatches for validation'.format( n_batches['training'], n_batches['validation'] ), flush=True )

            # Origine ex pura ad optimum futurum, ad optimum futurum iam nunc egressus sum
            for i_epoch in range( self.initial_step + 1, FLAGS.epochs + self.initial_step + 1):
                for subset_name in ['training', 'validation']:
                    if n_batches[subset_name] > 0:
                        sum_err_deltarho, sum_err_e, sum_wEregularization, n_molecules = 0., 0., 0., 0
                        list_of_i_batches = list(range( n_batches[subset_name] ))
                        if subset_name == 'training':
                            np.random.shuffle( list_of_i_batches )
                        for i_batch in list_of_i_batches:
                            try:
                                conditions_in_this_batch = np.load( FLAGS.dataset + '/' + subset_name + '_conditions_batch' + str(i_batch) + '.npy' )
                                deltarhos_in_this_batch = np.load( FLAGS.dataset + '/' + subset_name + '_deltarhos_batch' + str(i_batch) + '.npy' )
                                energies_in_this_batch = np.load( FLAGS.dataset + '/' + subset_name + '_energies_batch' + str(i_batch) + '.npy' )
                            except:
                                print('Error: for {} batch {}, cannot find at least one of the following files: {}, {}, and {}'
                                      .format(subset_name, i_batch, subset_name + '_conditions_batch' + str(i_batch) + '.npy', 
                                      subset_name + '_deltarhos_batch' + str(i_batch) + '.npy', subset_name + '_energies_batch' + str(i_batch) + '.npy'), flush=True )
                            if subset_name == 'training': #{
                                 _ = sess.run( optimizer, feed_dict={self.conditions_placeholder: conditions_in_this_batch,
                                                                    self.deltarhos_placeholder: deltarhos_in_this_batch,
                                                                    self.energies_placeholder: energies_in_this_batch}      )
                            #}
                            # BTW, did you know that python supports curly braces in if statements? just see above
                            err_deltarho, err_e, wEregularization_penalty = sess.run( [self.loss_rho, self.loss_e, self.wEregularization],
                                                                                      feed_dict={self.conditions_placeholder: conditions_in_this_batch,
                                                                                                 self.deltarhos_placeholder: deltarhos_in_this_batch,
                                                                                                 self.energies_placeholder: energies_in_this_batch}  )
                            size_of_this_batch = len(deltarhos_in_this_batch)
                            sum_err_deltarho += err_deltarho
                            sum_err_e += err_e
                            sum_wEregularization += wEregularization_penalty
                            n_molecules += size_of_this_batch
                            print('Epoch [{}] batch [{}], {}: L1_rho per molecule: {:4.6f}, L1_e per molecule: {:4.3f} kcal/mol'
                                  .format(i_epoch, i_batch, subset_name, err_deltarho/size_of_this_batch/1e3, err_e/size_of_this_batch*hartree_to_kcalmol), end='', flush=True )
                            if FLAGS.weight_E != 0 and FLAGS.weight_wE != 0:
                                print(', regularization penalty per molecule: {:4.3f} kcal/mol'
                                      .format(wEregularization_penalty/size_of_this_batch/FLAGS.weight_E*hartree_to_kcalmol), flush=True )
                            else:
                                  print('', flush=True )
                        print('==End of epoch [{}], {}: {:4.3f} h, L1_rho per molecule: {:4.6f}, L1_e per molecule: {:4.3f} kcal/mol'
                              .format(i_epoch, subset_name, (time()-t0_train)/3600, sum_err_deltarho/n_molecules/1e3, sum_err_e/n_molecules*hartree_to_kcalmol), end='', flush=True )
                        if FLAGS.weight_E != 0 and FLAGS.weight_wE != 0:
                            print(', regularization penalty per molecule: {:4.3f} kcal/mol'
                                    .format(sum_wEregularization/n_molecules/FLAGS.weight_E*hartree_to_kcalmol), flush=True )
                        else:
                            print('', flush=True )
                    # at the end of each training epoch, save the checkpoint file ...
                    if subset_name == 'training':
                        self.save( FLAGS.checkpoint_dir, i_epoch )

    def transfer_learning(self):

        self.build_model()
        optimizer = {}
        optimizer['rho'] = tf.train.AdamOptimizer( FLAGS.learning_rate, beta1 = FLAGS.beta1, epsilon = FLAGS.epsilon).minimize( self.loss_rho,
            var_list = self.variables_in_rho_prediction )
        optimizer['E'] = tf.train.AdamOptimizer( FLAGS.learning_rate, beta1 = FLAGS.beta1, epsilon = FLAGS.epsilon).minimize( self.loss_e, 
            var_list = self.variables_in_E_prediction )
        n_batches = {}
        for subset_name in ['training', 'validation']:
            n_batches[subset_name] = {}
            n_batches[subset_name]['rho'] = len( glob(FLAGS.dataset_rho + '/' + subset_name + '_conditions_batch*.npy') )
            n_batches[subset_name]['E'] = len( glob(FLAGS.dataset_E + '/' + subset_name + '_conditions_batch*.npy') )
        miniepochs_per_epoch = {}
        for subset_name in ['training', 'validation']:
            miniepochs_per_epoch[subset_name] = {}
        miniepochs_per_epoch['training']['rho'] = FLAGS.miniepochs_for_rho_per_epoch
        miniepochs_per_epoch['training']['E'] = FLAGS.miniepochs_for_E_per_epoch
        for predicted_variable in ['rho', 'E']:
            miniepochs_per_epoch['validation'][predicted_variable] = 1

        with tf.Session() as sess:
            self.sess = sess
            tf.global_variables_initializer().run()
            if self.load(FLAGS.checkpoint_dir):
                print( 'Loaded model from checkpoint directory ' + FLAGS.checkpoint_dir)
            else:
                print( 'No checkpoint directory ' + FLAGS.checkpoint_dir + ' found, initialised a new model with random coefficients')
            t0_train = time()
            for predicted_variable in ['rho', 'E']:
                print('Transfer learning on {}: Found {} minibatches for training and {} minibatches for validation'
                      .format( predicted_variable, n_batches['training'][predicted_variable], n_batches['validation'][predicted_variable] ), flush=True )

            # Origine ex pura ad optimum futurum, ad optimum futurum iam nunc egressus sum
            for i_epoch in range( self.initial_step + 1, FLAGS.epochs + self.initial_step + 1):
                for subset_name in ['training', 'validation']:
                    predicted_variable = 'rho'
                    for i_miniepoch in range( miniepochs_per_epoch[subset_name][predicted_variable] ):
                        sum_err_deltarho, n_molecules = 0., 0
                        list_of_i_batches = list(range( n_batches[subset_name][predicted_variable] ))
                        if subset_name == 'training':
                            np.random.shuffle( list_of_i_batches )
                        for i_batch in list_of_i_batches:
                            try:
                                conditions_in_this_batch = np.load( FLAGS.dataset_rho + '/' + subset_name + '_conditions_batch' + str(i_batch) + '.npy' )
                                deltarhos_in_this_batch = np.load( FLAGS.dataset_rho + '/' + subset_name + '_deltarhos_batch' + str(i_batch) + '.npy' )
                            except:
                                print('Error: for {} pv {} batch {}, cannot open at least one of the following files: {},and {}'
                                      .format(subset_name, predicted_variable, i_batch, subset_name + '_conditions_batch' + str(i_batch) + '.npy',
                                      subset_name + '_deltarhos_batch' + str(i_batch) + '.npy'), flush=True )
                            size_of_this_batch = len(conditions_in_this_batch)
                            if size_of_this_batch > 0:
                                if subset_name == 'training':
                                    _ = sess.run( optimizer['rho'], feed_dict={self.conditions_placeholder: conditions_in_this_batch,
                                                                               self.deltarhos_placeholder: deltarhos_in_this_batch}      )
                                err_deltarho = sess.run( self.loss_rho, feed_dict={self.conditions_placeholder: conditions_in_this_batch,
                                                                                   self.deltarhos_placeholder: deltarhos_in_this_batch}  )
                                sum_err_deltarho += err_deltarho
                                n_molecules += size_of_this_batch
                                print('Epoch [{}] pv {} miniepoch {} batch [{}], {}: L1_rho per molecule: {:4.6f}'
                                      .format(i_epoch, predicted_variable, i_miniepoch, i_batch, subset_name, err_deltarho/size_of_this_batch/1e3), flush=True )
                        if n_molecules > 0:
                            print('==End of epoch [{}] {} pv {} miniepoch {}: {:4.3f} h, L1_rho per molecule: {:4.6f}'
                                  .format(i_epoch, subset_name, predicted_variable, i_miniepoch, (time()-t0_train)/3600, sum_err_deltarho/n_molecules/1e3), flush=True )
                    predicted_variable = 'E'
                    for i_miniepoch in range( miniepochs_per_epoch[subset_name][predicted_variable] ):
                        sum_err_e, sum_wEregularization, n_molecules = 0., 0., 0
                        list_of_i_batches = list(range( n_batches[subset_name][predicted_variable] ))
                        if subset_name == 'training':
                            np.random.shuffle( list_of_i_batches )
                        for i_batch in list_of_i_batches:
                            try:
                                conditions_in_this_batch = np.load( FLAGS.dataset_E + '/' + subset_name + '_conditions_batch' + str(i_batch) + '.npy' )
                                energies_in_this_batch = np.load( FLAGS.dataset_E + '/' + subset_name + '_energies_batch' + str(i_batch) + '.npy' )
                            except:
                                print('Error: for {} pv {} batch {}, cannot open at least one of the following files: {},and {}'
                                      .format(subset_name, predicted_variable, i_batch, subset_name + '_conditions_batch' + str(i_batch) + '.npy',
                                      subset_name + '_energies_batch' + str(i_batch) + '.npy'), flush=True )
                            size_of_this_batch = len(conditions_in_this_batch)
                            if size_of_this_batch > 0:
                                if subset_name == 'training':
                                    _ = sess.run( optimizer['E'], feed_dict={self.conditions_placeholder: conditions_in_this_batch,
                                                                             self.energies_placeholder: energies_in_this_batch}      )
                                err_e, wEregularization_penalty = sess.run( [self.loss_e, self.wEregularization],
                                                                                          feed_dict={self.conditions_placeholder: conditions_in_this_batch,
                                                                                          self.energies_placeholder: energies_in_this_batch}  )
                                sum_err_e += err_e
                                sum_wEregularization += wEregularization_penalty
                                n_molecules += size_of_this_batch
                                print('Epoch [{}] pv {} miniepoch {} batch [{}], {}: L1_e per molecule: {:4.3f} kcal/mol'
                                      .format(i_epoch, predicted_variable, i_miniepoch, i_batch, subset_name, err_e/size_of_this_batch*hartree_to_kcalmol), end='', flush=True )
                                if FLAGS.weight_E != 0 and FLAGS.weight_wE != 0:
                                      print(', regularization penalty per molecule: {:4.3f} kcal/mol'
                                            .format(wEregularization_penalty/size_of_this_batch/FLAGS.weight_E*hartree_to_kcalmol), flush=True )
                                else:
                                      print('', flush=True )
                        if n_molecules > 0:
                            print('==End of epoch [{}] {} pv {} miniepoch {}: {:4.3f} h, L1_e per molecule: {:4.3f} kcal/mol'
                                  .format(i_epoch, subset_name, predicted_variable, i_miniepoch, (time()-t0_train)/3600, sum_err_e/n_molecules*hartree_to_kcalmol), end='', flush=True )
                            if FLAGS.weight_E != 0 and FLAGS.weight_wE != 0:
                                print(', regularization penalty per molecule: {:4.3f} kcal/mol'
                                        .format(sum_wEregularization/n_molecules/FLAGS.weight_E*hartree_to_kcalmol), flush=True )
                            else:
                                print('', flush=True )

                    # at the end of each training epoch, save the checkpoint file ...
                    if subset_name == 'training':
                        self.save( FLAGS.checkpoint_dir, i_epoch )



    def predict(self):

        self.build_model()
        with tf.Session() as sess:
            self.sess = sess
            if self.load(FLAGS.checkpoint_dir):
                print( 'Loaded model from checkpoint directory ' + FLAGS.checkpoint_dir)
            else:
                print( 'Error: No checkpoint directory ' + FLAGS.checkpoint_dir)
                exit()

            energy_errors = []
            rho_errors, rho_rel_errors = [], []
            for conditions_filename in glob(FLAGS.condition_files):

                # load conditions (that is HF densities) and predict deltarhos and energies
                print('Loading file {}'.format(conditions_filename), flush=True )
                try:
                    conditions_in_this_batch = np.load( conditions_filename )
                except:
                    print("Error: couldn't load file {}, skipping it".format(conditions_filename), flush=True )
                    continue
                conditions_in_this_batch = np.reshape( conditions_in_this_batch, [-1, FLAGS.Ngrid1, FLAGS.Ngrid2, FLAGS.Ngrid3, 1] )
                size_of_this_batch = len( conditions_in_this_batch )
                predicted_deltarhos_in_this_batch, predicted_energies_in_this_batch = sess.run( self.dnn_output,
                                                                                  feed_dict={self.conditions_placeholder: conditions_in_this_batch} )
                if len(predicted_energies_in_this_batch) == 1:
                    print('Correction to energy predicted by this DNN for {} (in Hartrees) is '.format(conditions_filename), end='', flush=True )
                else:
                    print('Corrections to energies predicted by this DNN for {} (in Hartrees) are '.format(conditions_filename), end='', flush=True )
                for i in range( len(predicted_energies_in_this_batch) - 1 ):
                    print('{:6.10f} '.format(predicted_energies_in_this_batch[i]), end='', flush=True )
                print('{:6.10f}'.format(predicted_energies_in_this_batch[ len(predicted_energies_in_this_batch) - 1 ]), flush=True )

                # save predictions as npy files, if requested
                assert FLAGS.save_files_with_predictions in ['yes', 'no'], \
                    "Error: save_files_with_predictions must be either 'yes' or 'no'; found '" + FLAGS.save_files_with_predictions + "' instead."
                if FLAGS.save_files_with_predictions == 'yes':
                    if "conditions" in conditions_filename:
                        predicted_deltarhos_filename = conditions_filename.replace("conditions", "deltarhos_predicted")
                        predicted_energies_filename = conditions_filename.replace("conditions", "energies_predicted")
                    else:
                        predicted_deltarhos_filename = conditions_filename.replace(".npy", ".deltarhos_predicted.npy")
                        predicted_energies_filename = conditions_filename.replace(".npy", ".energies_predicted.npy")
                    print('Saving predicted deltarhos and energies in files {} and {}'.format(predicted_deltarhos_filename, predicted_energies_filename), flush=True )
                    np.save(predicted_deltarhos_filename, predicted_deltarhos_in_this_batch)
                    np.save(predicted_energies_filename, predicted_energies_in_this_batch)

                # if ground truth values of energies and/or deltarhos are available, compute the error of the DNN
                assert FLAGS.use_deltarho_npy_files in ['yes', 'no'], \
                    "Error: use_deltarho_npy_files must be either 'yes' or 'no'; found '" + FLAGS.use_deltarho_npy_files + "' instead."

                # figuring out the names of the files with the ground truth
                if "conditions" in conditions_filename:
                    ground_truth_energies_filename = conditions_filename.replace("conditions", "energies")
                    if FLAGS.use_deltarho_npy_files == 'yes':
                        ground_truth_deltarhos_filename = conditions_filename.replace("conditions", "deltarhos")
                    else:
                        ground_truth_rhos_filename = conditions_filename.replace("conditions", "rhos")
                else:
                    #ground_truth_energies_filename = conditions_filename.replace(".npy", ".energies.npy")
                    ground_truth_energies_filename = conditions_filename.replace("_HF_cc-pVDZ_centered.64x64x64.npy", ".QM9.energy.npy")
                    if FLAGS.use_deltarho_npy_files == 'yes':
                        ground_truth_deltarhos_filename = conditions_filename.replace(".npy", ".deltarhos.npy")
                    else:
                        #ground_truth_rhos_filename = conditions_filename.replace(".npy", ".rhos.npy")
                        ground_truth_rhos_filename = conditions_filename.replace("_HF_cc-pVDZ_centered.64x64x64.npy", "_PBE1PBE_pcS-3_centered.64x64x64.npy")

                # compute the errors in the predicted densities
                if ( FLAGS.use_deltarho_npy_files == 'yes' and os.path.isfile( ground_truth_deltarhos_filename ) ) or \
                        ( FLAGS.use_deltarho_npy_files == 'no' and os.path.isfile( ground_truth_rhos_filename ) ):
                    if FLAGS.use_deltarho_npy_files == 'yes':
                        deltarhos_in_this_batch = np.load( ground_truth_deltarhos_filename )
                        deltarhos_in_this_batch = np.reshape( deltarhos_in_this_batch, [-1, FLAGS.Ngrid1, FLAGS.Ngrid2, FLAGS.Ngrid3, 1] )
                    else:
                        rhos_in_this_batch = np.load( ground_truth_rhos_filename )
                        deltarhos_in_this_batch = np.reshape( rhos_in_this_batch, [-1, FLAGS.Ngrid1, FLAGS.Ngrid2, FLAGS.Ngrid3, 1] ) - conditions_in_this_batch
                    abs_delta = abs( predicted_deltarhos_in_this_batch - deltarhos_in_this_batch )
                    abs_delta_HF = abs( predicted_deltarhos_in_this_batch)
                    current_rho_errors, current_rho_rel_errors = [], []
                    for i in range(len(abs_delta)):
                        current_rho_error = abs_delta[i].sum()/1e3
                        current_rho_HF_error = abs_delta_HF[i].sum()/1e3
                            # 1e-3 above is the voxel volume: the original grid spacing in 256x256x256 files is 0.1 Bohr, so the voxel volume is (0.1)^3 Bohr^3
                        current_rho_errors.extend( [current_rho_error] )
                        current_rho_rel_errors.extend( [current_rho_error/current_rho_HF_error] )
                    print('{} L1_rho values: '.format(conditions_filename), end='', flush=True )
                    for i in range( len(current_rho_errors) ):
                        print(' {:6.10f}'.format( current_rho_errors[i] ), end='', flush=True )
                    print('', flush=True )
                    print('{} L1_rho_rel_to_HF values: '.format(conditions_filename), end='', flush=True )
                    for i in range( len(current_rho_rel_errors) ):
                        print(' {:6.10f}'.format( current_rho_rel_errors[i] ), end='', flush=True )
                    print('', flush=True )
                    err_deltarho = abs_delta.sum()
                    err_deltarho_HF = abs_delta_HF.sum()
                    print('{}: L1_rho per molecule: {:4.6f}; relative to HF (zero DNN output): {:4.6f}'
                            .format(conditions_filename, err_deltarho/size_of_this_batch/1e3, err_deltarho/err_deltarho_HF ), flush=True )
                    rho_errors.extend( current_rho_errors )
                    rho_rel_errors.extend( current_rho_rel_errors )

                # compute the errors in the predicted energies
                if os.path.isfile( ground_truth_energies_filename ):
                    energies_in_this_batch = np.load( ground_truth_energies_filename )
                    energies_in_this_batch = np.reshape( energies_in_this_batch, [-1] )
                    print('{} energies, in kcal/mol: ground truth:'.format(conditions_filename), end='', flush=True )
                    for i in range( len(energies_in_this_batch) ):
                        print(' {:6.10f}'.format( energies_in_this_batch[i]*hartree_to_kcalmol ), end='', flush=True )
                    print(', predicted w DNN:', end='', flush=True )
                    for i in range( len(predicted_energies_in_this_batch) ):
                        print(' {:6.10f}'.format( predicted_energies_in_this_batch[i]*hartree_to_kcalmol ), end='', flush=True )
                    print('', flush=True )
                    err_e = abs( predicted_energies_in_this_batch - energies_in_this_batch ).sum()
                    print('{}: L1_e per molecule: {:4.3f} kcal/mol'
                            .format(conditions_filename, err_e/size_of_this_batch*hartree_to_kcalmol ), flush=True )
                    energy_errors.extend( abs( predicted_energies_in_this_batch - energies_in_this_batch ) )

                print('', flush=True)
            # statistics on errors for all molecules
            energy_errors = np.abs(energy_errors)
            print('In total, calculated {} energies: MAE {} kcal/mol, RMSE {} kcal/mol, median of abs error {} kcal/mol'
                            .format( len(energy_errors), np.mean(energy_errors)*hartree_to_kcalmol,
                                   np.sqrt( np.mean(energy_errors**2) )*hartree_to_kcalmol, np.median(energy_errors)*hartree_to_kcalmol ), flush=True )
#            print('In total, calculated {} electron densities: L1_rho per molecule: MAE {}, median of abs error {}; L1_rho_rel_to_HF: MAE {}, median of abs error {}'
#                            .format( len(rho_errors), np.mean(rho_errors), np.median(rho_errors), np.mean(rho_rel_errors), np.median(rho_rel_errors) ), flush=True )

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('Ngrid1', 64, "Size of grid along dimension 1")
tf.app.flags.DEFINE_integer('Ngrid2', 64, "Size of grid along dimension 2")
tf.app.flags.DEFINE_integer('Ngrid3', 64, "Size of grid along dimension 3")
tf.app.flags.DEFINE_string('mode', 'predict', "What to do: 'predict' (default) Use the DNN to make predictions, 'generate_minibatches' generate minibatch npy files, 'train' train the DNN")
tf.app.flags.DEFINE_string('file_w_QM9_indices', '../../QM9_Stanford_data/energies/QM9_indices_for_energy_discrepancies_133780molecules.npy', "File w QM9 indices of molecules")
tf.app.flags.DEFINE_string('file_w_corrections_to_energies', '../../QM9_Stanford_data/energies/QM9_energy_discrepancies_133780molecules.npy', "File w QM9 energy corrections, in the same order as QM9 indices in file_w_QM9_indices")
tf.app.flags.DEFINE_string('beginning_of_condition_density_filename', 'qm9_', "The part of the filename with a condition density before the QM9 index")
tf.app.flags.DEFINE_string('end_of_condition_density_filename', '_HF_cc-pVDZ_centered.64x64x64.npy', "The part of the filename with a condition density after the QM9 index")
tf.app.flags.DEFINE_string('beginning_of_rho_or_deltarho_filename', 'diff_qm9_', "The part of the filename with rho or deltarho before the QM9 index")
tf.app.flags.DEFINE_string('end_of_rho_or_deltarho_filename', '_PBE1PBE_pcS-3_HF_cc-pVDZ_centered.64x64x64.npy', "The part of the filename with rho or deltarho after the QM9 index")
tf.app.flags.DEFINE_string('directory_w_densities', '../../QM9_Stanford_data/densities', "Directory w densities of molecules")
tf.app.flags.DEFINE_string('dataset', '../../QM9_Stanford_data/minibatches_split_based_on_3rd_digit_9ha_batchsize16', "Path to the used dataset with minibatch npy files")
tf.app.flags.DEFINE_integer('batch_size', 16, "Number of molecules to process in a batch")
tf.app.flags.DEFINE_string('split_method', 'based_on_3rd_digit_9ha', "How to split the dataset into training, validation and test subsets; allowed values: 'based_on_3rd_digit_9ha' (default), 'based_on_3rd_digit', 'no_validation', 'train_on_all_data', 'train_on_molecules_w_3or4_heavy_atoms'")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint', "Directory to keep checkpoint files")
tf.app.flags.DEFINE_float('weight_E', 10000., "Weight for the energy term in the loss function")
tf.app.flags.DEFINE_float('prefactor_tanh_smoothing_rho', 0.02, "Prefactor in input = tanh(prefactor*rho_HF)")
tf.app.flags.DEFINE_float('beta1', 0.5, "Beta1 coefficient in AdamOptimizer")
tf.app.flags.DEFINE_float('learning_rate', 0.0002, "Learning rate")
tf.app.flags.DEFINE_integer('epochs', 100, "Number of epochs in training")
tf.app.flags.DEFINE_float('weight_wE', 0., "Weight for the regularization term for the energy in the loss function")
tf.app.flags.DEFINE_float('epsilon', 1e-08, "epsilon coefficient in AdamOptimizer")
tf.app.flags.DEFINE_integer('miniepochs_for_rho_per_epoch', 1, "How many miniepochs (cycles) of learning for rho to do per epoch in transfer learning")
tf.app.flags.DEFINE_integer('miniepochs_for_E_per_epoch', 1, "How many miniepochs (cycles) of learning for E to do per epoch in transfer learning")
tf.app.flags.DEFINE_string('dataset_rho', '../../QM9_Stanford_data/transfer_learning_rho_minibatches_batchsize16', "Path to the dataset with minibatch npy files for rho transfer learning")
tf.app.flags.DEFINE_string('dataset_E', '../../QM9_Stanford_data/transfer_learning_E_minibatches_batchsize16', "Path to the dataset with minibatch npy files for E transfer learning")
tf.app.flags.DEFINE_string('condition_files', '../../QM9_Stanford_data/minibatches_batchsize16/testing1_conditions_batch*.npy', "Files with conditions for predictions (if mask, use quotes) -- make sure the mask includes only condition files, but not deltarho and energy files!")
tf.app.flags.DEFINE_string('save_files_with_predictions', 'no', "Save files with predicted dd energies and delta rhos (yes/no)")
hartree_to_kcalmol = 627.509
tf.app.flags.DEFINE_string('use_deltarho_npy_files', 'yes', "If yes, deltarho npy files are used; if no, correlated rho npy files are used")


model = PML_QC()
time0 = time()

# just in case...
if True == False:
    print('There is something wrong in this universe...')
    exit()

if FLAGS.mode == 'generate_minibatches':
    model.generate_minibatches()
elif FLAGS.mode == 'train':
    model.train()
elif FLAGS.mode == 'predict':
    model.predict()
elif FLAGS.mode == 'transfer_learning':
    model.transfer_learning()
else:
    print('Illegal option for --mode')

time_now = time()
print('time to run: {} s'.format(time_now-time0) )

# thanks for being with us. Have a good rest of your day!
# Hmm, why people always wish only a good rest of the day?
# Let's do it in this way:
# Have a wonderful rest of your meaningless life!
# Kisses & hugs
