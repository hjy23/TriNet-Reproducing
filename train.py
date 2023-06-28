import argparse
import distutils.util
import pickle
import sys
import time

import numpy as np
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
sess = tf.compat.v1.Session(config=config)
from tensorflow import keras
# from feature_DCGR import *
# from feature_properties import *
# from feature_PSSM import *
from Model import *
from utils import *
import copy
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from preprocess import get_DCGR_features, get_properties_features, get_PSSM_features, load_fasta, get_blosum_features

"""
data processing part
"""


def load_dataset(dataset, fasta_file, pssm_dir, use_pssm, feature_path=f'./feature',
                 shuffle=True, shuffle_seed=7, mode='train'):
    feature_path = f'{feature_path}/{dataset}'
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    names, sequences, labels = load_fasta(fasta_file)
    dcgr_feas = get_DCGR_features(sequences,
                                  f'{feature_path}/dcgr_features_{mode}.pkl')  # 1376*158*8,through CNN
    prpt_feas = get_properties_features(sequences, f'{feature_path}/prpt_features_{mode}.pkl',
                                        theta=50)  # 1376*50*8, through Encoder
    if use_pssm:
        evolution_feas = get_PSSM_features(sequences, f'{feature_path}/pssm_features_{mode}.pkl',
                                           pssm_dir=pssm_dir, theta=50)  # 1376*50*20, through BiLSTM
    else:
        evolution_feas = get_blosum_features(sequences, f'{feature_path}/blosum_features_{mode}.pkl',
                                             f'./feature/blosum_dict.pkl', theta=50)  # 1376*50*20, through BiLSTM

    # shuffle
    if shuffle:
        print('Shuffle the data...')
        dcgr_feas, prpt_feas, evolution_feas, labels = shuffle_dataset(dcgr_feas, prpt_feas, evolution_feas, labels,
                                                                       shuffle_seed)

    return dcgr_feas, prpt_feas, evolution_feas, labels


def load_dataset_raw(dataset, train_fasta, test_fasta, train_pssm, test_pssm, use_pssm, feature_path=f'./feature',
                     shuffle=True, shuffle_seed=7):
    feature_path = f'{feature_path}/{dataset}'
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    names_tarin, sequences_train, label_tr_eval = load_fasta(train_fasta)
    names_test, sequences_test, test_label = load_fasta(test_fasta)
    dcgr_tr_eval = get_DCGR_features(sequences_train,
                                     f'{feature_path}/dcgr_features_train.pkl')  # 1376*158*8,through CNN
    prpt_tr_eval = get_properties_features(sequences_train, f'{feature_path}/prpt_features_train.pkl',
                                           theta=50)  # 1376*50*8, through Encoder
    if use_pssm:
        pssm_tr_eval = get_PSSM_features(sequences_train, f'{feature_path}/pssm_features_train.pkl',
                                         pssm_dir=train_pssm, theta=50)  # 1376*50*20, through BiLSTM
        pssm_test = get_PSSM_features(sequences_test, f'{feature_path}/pssm_features_test.pkl', pssm_dir=test_pssm,
                                      theta=50)  # 342*50*20, through BiLSTM
    else:
        pssm_tr_eval = get_blosum_features(sequences_train, f'{feature_path}/blosum_features_train.pkl',
                                           f'./feature/blosum_dict.pkl', theta=50)  # 1376*50*20, through BiLSTM
        pssm_test = get_blosum_features(sequences_test, f'{feature_path}/blosum_features_test.pkl',
                                        f'./feature/blosum_dict.pkl', theta=50)  # 1376*50*20, through BiLSTM
    dcgr_test = get_DCGR_features(sequences_test, f'{feature_path}/dcgr_features_test.pkl')  # 342*158*8, through CNN
    prpt_test = get_properties_features(sequences_test, f'{feature_path}/prpt_features_test.pkl',
                                        theta=50)  # 342*50*8, through Encoder

    # shuffle
    if shuffle:
        print('Shuffle the data...')
        dcgr_tr_eval, prpt_tr_eval, pssm_tr_eval, label_tr_eval = shuffle_dataset(dcgr_tr_eval, prpt_tr_eval,
                                                                                  pssm_tr_eval, label_tr_eval,
                                                                                  shuffle_seed)

        dcgr_test, prpt_test, pssm_test, test_label = shuffle_dataset(dcgr_test, prpt_test, pssm_test, test_label,
                                                                      shuffle_seed)
    test_data = [dcgr_test, pssm_test, prpt_test]
    return dcgr_tr_eval, prpt_tr_eval, pssm_tr_eval, label_tr_eval, test_data, test_label


def shuffle_dataset(dcgr, prpt, pssm, labels, shuffle_seed):
    np.random.seed(shuffle_seed)
    pos_num = len(np.where(np.array(labels) == 1)[0])
    # shuffle index
    index1 = np.arange(pos_num)  # positive sample
    np.random.shuffle(index1)
    index2 = np.arange(pos_num, len(labels))  # negative sample
    np.random.shuffle(index2)
    index = np.append(index1, index2)
    dcgr = dcgr[index, :, :]
    prpt = prpt[index, :, :]
    pssm = pssm[index, :, :]
    labels = np.array(labels)
    labels = labels[index]
    return dcgr, prpt, pssm, labels


def fold_i_dataset(dcgr, prpt, pssm, labels, split_seed, val_ratio):
    # Take 20% of the trainging dataset to be validation ，X_train X_val y_train y_val
    train1, val1, train_label, val_label = train_test_split(
        dcgr, labels, test_size=val_ratio, random_state=split_seed)
    train2, val2, _, _ = train_test_split(
        prpt, labels, test_size=val_ratio, random_state=split_seed)
    train3, val3, _, _ = train_test_split(
        pssm, labels, test_size=val_ratio, random_state=split_seed)

    train_data = [train1, train3, train2]
    val_data = [val1, val3, val2]
    # train_data = train3
    # val_data = val3
    # train_eval_data=[dcgr,pssm,prpt]
    # train_eval_label=np.array(labels)
    # return train_eval_data, train_eval_label

    return train_data, train_label, val_data, val_label


"""
training part
"""


def set_seed(seed=200):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, filepath_model, validation_data):
        self.filepath_model = filepath_model
        self.validation_data = validation_data

    def on_train_begin(self, logs=None):
        self.val_accs = []
        self.best_acc = 0
        self.best_loss = 0  # loss of the best f1

    def on_epoch_end(self, epoch, logs=None):
        val_predict = list(self.model.predict(self.validation_data[0]))
        val_targ = self.validation_data[1]
        val_loss = cross_entropy_error(y=val_predict, label=val_targ)
        val_acc, threshold = threshold_adaptive_acc(val_targ, val_predict)
        val_acc_5, _ = threshold_adaptive_acc(val_targ, val_predict, threshold=np.array(0.5))
        self.val_accs.append(val_acc_5)
        print('max val acc is {}'.format(max(self.val_accs)))
        if val_acc_5 > self.best_acc:
            self.model.save(self.filepath_model, overwrite=True)
            self.best_acc = val_acc_5
            self.best_loss = val_loss
            print('new best acc !!!!!')

        elif (val_acc_5 == self.best_acc):
            if (val_loss < self.best_loss):
                self.model.save(self.filepath_model, overwrite=True)
                self.best_loss = val_loss
                print('new best acc !!!!!new lowest loss')

        return


def train_val_change(train_data, val_data, pred_all_train, pred_all_val, train_label, val_label):
    sample_weight = np.array([1] * len(train_label), dtype=float)
    train_result = [0] * len(train_label)
    val_result = [0] * len(val_label)
    train_index = []
    val_index = []
    train_index_error = []
    pred_all_train = np.array(pred_all_train)
    pred_all_val = np.array(pred_all_val)
    train_temp = copy.deepcopy(train_data)
    train_temp_label = copy.deepcopy(train_label)
    pred_train = pred_all_train[-10:]
    pred_val = pred_all_val[-10:]
    pred_train_label = 1 * (pred_train >= 0.5)
    pred_val_label = 1 * (pred_val >= 0.5)
    for i in range(len(train_label)):
        train_result[i] = np.count_nonzero((pred_train_label - train_label)[:, i])
    for i in range(len(val_label)):
        val_result[i] = np.count_nonzero((pred_val_label - val_label)[:, i])
    # pick samples whose error rate is greater than 50%
    for i in range(len(val_label)):
        if val_result[i] >= 5:
            val_index.append((i, val_result[i]))
    # changed validation sample is half the above chosen sample
    # val_index_change=sorted(val_index,key=lambda x:x[1],reverse=True)[:(len(val_index)//2)]
    val_index_change = val_index[:(len(val_index) // 2)]
    val_index_change = np.array([index[0] for index in val_index_change])
    # changed training sample is those with lowest error rate in previous 10 epochs
    for i in range(len(train_label)):
        if train_result[i] == 0:
            train_index.append((i, train_result[i]))
    train_index = np.array(train_index)
    np.random.shuffle(train_index)
    train_index_change = train_index[:len(val_index_change)]
    train_index_change = np.array([index[0] for index in train_index_change])
    # changing training and validation data and label
    for i in range(3):
        train_data[i][train_index_change] = val_data[i][val_index_change]
        val_data[i][val_index_change] = train_temp[i][train_index_change]
    train_label[train_index_change] = val_label[val_index_change]
    val_label[val_index_change] = train_temp_label[train_index_change]
    for i in range(len(train_label)):
        if train_result[i] >= 5:
            train_index_error.append((i, train_result[i]))
    if len(train_index_error) != 0:
        train_index_error = np.array([index[0] for index in train_index_error])
        sample_weight[train_index_error] = 1
    sample_weight[train_index_change] = 1
    return train_data, val_data, sample_weight, train_label, val_label


def weighted_binary_crossentropy(y_true, y_pred, sample_weights):
    y_true = tf.cast(y_true, tf.float32)
    sample_weights = tf.cast(sample_weights, tf.float32)
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_loss = sample_weights * bce
    loss = K.mean(weighted_loss)
    return loss


def TVI_process(params, model, train_data, train_label, val_data, val_label, batch_size, epochs=100):
    inverse_time_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=params.lr, decay_steps=params.decay_step,
        decay_rate=params.decay_rate)
    optimizer = tf.keras.optimizers.Adam(inverse_time_decay)
    loss1 = keras.losses.binary_crossentropy
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    n_steps = len(train_data[0]) // batch_size
    pred_all_train = []
    pred_all_val = []
    model.save_weights('initialize_model_main_train.h5')

    for epoch in range(epochs):
        if epoch <= 40:
            print("\nStart of epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.
            for step in range(n_steps + 1):
                if step == n_steps:
                    x_batch_train = [train_data[0][n_steps * batch_size:], train_data[1][n_steps * batch_size:],
                                     train_data[2][n_steps * batch_size:]]
                    y_batch_train = train_label[n_steps * batch_size:]
                else:
                    x_batch_train = [train_data[0][step * batch_size:(step + 1) * batch_size],
                                     train_data[1][step * batch_size:(step + 1) * batch_size],
                                     train_data[2][step * batch_size:(step + 1) * batch_size]]
                    y_batch_train = train_label[step * batch_size:(step + 1) * batch_size]
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch_train, training=True)
                    y_pred = tf.squeeze(y_pred)
                    loss_value = loss1(y_batch_train, y_pred)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_acc_metric.update_state(y_batch_train, y_pred)
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
        if epoch > 40 and epoch <= 50:
            print("\nStart of epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.
            pred_epoch_train = []
            pred_epoch_val = []
            for step in range(n_steps + 1):

                if step == n_steps:
                    x_batch_train = [train_data[0][n_steps * batch_size:], train_data[1][n_steps * batch_size:],
                                     train_data[2][n_steps * batch_size:]]
                    x_batch_val = [val_data[0][n_steps * batch_size:], val_data[1][n_steps * batch_size:],
                                   val_data[2][n_steps * batch_size:]]
                    y_batch_train = train_label[n_steps * batch_size:]
                else:
                    x_batch_train = [train_data[0][step * batch_size:(step + 1) * batch_size],
                                     train_data[1][step * batch_size:(step + 1) * batch_size],
                                     train_data[2][step * batch_size:(step + 1) * batch_size]]
                    x_batch_val = [val_data[0][step * batch_size:(step + 1) * batch_size],
                                   val_data[1][step * batch_size:(step + 1) * batch_size],
                                   val_data[2][step * batch_size:(step + 1) * batch_size]]
                    y_batch_val = val_label[step * batch_size:(step + 1) * batch_size]
                    y_batch_train = train_label[step * batch_size:(step + 1) * batch_size]
                with tf.GradientTape() as tape:
                    y_pred_train = model(x_batch_train, training=True)
                    y_pred_train = tf.squeeze(y_pred_train)
                    pred_epoch_train.extend(y_pred_train)
                    try:
                        y_pred_val = model(x_batch_val, training=False)
                        y_pred_val = tf.squeeze(y_pred_val)
                        pred_epoch_val.extend(y_pred_val)
                    except BaseException as e:
                        pass

                    loss_value = loss1(y_batch_train, y_pred_train)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                # Update training metric.
                train_acc_metric.update_state(y_batch_train, y_pred_train)
                # val_acc_metric.update_state(y_batch_val, y_pred_val)
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            # val_acc = val_acc_metric.result()
            # print("validation acc over epoch: %.4f" % (float(val_acc),))
            pred_all_train.append(pred_epoch_train)
            pred_all_val.append(pred_epoch_val)
        if epoch == 50:
            train_data, val_data, sample_weights, train_label, val_label = train_val_change(train_data, val_data,
                                                                                            pred_all_train,
                                                                                            pred_all_val, train_label,
                                                                                            val_label)
            sample_weights = tf.constant(sample_weights)

        if epoch > 50 and epoch <= 90:
            print("\nStart of epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.
            pred_epoch_train = []
            pred_epoch_val = []
            for step in range(n_steps + 1):

                if step == n_steps:
                    x_batch_train = [train_data[0][n_steps * batch_size:], train_data[1][n_steps * batch_size:],
                                     train_data[2][n_steps * batch_size:]]
                    x_batch_val = [val_data[0][n_steps * batch_size:], val_data[1][n_steps * batch_size:],
                                   val_data[2][n_steps * batch_size:]]
                    y_batch_train = train_label[n_steps * batch_size:]
                else:
                    x_batch_train = [train_data[0][step * batch_size:(step + 1) * batch_size],
                                     train_data[1][step * batch_size:(step + 1) * batch_size],
                                     train_data[2][step * batch_size:(step + 1) * batch_size]]
                    x_batch_val = [val_data[0][step * batch_size:(step + 1) * batch_size],
                                   val_data[1][step * batch_size:(step + 1) * batch_size],
                                   val_data[2][step * batch_size:(step + 1) * batch_size]]
                    y_batch_train = train_label[step * batch_size:(step + 1) * batch_size]
                with tf.GradientTape() as tape:
                    y_pred_train = model(x_batch_train, training=True)
                    y_pred_train = tf.squeeze(y_pred_train)
                    pred_epoch_train.extend(y_pred_train)
                    try:
                        y_pred_val = model(x_batch_val, training=False)
                        y_pred_val = tf.squeeze(y_pred_val)
                        pred_epoch_val.extend(y_pred_val)
                    except BaseException as e:
                        pass
                    # loss_value = weighted_binary_crossentropy(y_batch_train, y_pred_train, sample_weights_batch)
                    loss_value = loss1(y_batch_train, y_pred_train)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                # Update training metric.
                train_acc_metric.update_state(y_batch_train, y_pred_train)
                # val_acc_metric.update_state(y_batch_val, y_pred_val)
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            pred_all_train.append(pred_epoch_train)
            pred_all_val.append(pred_epoch_val)
        if epoch == 90:
            train_data, val_data, sample_weights, train_label, val_label = train_val_change(train_data, val_data,
                                                                                            pred_all_train,
                                                                                            pred_all_val, train_label,
                                                                                            val_label)
            sample_weights = tf.constant(sample_weights)
    return train_data, val_data, sample_weights, train_label, val_label


def train(params, seed=200, shuffle_seed=13, split_seed=7, val_ratio=0.2):
    with open(f'{params.model_path}/params.pkl', 'wb') as f:
        pickle.dump(params, f)
    batch_size = params.batch_size
    global sample_weights, train_data, val_data, train_label, val_label
    set_seed(seed=seed)
    dataset_name = params.train_fasta.split('/')[-1].split('.')[0]
    print('Loading data...')
    dcgr_tr_eval, prpt_tr_eval, pssm_tr_eval, label_tr_eval = load_dataset(dataset=dataset_name,
                                                                           fasta_file=params.train_fasta,
                                                                           pssm_dir=params.train_pssm,
                                                                           use_pssm=params.use_PSSM,
                                                                           shuffle=True,
                                                                           shuffle_seed=shuffle_seed,
                                                                           mode='train')
    if params.use_test:
        dcgr_test, prpt_test, pssm_test, label_test = load_dataset(dataset=dataset_name,
                                                                   fasta_file=params.test_fasta,
                                                                   pssm_dir=params.test_pssm,
                                                                   use_pssm=params.use_PSSM,
                                                                   shuffle=True,
                                                                   shuffle_seed=shuffle_seed,
                                                                   mode='test')
        test_data = [dcgr_test, pssm_test, prpt_test]
    print('Finish loading data!')
    # all_test_labels = []

    # if not os.path.exists(params.model_path):
    #     os.makedirs(params.model_path)
    # split training, validation and testing dataset
    train_data, train_label, val_data, val_label = fold_i_dataset(
        dcgr_tr_eval, prpt_tr_eval, pssm_tr_eval, label_tr_eval, split_seed=split_seed, val_ratio=val_ratio)
    # all_test_labels.append(test_label)
    print('loding finished')

    # load the model
    model = Model(params)

    # define some hypermeters, optmimize and loss function
    if params.use_TVI:
        train_data, val_data, sample_weights, train_label, val_label = TVI_process(params, model, train_data,
                                                                                   train_label, val_data, val_label,
                                                                                   batch_size, epochs=100)

    # need to initialize the weight
    inverse_time_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=params.lr, decay_steps=params.decay_step,
        decay_rate=params.decay_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(inverse_time_decay),
                  metrics=['accuracy'],
                  # run_eagerly=True
                  )
    model.load_weights('initialize_model_main_train.h5')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'{params.model_path}/logs', write_graph=True,
                                                          write_images=True)
    print("Train...")
    filepath_model_main = f'{params.model_path}/model.h5'
    # history = model.fit(x=train_data, y=train_label,
    #                     batch_size=batch_size, epochs=params.epoch, shuffle=True,
    #                     callbacks=[Metrics(filepath_model_main, (val_data, val_label))], verbose=2)
    # history = model.train_on_batch(x=train_data, y=train_label)
    history = model.fit(x=train_data, y=train_label,
                        batch_size=batch_size, epochs=params.epoch, shuffle=True,
                        callbacks=[Metrics(filepath_model_main, (val_data, val_label)), tensorboard_callback],
                        verbose=2)
    model.load_weights(filepath_model_main)
    print('model is finished')

    # predict
    if params.use_test:
        print('Test result:')
        pred_probability = model.predict(test_data)
        pred_label = np.squeeze(1 * (pred_probability >= 0.5))
        # calculate metrics
        acc, precision, sensitivity, specificity, f1_score, MCC = calculate_performace(label_test, pred_probability,
                                                                                       threshold=0.5)
        testOutpath = f'./{params.model_path}/test/'
        if not os.path.exists(testOutpath):
            os.makedirs(testOutpath)
        with open(f'{testOutpath}/test_label.pkl', 'wb') as f:
            pickle.dump(label_test, f)
        with open(f'{testOutpath}/test_pred.pkl', 'wb') as f:
            pickle.dump(pred_probability, f)
        with open(f'{testOutpath}/pred_label.pkl', 'wb') as f:
            pickle.dump(pred_label, f)
        draw_ROC(label_test, pred_probability, f'{testOutpath}/ROC.png')

    # performance = [acc, sensitivity, specificity, precision, f1_score, MCC]

    # print('---' * 50)
    # draw picture
    # all_t_labels = []
    # for testfold in all_test_labels:
    #     for label in testfold:
    #         all_t_labels.append(label)
    # all_t_labels = np.array(all_t_labels)
    # for predfold in all_pred_probability:
    #     for prob in predfold:
    #         all_p_probability.append(prob)
    # all_p_probability = np.array(all_p_probability)
    # draw_ROC(all_t_labels, all_p_probability)

    # print(performance)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--type", "-type", dest="type", type=str,
                        help="The type of training model is antimicrobial peptide or anticancer peptide, 'ACP' or 'AMP'.")
    parser.add_argument("--use_PSSM", "-use_PSSM", dest='use_PSSM', type=lambda x: bool(distutils.util.strtobool(x)),
                        help='Whether to use the pssm feature.')
    parser.add_argument("--use_TVI", "-use_TVI", dest='use_TVI', type=lambda x: bool(distutils.util.strtobool(x)),
                        help='Whether to use TVI process to select the more appropriate training and validation set.')
    parser.add_argument("--train_fasta", "-train_fasta", dest='train_fasta', type=str,
                        help='The path of the train FASTA file.')
    parser.add_argument("--train_pssm", "-train_pssm", dest='train_pssm', type=str,
                        help='The path of the train PSSM files.')
    parser.add_argument("--use_test", "-use_test", dest='use_test', type=lambda x: bool(distutils.util.strtobool(x)),
                        help='Whether to use an independent test set to evaluate the trained model.')
    parser.add_argument("--test_fasta", "-test_fasta", dest='test_fasta', type=str,
                        help='The path of the test FASTA file.')
    parser.add_argument("--test_pssm", "-test_pssm", dest='test_pssm', type=str,
                        help='The path of the test PSSM files.')
    # parser.add_argument("--train_fasta", "-train_fasta", dest='train_fasta', type=str, default='./data/AMPlify_train.txt',
    #                     help='The path of the train FASTA file.')
    # parser.add_argument("--train_pssm", "-train_pssm", dest='train_pssm', type=str, default='./data/pssm_AMPlify_train/',
    #                     help='The path of the train PSSM files.')
    # parser.add_argument("--test_fasta", "-test_fasta", dest='test_fasta', type=str, default='./data/AMPlifytest.txt',
    #                     help='The path of the test FASTA file.')
    # parser.add_argument("--test_pssm", "-test_pssm", dest='test_pssm', type=str, default='./data/pssm_AMPlify_test/',
    #                     help='The path of the test PSSM files.')
    parser.add_argument("--dense_unit_dcgr", "-dense_unit_dcgr", dest="dense_unit_dcgr", type=int, default=350,
                        help='The dimension of features learned from DCGR feature matrix.')
    parser.add_argument("--dense_unit_pssm", "-dense_unit_pssm", dest='dense_unit_pssm', type=int, default=120,
                        help='The dimension of features learned from PSSM feature.')
    parser.add_argument("--dense_unit_all", "-dense_unit_all", dest='dense_unit_all', type=int, default=200,
                        help='The dimension of output of the concatenation of the results of the three stages '
                             'containing the fingerprint, evolutionary and physicochemical property information.')
    parser.add_argument("--dff", "-dff", dest='dff', type=int, default=6,
                        help='The dimension of hidden units in the middle linear layer of a point-wise feed-forward network in the Encoder of the third stage.')
    parser.add_argument("--head", "-head", dest='head', type=int, default=2,
                        help='The number of heads in the multi-head self-attention mechanism in the Encoder of the third stage.')
    # parser.add_argument("--activation_type", "-activation_type", dest='activation_type', type=str, default='relu',
    #                     help='The type of activation function that applies to the output of the concatenation of the results of the three stages.')
    parser.add_argument("--d1", "-d1", dest='d1', type=float, default=0.5667255416898414,
                        help='The probability of randomly dropping input units during each update in the training period of the first stage.')
    parser.add_argument("--d2", "-d2", dest='d2', type=float, default=0.5386135610624118,
                        help='The probability of randomly dropping input units during each update in the training period after the concatenation of the results of the three stages.')
    parser.add_argument("--lr", "-lr", dest='lr', type=float, default=0.0004,
                        help='The initial learning rate used to control the step size of parameter updates in each update.')
    parser.add_argument("--decay_rate", "-decay_rate", dest='decay_rate', type=float, default=0.9509473845015629,
                        help='The percentage decline of learning rate in each step of the dynamic learning rate adjustment.')
    parser.add_argument("--decay_step", "-decay_step", dest='decay_step', type=int, default=500,
                        help='The number of steps of learning rate decay in each step of the dynamic learning rate adjustment.')
    parser.add_argument("--batch_size", "-batch_size", dest='batch_size', type=int, default=64,
                        help='Batch size for training deep model.')
    # parser.add_argument("--ratio", "-ratio", dest='ratio', type=int, default=8,
    #                     help='The ratio adjusting the hidden layer size of the channel attention mechanism in the first stage.')
    parser.add_argument("--epoch", "-epoch", dest='epoch', type=int, default=50, help='Training epochs.')

    return parser.parse_args()


def checkargs(args):
    if args.type is None or args.use_PSSM is None or args.use_test is None or args.use_TVI is None:
        print('ERROR: please input the necessary parameters!')
        raise ValueError

    if args.type not in ['ACP', 'AMP']:
        print(f'ERROR: type "{args.type}" is not supported by TriNet!')
        raise ValueError

    if args.use_PSSM:
        if args.train_pssm is None:
            print('ERROR: please input the paths of train and test pssm files!')
            raise ValueError

    if args.use_test:
        if args.test_fasta is None:
            print('ERROR: please input the path of test fasta file!')
            raise ValueError
        if args.use_PSSM and args.test_pssm is None:
            print('ERROR: please input the path of test pssm files!')
            raise ValueError
    else:
        print('No test independent dataset to evaluate the trained model!!!')

    # if not args.activation_type in ['relu', 'sigmoid', 'tanh']:
    #     print('Please pass in the correct type of the activation function!!!')

    return


class Config():
    def __init__(self, args):

        # self.ligand = 'P' + args.ligand if args.ligand != 'HEME' else 'PHEM'
        # self.Dataset_dir = f'./data/'
        self.type = args.type
        self.use_PSSM = args.use_PSSM
        self.use_TVI = args.use_TVI
        self.use_test = args.use_test
        self.train_fasta = args.train_fasta
        self.train_pssm = args.train_pssm
        self.test_fasta = args.test_fasta
        self.test_pssm = args.test_pssm
        # self.activation_type = args.activation_type
        self.batch_size = args.batch_size
        self.d1 = args.d1
        self.d2 = args.d2
        self.decay_rate = args.decay_rate
        self.decay_step = args.decay_step
        self.dense_unit_dcgr = args.dense_unit_dcgr
        self.dense_unit_all = args.dense_unit_all
        self.dense_unit_pssm = args.dense_unit_pssm
        self.dff = args.dff
        self.epoch = args.epoch
        self.head = args.head
        self.lr = args.lr
        # self.ratio = args.ratio
        self.checkpoints = f'./checkpoints/'
        self.model_time = None
        # self.model_time = '2023-04-13-10:23:05'
        self.train = True
        if self.model_time is not None:
            # self.model_path = self.Dataset_dir + '/checkpoints/' + self.model_time
            self.model_path = self.checkpoints + self.model_time
        else:
            localtime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
            # self.model_path = self.Dataset_dir + '/checkpoints/' + localtime
            self.model_path = self.checkpoints + localtime
            os.makedirs(self.model_path)

    def print_config(self):
        for name, value in vars(self).items():
            print('{} = {}'.format(name, value))



class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


if __name__ == '__main__':
    arguments = parse_args()
    checkargs(arguments)
    params = Config(arguments)
    sys.stdout = Logger(params.model_path + '/training.log')
    params.print_config()
    train(params)
    sys.stdout.log.close()
