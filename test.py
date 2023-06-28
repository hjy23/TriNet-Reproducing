import sys
import argparse
import numpy as np
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from scipy.spatial.distance import pdist, squareform
# from model.Model_amp import *
# from model.Model_acp import *
from Model import Model
from math import pi
import pandas as pd


def load_amp(dir):
    amps = []
    labels = []
    with open(dir) as file:
        for line in file:
            if line[0] == '>':
                labels.append(int(line[-2]))
            else:
                amps.append(line.rstrip())
    return amps, labels


def load_acp(dir):
    acps = []
    labels = []
    with open(dir) as file:
        for line in file:
            if line[0] == '>':
                labels.append(int(line[-2]))
            else:
                acps.append(line.rstrip())
    return acps, labels


def load_properties158(dir):
    """
    Load 158 physicochemical properties,
    each of which is a string of 20 characters.
    :param dir:
        the directory of properties158.txt
    :return:
        prt: a list of 158 properties
    """
    properties = []
    with open(dir) as file:
        for line in file:
            properties.append(line.rstrip())
    return properties


def coordinate(R=1):
    """
    Twenty points were uniformly taken in a circle of radius R,
    corresponding to 20 amino acids in physicochemical properties
    :param R: radius of the circle
    :return:
        points: 20 points at the circle
    """
    angle = np.arange(1, 21) * 2 * pi / 20
    points = np.vstack((R * np.cos(angle), R * np.sin(angle)))
    points = points.transpose((1, 0))
    return points


def graph_representation(acp, properties, points):
    """
    One property correspond to a graph representation, 8 eigenvalues.
    Get 158 graph representations of an acp.

    :param acp:         an acp sequence
    :param properties:  158 physicochemical properties
    :param points:      20 evenly scattered points on the circle
    :return:            The 158 * 8 eigenmatrix of a sequence
    """
    features = np.zeros((158, 8), dtype=float)
    for index, property in enumerate(properties):
        p = [[0, 0]]
        for i, char in enumerate(acp):
            pos = property.find(char)
            p.append([(points[pos][0] + p[i][0]) / 2, (points[pos][1] + p[i][1]) / 2])
        del p[0]
        features[index, :] = parts_8(p)
    return features


def parts_8(pts):
    """
    Divide pts into P0-P7 regions.
    Compute 8 eigenvalue of each region.

    :param pts: A list of acp length * 2
    :return:    8 eigenvalue
    """
    parts = [[], [], [], [], [], [], [], []]
    for pt in pts:  # 遍历每个点，吧它们分到8个区域中去
        # P0-P3
        if pt[0] > 0 and pt[1] >= 0:
            parts[0].append(pt)
        elif pt[0] <= 0 and pt[1] > 0:
            parts[1].append(pt)
        elif pt[0] < 0 and pt[1] <= 0:
            parts[2].append(pt)
        elif pt[0] >= 0 and pt[1] < 0:
            parts[3].append(pt)
        # P4-P7
        if pt[1] <= pt[0] and pt[1] > -pt[0]:
            parts[4].append(pt)
        elif pt[1] > pt[0] and pt[1] >= -pt[0]:
            parts[5].append(pt)
        elif pt[1] >= pt[0] and pt[1] < -pt[0]:
            parts[6].append(pt)
        elif pt[1] < pt[0] and pt[1] <= -pt[0]:
            parts[7].append(pt)
    # calculate eigenvalue of of each region
    feature = np.zeros((8), dtype=float)
    for i in range(len(parts)):
        part = np.array(parts[i][:])
        part = part.reshape(-1, 2)
        d = pdist(part)
        D = squareform(d)
        if len(D) != 0:
            e, _ = np.linalg.eig(D)
            feature[i] = max(e)
    return feature


def get_DCGR_features(dir):
    '''
        Get DCGR features of each sequence according to acp dataset and 158 properties.
        :param dir: directory of acp dataset
        :return:    DCGR features of size acp length * 158 * 8
    '''
    acps, labels = load_acp(dir)

    properties = load_properties158(sys.path[0] + r'/data/properties158.txt')
    points = coordinate()

    features = []
    for i, acp in enumerate(acps):
        feature = graph_representation(acp, properties, points)
        features.append(feature)
    return features


def get_acps_len(dir):
    acps, labels = load_acp(dir)
    acps_len = []
    for acp in acps:
        acps_len.append(len(acp))
    return acps_len


def get_amps_len(dir):
    amps, labels = load_amp(dir)
    amps_len = []
    for amp in amps:
        amps_len.append(len(amp))
    return amps_len


def get_PSSM_features(acp_dir, pssm_dir, theta=50):
    '''
      Get PSSM features of each sequence according to directory acp_dir and pssm_dir.
      While some sequences may not have PSSM files, their features will be replaced by 0 matrices.

      :param acp_dir:     directory of acp dataset
      :param pssm_dir:    directory of generated PSSM features
      :param theta:       intercept the first theta amino acids of the sequence
      :return:            PSSM features of size acp length * theta * 20
      '''
    each_acp_len = get_acps_len(acp_dir)
    pssms = []
    with open(acp_dir, 'r') as f:
        f = f.readlines()
        list = []
        for line in f:
            if line[0] == '>':
                list.append(line)
    num = len(list) + 1
    for i in range(1, num):
        dir_i = pssm_dir + str(i) + '.txt'
        pssm = np.zeros((theta, 20))
        # Judge whether the i-th sequence has a PSSM file
        if os.path.exists(dir_i):  # 如果存在
            pssm = np.zeros((theta, 20))
            # start to read PSSM feature
            f = open(dir_i)
            line = f.readline()
            line = f.readline()
            line = f.readline()

            for j in range(min(each_acp_len[i - 1], theta)):
                line = f.readline()
                line = line[9:]
                s = line.split()
                s = s[:20]
                try:
                    s = [int(x) for x in s]
                except BaseException as e:
                    print(e, dir_i)
                # s = [int(x) for x in s]

                s = np.array(s)
                try:
                    pssm[j] = s
                except Exception as e:
                    print(e)
        pssms.append(pssm)
    return pssms


element = 'ALRKNMDFCPQSETGWHYIV'


def get_PSSM_features_fast(acp_dir):
    F = open(sys.path[0] + r'/model/blosum_dict.pkl', 'rb')
    pssm_dict = pickle.load(F)
    pssms = []
    with open(acp_dir, 'r') as f:
        f = f.readlines()
        list = []
        for line in f:
            if line[0] == '>':
                pass
            else:
                list.append(line.rstrip())
    for amp in list:
        pssm = np.zeros((50, 20))
        for index, amino in enumerate(amp):
            if index < 50:
                pssm[index] = pssm_dict[amino]
        pssm = np.array(pssm)
        pssms.append(pssm)
    pssm_final = np.array(pssms)[:, :50]
    return pssm_final


def load_amino_acids_properties(dir=sys.path[0] + r'/data/properties_of_amino_acids.xlsx'):
    df = pd.read_excel(dir)
    data = df.values
    return data[:20, 1:]


def std_properties(properties):
    # standardize 158 prperties
    for i in range(20):
        property = properties[:, i]
        nrm = np.linalg.norm(property)
        properties[:, i] = property / nrm
    return properties


def get_properties_features(dir, theta=50):
    '''
    Replace each amino acid with the corresponding 20-values' property.

    :param dir:     directory of acp dataset
    :param theta:   intercept the first theta amino acids of the sequence
    :return:        property features of size acp length * theta * 8
    '''
    properties = load_amino_acids_properties()
    properties = std_properties(properties)
    acps, labels = load_acp(dir)
    features = []
    for acp in acps:
        acp_len = len(acp)
        feature = np.zeros((theta, 8), dtype=float)
        for i in range(min(acp_len, theta)):
            char = acp[i]
            index = element.find(char)
            feature[i, :] = properties[index, 0:8]
        features.append(feature)
    return features


def load_dataset_amp(dir, pssm_dir):
    dcgr = get_DCGR_features(dir)  # feature get from DCGR
    pssm = get_PSSM_features(dir, pssm_dir)  # feature get from PRPT
    prpt = get_properties_features(dir)  # feature get from PSSM
    amps, _ = load_amp(dir)
    test_data = [np.array(dcgr), np.array(pssm), np.array(prpt)]
    return test_data, amps


"""
testing part
"""


def amp_test(model_path, dir, pssm_dir, output_dir):
    test_data, amp = load_dataset_amp(dir, pssm_dir)
    with open(f'{model_path}/params.pkl', 'rb') as f:
        params = pickle.load(f)
    model = Model(params)
    filepath = f'{model_path}/model.h5'
    model.load_weights(filepath)
    pred_probability = np.squeeze(model.predict(test_data))
    pred_label = np.squeeze(1 * (pred_probability >= 0.5))
    D = {'sequence': amp, 'probability': pred_probability, 'AMP or non-AMP': pred_label}
    D = pd.DataFrame(D)
    D.to_csv(output_dir, index=False)
    return D


def load_dataset_acp(dir, pssm_dir):
    dcgr = get_DCGR_features(dir)  # feature get from DCGR
    pssm = get_PSSM_features(dir, pssm_dir)  # feature get from PRPT
    prpt = get_properties_features(dir)  # feature get from PSSM
    acps, _ = load_acp(dir)
    test_data = [np.array(dcgr), np.array(pssm), np.array(prpt)]
    return test_data, acps


def load_dataset_fast(dir):
    dcgr = get_DCGR_features(dir)  # feature get from DCGR
    pssm = get_PSSM_features_fast(dir)  # feature get from PRPT
    prpt = get_properties_features(dir)  # feature get from PSSM
    acps, _ = load_acp(dir)
    test_data = [np.array(dcgr), np.array(pssm), np.array(prpt)]
    return test_data, acps

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
"""
testing part
"""


def acp_test(model_path, dir, pssm_dir, output_dir):
    with open(f'{model_path}/params.pkl', 'rb') as f:
        params = pickle.load(f)
    test_data, acp = load_dataset_acp(dir, pssm_dir)
    model = Model(params)
    filepath = f'{model_path}/model.h5'
    model.load_weights(filepath)
    # debug_ = model.call(test_data)
    pred_probability = np.squeeze(model.predict(test_data))
    pred_label = np.squeeze(1 * (pred_probability >= 0.5))
    D = {'sequence': acp, 'probability': pred_probability, 'ACP or non-ACP': pred_label}
    D = pd.DataFrame(D)
    D.to_csv(output_dir, index=False)


def acp_test_fast(model_path, dir, output_dir):
    test_data, acp = load_dataset_fast(dir)
    with open(f'{model_path}/params.pkl', 'rb') as f:
        params = pickle.load(f)
    model = Model(params)
    filepath = f'{model_path}/model.h5'
    model.load_weights(filepath)
    pred_probability = np.squeeze(model.predict(test_data))
    pred_label = np.squeeze(1 * (pred_probability >= 0.5))
    D = {'sequence': acp, 'probability': pred_probability, 'ACP or non-ACP': pred_label}
    D = pd.DataFrame(D)
    D.to_csv(output_dir, index=False)


def amp_test_fast(model_path, dir, output_dir):
    test_data, amp = load_dataset_fast(dir)
    with open(f'{model_path}/params.pkl', 'rb') as f:
        params = pickle.load(f)
    model = Model(params)
    filepath = f'{model_path}/model.h5'
    model.load_weights(filepath)
    pred_probability = np.squeeze(model.predict(test_data))
    pred_label = np.squeeze(1 * (pred_probability >= 0.5))
    D = {'sequence': amp, 'probability': pred_probability, 'AMP or non-AMP': pred_label}
    D = pd.DataFrame(D)
    D.to_csv(output_dir, index=False)


if __name__ == '__main__':
    curdir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--PSSM_file', '-p', default='./pssm_acp_example/', help='path of PSSM/PSSM_file', type=str)
    parser.add_argument('--sequence_file', '-s', default='./ACP_example.fasta', help='path of sequence file', type=str)
    parser.add_argument('--output', '-o', default='./outputacp.csv',
                        help='path of Trinet result,defaut with path of current path/output ', type=str)
    parser.add_argument('--operation_mode', '-mode', default='sc',
                        help='f for fast mode and s for standard mode, c for anticancer prediction and m for antimicrobial prediction',
                        type=str)
    parser.add_argument('--model_path', '-mp',
                        help='The path of the model to be loaded',
                        type=str)

    args = parser.parse_args()
    operation_mode = args.operation_mode
    input_file = os.path.join(curdir, args.sequence_file)
    output = os.path.join(curdir, args.output)
    if args.model_path == None:
        print('Please pass in the model path where you want to use.')
        exit(-1)
    if operation_mode == 'sc' or operation_mode == 'sm':
        pssm_path = os.path.join(curdir, args.PSSM_file)
        if pssm_path[-1] != '/':
            pssm_path = pssm_path + '/'
        try:
            list = os.listdir(pssm_path)
        except Exception as e:
            print('path for pssm not exists,you path is {},please use the correct input form of PSSM files'.format(
                pssm_path))
            exit(-1)

        if input_file == None or pssm_path == None or operation_mode == None:
            print('Wrong input, please check your input!')
        else:
            if operation_mode == 'sc':
                acp_test(model_path=args.model_path, dir=input_file, pssm_dir=pssm_path, output_dir=output)
            if operation_mode == 'sm':
                amp_test(model_path=args.model_path, dir=input_file, pssm_dir=pssm_path, output_dir=output)
            if operation_mode == 'fc':
                acp_test_fast(model_path=args.model_path, dir=input_file, output_dir=output)
            if operation_mode == 'fm':
                amp_test_fast(model_path=args.model_path, dir=input_file, output_dir=output)
            print('finished')
    else:
        if input_file == None or operation_mode == None:
            print('Wrong input, please check your input!')
        else:

            if operation_mode == 'fc':
                acp_test_fast(dir=input_file, output_dir=output)
            if operation_mode == 'fm':
                amp_test_fast(dir=input_file, output_dir=output)
            print('finished')
