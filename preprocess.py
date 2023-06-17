import pickle

import numpy as np
from math import pi, cos, sin
from scipy.spatial.distance import pdist, squareform
import csv
import os
import pandas as pd


def load_fasta(dir):
    """
    Load peptide sequences and labels from ACPXXX.txt
    :param dir:
        directory of ACPXXX.txt
    :return:
        acps:   peptide sequences
        labels: peptide labels
    """
    names = []
    sequences = []
    labels = []
    with open(dir) as file:
        for line in file:
            if line[0] == '>':
                labels.append(int(line[-2]))
                names.append(line.split()[0][1:])
            else:
                sequences.append(line.rstrip())
    return names, sequences, labels


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
    # print(features)
    return features


def get_DCGR_features(sequences, out_path):
    '''
        Get DCGR features of each sequence according to acp dataset and 158 properties.
        :param dir: directory of acp dataset
        :return:    DCGR features of size acp length * 158 * 8
    '''
    if not os.path.exists(out_path):
        # sequences, labels = load_fasta(input_path)
        properties = load_properties158(f'./data/properties158.txt')
        points = coordinate()

        features = []
        for i, seq in enumerate(sequences):
            feature = graph_representation(seq, properties, points)
            features.append(feature)
        features = np.array(features)
        with open(out_path, 'wb') as f:
            pickle.dump(features, f)
    else:
        with open(out_path, 'rb') as f:
            features = pickle.load(f)

    return features


def load_amino_acids_properties(dir=r'./data/properties_of_amino_acids.xlsx'):
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


def get_properties_features(sequences, out_path, theta=50):
    '''
    Replace each amino acid with the corresponding 20-values' property.

    :param dir:     directory of acp dataset
    :param theta:   intercept the first theta amino acids of the sequence
    :return:        property features of size acp length * theta * 8
    '''
    if not os.path.exists(out_path):
        properties = load_amino_acids_properties()
        properties = std_properties(properties)
        # sequences, labels = load_fasta(input_path)
        element = 'ALRKNMDFCPQSETGWHYIV'
        features = []
        for seq in sequences:
            sequence_len = len(seq)
            feature = np.zeros((theta, 8), dtype=float)
            for i in range(min(sequence_len, theta)):
                char = seq[i]
                index = element.find(char)
                feature[i, :] = properties[index, 0:8]
            features.append(feature)
        features = np.array(features)
        with open(out_path, 'wb') as f:
            pickle.dump(features, f)
    else:
        with open(out_path, 'rb') as f:
            features = pickle.load(f)

    return features


def get_seqs_len(sequences):
    # sequences, labels = load_fasta(dir)
    seqs_len = []
    for seq in sequences:
        seqs_len.append(len(seq))
    return seqs_len


def get_PSSM_features(sequences, out_path, pssm_dir, theta=50):
    '''
      Get PSSM features of each sequence according to directory acp_dir and pssm_dir.
      While some sequences may not have PSSM files, their features will be replaced by 0 matrices.

      :param acp_dir:     directory of acp dataset
      :param pssm_dir:    directory of generated PSSM features
      :param theta:       intercept the first theta amino acids of the sequence
      :return:            PSSM features of size acp length * theta * 20
      '''

    if not os.path.exists(out_path):
        each_seq_len = get_seqs_len(sequences)
        features = []
        num = len(each_seq_len)
        for i in range(num):
            dir_i = pssm_dir + str(i+1) + '.txt'
            feature = np.zeros((theta, 20))
            # Judge whether the i-th sequence has a PSSM file
            if os.path.exists(dir_i):
                feature = np.zeros((theta, 20))
                f = open(dir_i)
                line = f.readline()
                line = f.readline()
                line = f.readline()
                for j in range(min(each_seq_len[i], theta)):
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
                        feature[j] = s
                    except Exception as e:
                        print(e)
            else:
                print(f'PSSM profile not exist!!!')
            features.append(feature)
        features = np.array(features)
        with open(out_path, 'wb') as f:
            pickle.dump(features, f)
    else:
        with open(out_path, 'rb') as f:
            features = pickle.load(f)

    return features

def get_blosum_features(sequences, out_path, blosum_path, theta=50):
    '''
      Get PSSM features of each sequence according to directory acp_dir and pssm_dir.
      While some sequences may not have PSSM files, their features will be replaced by 0 matrices.

      :param acp_dir:     directory of acp dataset
      :param pssm_dir:    directory of generated PSSM features
      :param theta:       intercept the first theta amino acids of the sequence
      :return:            PSSM features of size acp length * theta * 20
      '''
    if not os.path.exists(out_path):
        with open(f'{blosum_path}', 'rb') as f:
            blosum_dict = pickle.load(f)
        features = []
        for seq in sequences:
            feature = np.zeros((theta, 20))
            for index, amino in enumerate(seq):
                feature[index] = blosum_dict[amino]
                if index == theta:
                    break
            feature = np.array(feature)
            features.append(feature)
        features = np.array(features)[:, :theta]
        with open(out_path, 'wb') as f:
            pickle.dump(features, f)
    else:
        with open(out_path, 'rb') as f:
            features = pickle.load(f)
    return features
