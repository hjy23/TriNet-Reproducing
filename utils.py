import numpy as np
import random
import os
from matplotlib import pyplot as plt
import prettytable as pt
from sklearn.metrics import roc_curve, auc, f1_score


def load_amp(dir):
    """
    Load peptide sequences and labels from AMPXXX.txt
    :param dir:
        directory of AMPXXX.txt
    :return:
        amps:   peptide sequences
        labels: peptide labels
    """
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
    """
    Load peptide sequences and labels from ACPXXX.txt
    :param dir:
        directory of ACPXXX.txt
    :return:
        acps:   peptide sequences
        labels: peptide labels
    """
    acps = []
    labels = []
    with open(dir) as file:
        for line in file:
            if line[0] == '>':
                labels.append(int(line[-2]))
            else:
                acps.append(line.rstrip())
    return acps, labels


def calculate_performace(true_label, pred_label, threshold):
    pred_label = np.squeeze(1 * (pred_label >= threshold))

    tp = 0  # true,   pred:positive   1判断成1
    fp = 0  # false,  pred:positive   0判断成1(误报)
    tn = 0  # true,   pred:negative   0判断成0
    fn = 0  # false,  pred:negative   1判断成0(错报)
    for index in range(len(true_label)):
        if true_label[index] == 1:
            if true_label[index] == pred_label[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if true_label[index] == pred_label[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / len(true_label)  # 准确率：所有的中分对的
    precision = float(tp) / (tp + fp)  # 精确度：越策
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    f1_score = 2 * precision * sensitivity / (precision + sensitivity)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    test_matrices = [acc, precision, sensitivity, specificity, f1_score, MCC]
    tb = pt.PrettyTable()
    tb.field_names = ['Dataset', 'th', 'ACC', 'Pre', 'Sen', 'Spe', 'F1', 'MCC']
    row_list = ['test', '0.5']
    for i in range(len(tb.field_names) - 2):
        row_list.append('{:.3f}'.format(test_matrices[i]))
    tb.add_row(row_list)
    print(tb)
    return acc, precision, sensitivity, specificity, f1_score, MCC


def draw_ROC(labels, probality, save_path):
    def plot_roc_curve(labels, probality, auc_tag=True):
        # fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
        labels = np.squeeze(labels)
        probality = np.squeeze(probality)
        fpr, tpr, thresholds = roc_curve(labels, probality)  # probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        if auc_tag:
            rects1 = plt.plot(fpr, tpr, label='proposed method' + ' (AUC=%6.3f) ' % roc_auc)
        else:
            rects1 = plt.plot(fpr, tpr, label='proposed method')

    plt.figure(1)
    plot_roc_curve(labels, probality)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(save_path)


def threshold_adaptive_acc(val_targ, val_pred, threshold=np.arange(0.3, 0.7, 0.02)):
    num_pred = len(val_pred)

    val_pred = np.array(val_pred)
    val_pred = val_pred.reshape((1, -1))
    threshold = threshold.reshape((-1, 1))

    labels = 1 * (val_pred >= threshold)
    acc = np.sum((labels - val_targ) == 0, axis=1) / num_pred
    return max(acc), threshold[np.argmax(acc)]


def cross_entropy_error(y, label):
    y = np.array(y)
    y = 1 * (y >= 0.5)
    label = np.array(label)

    batch = y.shape[0]
    y = y.reshape(batch, 1)
    y = np.concatenate((1 - y, y), axis=1)

    err = -np.sum(np.log(y[np.arange(batch), label] + 1e-7)) / batch
    return err
