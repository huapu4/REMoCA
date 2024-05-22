# -*- encoding: utf-8 -*-
'''
@Time    : 2023/5/2
@Author  : Lin Zhenzhe, Zhang Shuyi
'''
import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes, normalize=False, save_name=None, cmap=plt.cm.Blues):
    """
    # draw confusion matrix
    - cm : index of the confusion matrix
    - classes : name of classes
    - normalize : True for showing percent, False for showing value
    """
    plt.rcParams['font.family'] = 'Microsoft Yahei'
    plt.clf()
    plt.figure(figsize=(10, 8))
    proportion = []
    for i in cm:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []
    p_cm = []
    for i in proportion:
        pt = "%.1f%%" % (i * 100)
        pshow.append(pt)
        p_cm.append(i)
    pshow = np.array(pshow).reshape(2, 2)
    p_cm = np.array(p_cm).reshape(2, 2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(p_cm, interpolation='nearest', cmap=cmap, aspect='auto')
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    # 字体   mri :29    moca:
    plt.xticks(tick_marks, classes, fontsize=29)
    plt.yticks(tick_marks, classes, fontsize=29, rotation=45)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    # 字体   mri :48    moca:
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i - 0.15, format(cm[i, j], fmt),
                 horizontalalignment="center", va="center",
                 color="white" if p_cm[i, j] > 0.5 else "black", fontsize=48)

        plt.text(j, i + 0.15, pshow[i, j],
                 horizontalalignment="center", va="center",
                 color="white" if p_cm[i, j] > 0.5 else "black", fontsize=48)
    # 字体   mri :32    moca:
    plt.ylabel('Ground truth', fontdict={'size': 32})
    plt.xlabel('Prediction', fontdict={'size': 32})
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(0.5, 1.12)
    plt.tight_layout()
    plt.savefig(save_name + '.svg', dpi=1000, format='svg')


if __name__ == '__main__':
    cnf_matrix = np.array([[454, 85],
                           [35, 461]])
    attack_types = ['Correct', 'Incorrect']
    plot_confusion_matrix(cm=cnf_matrix, classes=attack_types, save_name='05.竖直跟随矩阵')

    # cnf_matrix = np.array([[28, 8],
    #                        [26, 56]])
    # attack_types = ['MoCA socre≥23', 'MoCA socre<23']
    # plot_confusion_matrix(cm=cnf_matrix, classes=attack_types, save_name='moca23_exter' + '_hybrid data')

    # cnf_matrix = np.array([[15, 0],
    #                        [29, 71]])
    # attack_types = ['MoCA socre≥26', 'MoCA socre<26']
    # # plot_confusion_matrix(cm=cnf_matrix, classes=attack_types, save_name='moca26_exter' + '_hybrid data')
    # plot_confusion_matrix(cm=cnf_matrix, classes=attack_types, save_name='I.Hybrid MLM')


    # cnf_matrix = np.array([[13, 7],
    #                        [9, 14]])
    # attack_types = ['Fazekas scale>1', 'Fezakas scale≤1']
    # plot_confusion_matrix(cm=cnf_matrix, classes=attack_types, save_name='mri1_exter' + '_hybrid data')

    # cnf_matrix = np.array([[14, 12],
    #                        [6, 9]])
    # attack_types = ['Fazekas scale>2', 'Fezakas scale≤2']
    # plot_confusion_matrix(cm=cnf_matrix, classes=attack_types, save_name='I.Hybrid MLM')
