# -*- encoding: utf-8 -*-
'''
@Time    : 2023/5/2
@Author  : Lin Zhenzhe, Zhang Shuyi
'''

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib


def draw_roc_curve(y, prob, pic_name='ROC'):
    # input prediction and probability to generate the ROC and calculate the AUC
    plt.style.use('seaborn-darkgrid')
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    lw = 2
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=lw, label='Area under curve = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(pic_name, dpi=600)


def draw_bar(names: list, values: list):
    matplotlib.rc("font", family='MicroSoft YaHei', weight="bold")
    # plt.style.use('_mpl-gallery')
    # plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(20, 20))

    # plot
    fig, ax = plt.subplots(tight_layout=True)

    ax.bar(names, values, width=1, edgecolor="white", linewidth=0.7)
    plt.xticks(rotation=45)

    ax.set(ylim=(-1, 1))

    plt.title('Feature Coefficients')
    plt.tight_layout()
    plt.show()
    # plt.savefig('./relevance0126/1-Month Relevance', dpi=600)
