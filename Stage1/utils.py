# -*- coding: utf-8 -*-
"""
任务：画roc、excel记录实验结果
日期：2021/10/19 
by Paul
"""
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc
import openpyxl
import numpy as np

def cal_confidence_interval(value, n, inter_value=0.95):
    if inter_value == 0.9:
        const = 1.64
    elif inter_value == 0.95:
        const = 1.96
    elif inter_value == 0.98:
        const = 2.33
    elif inter_value == 0.99:
        const = 2.58

    confidence_interval_upper = value + const * np.sqrt((value * (1 - value)) / n)
    confidence_interval_lower = value - const * np.sqrt((value * (1 - value)) / n)
    if confidence_interval_lower < 0:
        confidence_interval_lower = 0
    if confidence_interval_upper > 1:
        confidence_interval_upper = 1
    if confidence_interval_upper == 1:
        ci = '({}-{})'.format('%.3f' % confidence_interval_lower, '1.000')
    else:
        ci = '({}-{})'.format('%.3f' % confidence_interval_lower, '%.3f' % confidence_interval_upper)
    return ci


matplotlib.use('Agg')

def output_class(y_true, score, class_index):
    single_true, single_score = [], []
    for i in range(len(y_true)):
        if y_true[i] == class_index:
            single_true.append(1)
            single_score.append(float(score[i][y_true[i]]))
        else:
            single_true.append(0)
            single_score.append(1 - float(score[i][y_true[i]]))
    fpr, tpr, threshold = roc_curve(single_true, single_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)
    return [fpr, tpr, roc_auc]

def draw_curve(index_list, name_list):
    lw = 1
    for i, item in enumerate(index_list):
        fpr, tpr, roc_auc = iter(item)
        line_name = name_list[i]
        plt.plot(fpr, tpr, lw=lw, label=line_name + ' AUC = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('ROC')
        plt.legend(loc="lower right")
    plt.savefig('roc', dpi=600)

def draw_auc_curve(y, prob,pic_name):
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

def compare_model(value1, value2, cp_value):
    if value1 * value2 > cp_value:
        return True, value1 * value2
    else:
        return False, cp_value

def list2excel(excel_name, data):
    '''
    :param excel_name: excel名字
    :param data: 输入的数据，列表嵌套 [[1,2,3],[4,5,6]]
    '''
    excel_name += '.xlsx'
    wb = openpyxl.Workbook()
    ws = wb.active
    for column in range(0, len(data)):
        for row, index in enumerate(data[column]):
            ws.cell(row + 1, column + 1).value = data[column][row]
    wb.save(excel_name)

def cal_auc(y_true, proba):
    score = []
    for num, i in enumerate(y_true):
        if i == 0:
            score.append(1 - proba[num][i])
        else:
            score.append(proba[num][i])
    fpr, tpr, threshold = roc_curve(y_true, score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)
    return [list(fpr), list(tpr), roc_auc]

def draw_multi_curve(index_list, name_list, color_list, pic_name, len_list):
    lw = 4
    plt.figure(figsize=(8, 8))
    for i, item in enumerate(index_list):
        fpr, tpr, roc_auc = iter(item)
        ci = cal_confidence_interval(roc_auc, len_list[i], 0.95)
        line_name = name_list[i]
        # plt.plot(fpr, tpr, lw=lw, label=line_name + ' AUC = %0.3f' % roc_auc)
        plt.plot(fpr, tpr, c=color_list[i], lw=lw,
                 label=line_name + '\n' + 'AUC={:.3f} (95% CI:{})'.format(roc_auc, ci))
        # plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks([0,0.2,0.4,0.6,0.8,1.0],['0',0.2,0.4,0.6,0.8,1.0], fontsize=18)
        plt.yticks([0, 0.2,0.4,0.6,0.8,1.0],['0',0.2,0.4,0.6,0.8,1.0],  fontsize=18)
        # plt.grid(linestyle='-.')
        plt.xlabel('1-Specificity', fontdict={'size': 20})
        plt.ylabel('Sensitivity', fontdict={'size': 20})
        # 边框
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        # plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
        # plt.gca().yaxis.get_majorticklabels()[0].set_x(-0.05)

        # plt.title('ROC')
        plt.legend(loc="lower right", prop={'family': 'SimHei', 'size': 18})
    # plt.savefig(pic_name, bBox_inches='tight', dpi=600)
    plt.savefig(pic_name, bBox_inches='tight', dpi=600, format="svg")