# -*- encoding: utf-8 -*-
'''
@Time    : 2023/5/2
@Author  : Lin Zhenzhe, Zhang Shuyi
'''
import numpy as np
import json


def cal_confidence_interval(value, n, inter_value=0.95):
    # calculate the confidence interval with binomial distribution
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
    return [round(value, 3), ci]


def all_indicators(json_name, tn, fp, fn, tp, auc):
    # loading all index of the model into a json file
    adict = {}
    adict[json_name] = {}
    n = tn + fp + fn + tp
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    acc = (tp + tn) / (tp + fp + fn + tn)
    f1 = (2 * ppv * sen) / (ppv + sen)
    adict[json_name]['sum'] = int(n)
    adict[json_name]['auc'] = cal_confidence_interval(auc, n)
    adict[json_name]['acc'] = cal_confidence_interval(acc, n)
    adict[json_name]['sen'] = cal_confidence_interval(sen, n)
    adict[json_name]['spe'] = cal_confidence_interval(spe, n)
    adict[json_name]['ppv'] = cal_confidence_interval(ppv, n)
    adict[json_name]['npv'] = cal_confidence_interval(npv, n)
    adict[json_name]['f1'] = cal_confidence_interval(f1, n)
    with open("{}.json".format(json_name), "w") as f:
        f.write(json.dumps(adict, ensure_ascii=False, indent=4, separators=(',', ':')))
    # print(cal_confidence_interval(auc, n))
    # print(cal_confidence_interval(acc, n))
    # print(cal_confidence_interval(sen, n))
    # print(cal_confidence_interval(spe, n))
    # print(cal_confidence_interval(ppv, n))
    # print(cal_confidence_interval(npv, n))
    # print(cal_confidence_interval(f1, n))
