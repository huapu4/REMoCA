# -*- encoding: utf-8 -*-
'''
@Time    : 2023/5/2
@Author  : Lin Zhenzhe, Zhang Shuyi
'''
import os.path

import pandas as pd
from plot_utils import *
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from cal_index import all_indicators
import pickle


def save_model(model_path, clf):
    # save the checkpoints
    with open(f'{model_path}.pickle', 'wb') as f:
        pickle.dump(clf, f)


def load_model(model_path):
    # load the checkpoints
    with open(f'{model_path}.pickle', 'rb') as f:
        model = pickle.load(f)
    return model


def rf_classify(train_x, train_y, test_x, test_y):
    print("Training by Random Forest")
    # clf = RandomForestClassifier(n_estimators=50, max_depth=2, max_features=9, n_jobs=-1, )
    clf = RandomForestClassifier(class_weight='balanced')
    clf.fit(train_x, train_y.values.ravel())
    pred_y = clf.predict(test_x)
    # print(classification_report(test_y.values.ravel(), pred_y))
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, pred_y).ravel()
    rfroc = metrics.plot_roc_curve(clf, test_x, test_y)
    print("AUC : {:.4f}, tpr : {:.4f}, tnr : {:.4f}".format(rfroc.roc_auc, (tp / (tp + fn)), (tn / (fp + tn))))


def ridge(train_x, train_y, test_x, test_y):
    print('RidgeClassifier')
    clf = RidgeClassifier(normalize=False, class_weight='balanced', solver='svd')
    clf.fit(train_x, train_y.values.ravel())
    pred_y = clf.predict(test_x)
    score_y = clf.decision_function(test_x)
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, pred_y).ravel()
    ridgeroc = metrics.plot_roc_curve(clf, test_x, test_y)
    plt.close()
    score_y = clf.decision_function(test_x)
    draw_roc_curve(test_y.values.ravel(), score_y, pic_name='MRI')
    print("AUC : {:.4f}, tpr : {:.4f}, tnr : {:.4f}".format(ridgeroc.roc_auc, (tp / (tp + fn)), (tn / (fp + tn))))
    names = train_x.columns.tolist()
    values = clf.coef_.tolist()[0]
    # print(clf.coef_)
    draw_bar(names, values)
    # pred_y = adjust_th(score_y, threshold=-0.1)
    # tn, fp, fn, tp = metrics.confusion_matrix(test_y, pred_y).ravel()
    # ridgeroc = metrics.plot_roc_curve(clf, test_x, test_y)
    # print(metrics.roc_auc_score(test_y, score_y))
    # plt.close()
    # print("AUC : {:.4f}, tpr : {:.4f}, tnr : {:.4f}".format(ridgeroc.roc_auc, (tp / (tp + fn)), (tn / (fp + tn))))


def lr(train_x, train_y, test_x, test_y, json_name):
    print('LogisticRegression')
    clf = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', max_iter=10000)
    clf.fit(train_x, train_y.values.ravel())
    pred_y = clf.predict(test_x)

    tn, fp, fn, tp = metrics.confusion_matrix(test_y, pred_y).ravel()
    print(metrics.confusion_matrix(test_y, pred_y))
    lrroc = metrics.plot_roc_curve(clf, test_x, test_y)
    plt.close()
    names = train_x.columns.tolist()
    values = clf.coef_.tolist()[0]
    # print(clf.coef_)
    # draw_bar(names, values)

    all_indicators(json_name, tn, fp, fn, tp, lrroc.roc_auc)


if __name__ == '__main__':
    # TODO
    moca = ['26', '23'][1]
    print('training on moca_{}:'.format(moca))
    json_base = 'moca_{}_base'.format(moca)
    json_em = 'moca_{}_em'.format(moca)
    json_all = 'moca_{}_all'.format(moca)
    train_file, test_file = f'./trainset.csv', f'./testset.csv'

    train_df, test_df = pd.read_csv(train_file, encoding='gbk'), pd.read_csv(test_file, encoding='gbk')
    train_x, train_y = train_df[train_df.columns[1:6]], train_df[train_df.columns[-1]]
    test_x, test_y = test_df[test_df.columns[1:6]], test_df[test_df.columns[-1]]
    lr(train_x, train_y, test_x, test_y, os.path.join('./results', json_base))

    train_x, train_y = train_df[train_df.columns[6:-1]], train_df[train_df.columns[-1]]
    test_x, test_y = test_df[test_df.columns[6:-1]], test_df[test_df.columns[-1]]
    lr(train_x, train_y, test_x, test_y, os.path.join('./results', json_em))

    train_x, train_y = train_df[train_df.columns[1:-1]], train_df[train_df.columns[-1]]
    test_x, test_y = test_df[test_df.columns[1:-1]], test_df[test_df.columns[-1]]
    lr(train_x, train_y, test_x, test_y, os.path.join('./results', json_all))
