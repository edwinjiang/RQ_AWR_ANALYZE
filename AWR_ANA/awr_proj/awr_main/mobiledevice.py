#!/usr/bin/python
#encoding:utf8

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def main_ana():

    data_dir = './mobiledevice/data/'
    test_file = os.path.join(data_dir,'test.csv')
    train_file = os.path.join(data_dir,'train.csv')

    test_data = pd.read_csv(test_file)
    train_data = pd.read_csv(train_file)

    print('\n===================== 任务1. 数据查看 =====================')
    print('训练集有{}条记录。'.format(len(train_data)))
    print('测试集有{}条记录。'.format(len(test_data)))

    #可视化数据集
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    #用countplot来看训练和测试两个数据集在数量上的分布情况。来初步确定两个数据集采集的好坏。
    plt.figure(figsize=(10,5))
    ax1 = plt.subplot(1,2,1)
    sns.countplot(x='Activity',data=train_data)
    plt.title('training set')
    plt.xticks(rotation=(45))
    plt.xlabel('class of activity')
    plt.ylabel('Number')

    plt.subplot(1,2,2)
    sns.countplot(x='Activity',data=test_data)
    plt.title('test set')
    plt.xticks(rotation=(45))
    plt.xlabel('class of activity')
    plt.ylabel('Number')
    plt.tight_layout()


    #构造特征和训练测试数据
    feat_name = train_data.columns[:-2].tolist()
    x_train = train_data[feat_name].values
    print ('total of features is {}'.format(x_train.shape[1]))
    x_test = test_data[feat_name].values

    #标签的处理
    train_label = train_data['Activity'].values
    test_label  = test_data['Activity'].values

    label_enc = LabelEncoder()

    y_train = label_enc.fit_transform(train_label)
    y_test  = label_enc.transform(test_label)

    print('类别标签：', label_enc.classes_)
    for i in range(len(label_enc.classes_)):
        print('编码 {} 对应标签 {}。'.format(i, label_enc.inverse_transform(i)))


    # 任务2. 数据建模及验证
    print('\n===================== 任务2. 数据建模及验证 =====================')
    model_name_param_dict = dict(KNN=[5,10,15],LR=[0.01, 1, 100],SVM=[100, 1000, 10000],DT=[50, 100, 150])
    results_ml = pd.DataFrame(columns=['Acruancy','Time'],index=list(model_name_param_dict.keys()))
    for model_name,params in model_name_param_dict.items():
        _, best_acc, mean_duration = train_model(x_train,y_train,x_test,y_test,params,model_name)
        results_ml.loc[model_name,'Acruancy'] = best_acc
        results_ml.loc[model_name,'Time'] = mean_duration


    # 任务3. 模型及结果比较
    print('\n===================== 任务3. 模型及结果比较 =====================')

    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    results_ml.plot(y=['Acruancy'],kind='bar', ylim=[80, 100], ax=ax1,title='Accu(%)', legend=False)

    ax2 = plt.subplot(1,2,2)
    results_ml.plot(y=['Time'],kind='bar',ax=ax2,title='Time',legend=False)
    plt.savefig(os.path.join(data_dir,'pred_results.png'))
    plt.show()

def train_model(x_train,y_train,x_test,y_test,params,model_name='SVM'):

    """
    据给定的参数训练模型，并返回
        1. 最优模型
        2. 平均训练耗时
        3. 准确率
    """

    scores = []
    durations = []
    models = []

    for param in params:
        if model_name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=param)
        elif model_name == 'LR':
            model = LogisticRegression(C=param)
        elif model_name == 'SVM':
            model = SVC(C=param)
        elif model_name == 'DT':
            model = DecisionTreeClassifier(max_depth=param)

        begin_tiem = time.time()
        model.fit(x_train,y_train)
        #model.predict(x_test,y_test)
        end_time = time.time()

        duration = end_time - begin_tiem
        print('{} 耗时{:.4f}s'.format(model,duration))

        durations.append(duration)
        score = model.score(x_test,y_test)
        scores.append(score)
        models.append(model)

    mean_duration = np.mean(durations)
    print('训练模型平均耗时{:.4f}s'.format(mean_duration))
    print()

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_model = models[best_idx]

    return best_model,best_score,mean_duration

# if __name__ =='__main__':
#     main_ana()