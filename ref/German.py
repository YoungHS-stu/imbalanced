#!/usr/bin/env python
# coding: utf-8

# In[256]:


#基础库
# Laoding Basic data analysis libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Basic programming and data visualization libraries
import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')
import smote_variants as sv


# In[183]:


train_df = pd.read_csv("german_credit_data_dataset_.csv")


# In[184]:


#构建随机森林进行分类
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(train_df.iloc[:,0:-1], train_df.iloc[:,-1], test_size = 0.3, random_state = 66) # random_state = 0 每次切分的数据都一样
#2是少数类


# In[185]:


y_train = y_train.astype(int)
y_test = y_test.astype(int)
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)
X_train=X_train.astype(int)
X_test=X_test.astype(int)


# In[186]:


#重采样

#随机下采样 rus
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
cc = RandomUnderSampler(sampling_strategy='majority',random_state=0)
X_rus, y_rus = cc.fit_sample(X_train, y_train)


# In[187]:


#随机过采样
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy='auto',random_state=0)
X_ros, y_ros = ros.fit_sample(X_train, y_train)


# In[188]:


#判断并输出多数类以及少数类
def deter(x,y):
    if len(y[y['label'] == 0])<len(y[y['label'] == 1]):
        y_less = y[y['label'] == 0 ]
        y_more = y[y['label'] == 1]
        x_less = x[y['label'] == 0]
        x_more = x[y['label'] == 1]
    else:
        y_less = y[y['label'] == 1]
        y_more = y[y['label'] == 0]
        x_less = x[y['label'] == 1]
        x_more = x[y['label'] == 0]
    return x_less,x_more,y_less,y_more

def verify_log(x,y):#查看简单的结果
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf = clf.fit(x,y)
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    aa=0
    for i in a:
        aa=aa+i
    print(aa)
    print(a)
def verify_lgbm(x,y):
    import json
    import lightgbm as lgb
    import pandas as pd
    lgb_train = lgb.Dataset(x, y) # 将数据保存到LightGBM二进制文件将使加载更快
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据

    # 将参数写成字典下形式
    params = {
        'learning_rate': 0.05,
        'lambda_l1':0.4,
        'lambda_l2':0.5,
        'num_iterations':10000,
        'max_depth':16,
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary', # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 35,   # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.6, # 建树的特征选择比例
        'bagging_fraction': 0.6, # 建树的样本采样比例
        'bagging_freq': 8,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    #learning_rate: 0.05,max_depth: 16,feature_fraction:0.6,bagging_fraction:0.6,bagging_freq:8,lambda_l1=0.4,cat_smooth:0,lambda_l2:0.5
    #num_iterations:100,num_leaves:35
    c=[]
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(x, y)
    y_pred = lgb_model.predict(X_test)
    a=metric(y_pred,y_test)
    aa=0
    for i in a:
        aa=aa+i
    return aa


# In[189]:


#RUS
#策略：随机删去多数样本直至平衡
import random
def rus(x,y,a=1.5):
    x_less,x_more,y_less,y_more=deter(x,y)
    del_num = int(len(y_more) - len(y_less)*a)
    for i in range(del_num):
        no = random.randrange(0,len(y_more))#选择删除第几号的少数类   
        x_more = x_more.drop(y_more.index[no])
        y_more = y_more.drop(y_more.index[no])
    x_new=x_less.append(x_more,ignore_index=True)
    y_new=y_less.append(y_more,ignore_index=True)
    
    return x_new,y_new
X_rus_1,y_rus_1=rus(X_train,y_train,1.5)


# In[190]:


#ROS伪代码
#策略：随机复制少数样本直到数据平衡
def ros(x,y,a=1):
    x_less,x_more,y_less,y_more=deter(x,y)

    generate_num = int(len(y_more) - len(y_less)*a)
    for i in range(generate_num):
        no = random.randrange(0,len(y_less))#选择生成第几号的少数类
        #x_new = x_less.iloc[no]#把选择的少数类拿出来
        #y_new = y_less.iloc[no]#把选择的少数类拿出来
        x_less = x_less.append(x_less.iloc[no],ignore_index=True)
        y_less = y_less.append(y_less.iloc[no],ignore_index=True)
    x_new=x_less.append(x_more,ignore_index=True)
    y_new=y_less.append(y_more,ignore_index=True)
    
    return x_new,y_new

'''    比较类别的多少，选定少数类
    复制少数类
    生成样本数=多数类-少数类
    循环（生成样本数）:
        随机生成数（范围少数类以内）
        采样加进少数类里'''
'''RUS:正确数量：117537 
总共量：84188
生成量=多数类/2-少数类
原始少数量：5830
原始多数量：78358'''
X_ros_1,y_ros_1 = ros(X_train,y_train,1)


# In[191]:


def clu_rus(x,y):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_clu, y_clu = cc.fit_sample(x, y)
    x = x.append(X_clu)
    y = y.append(y_clu)
    x,y = rus(x,y)

    return x,y
X_clu_rus,y_clu_rus=clu_rus(X_train,y_train)


# In[192]:


def clu_ros_tom(x,y):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_clu, y_clu = cc.fit_sample(x, y)
    x = x.append(X_clu)
    y = y.append(y_clu)
    x,y = ros(x,y)
    from imblearn.under_sampling import TomekLinks
    nm1 = TomekLinks(sampling_strategy='all')
    x, y = nm1.fit_sample(x, y)

    return x,y
X_clu_ros_tom,y_clu_ros_tom=clu_ros_tom(X_train,y_train)


# In[193]:


def clu_tom_ros(x,y):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_clu, y_clu = cc.fit_sample(x, y)
    x = x.append(X_clu)
    y = y.append(y_clu)
    from imblearn.under_sampling import TomekLinks
    nm1 = TomekLinks(sampling_strategy='all')
    x, y = nm1.fit_sample(x, y)
    x,y = ros(x,y)

    return x,y
X_clu_tom_ros,y_clu_tom_ros=clu_tom_ros(X_train,y_train)


# In[194]:


def clu_tom_rus(x,y):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_clu, y_clu = cc.fit_sample(x, y)
    x = x.append(X_clu)
    y = y.append(y_clu)
    from imblearn.under_sampling import TomekLinks
    nm1 = TomekLinks(sampling_strategy='all')
    x, y = nm1.fit_sample(x, y)
    x,y = rus(x,y)

    return x,y
X_clu_tom_rus,y_clu_tom_rus=clu_tom_rus(X_train,y_train)


# In[195]:


def b_smo_ros(x,y):
    from imblearn.over_sampling import BorderlineSMOTE
    sm = BorderlineSMOTE(random_state=42,kind="borderline-1")
    x, y = ros(x,y)
    x, y = sm.fit_resample(x, y)

    return x, y
X_b_smo_ros,y_b_smo_ros = b_smo_ros(X_train,y_train)


# In[196]:


def b_smo_rus(x,y):
    from imblearn.over_sampling import BorderlineSMOTE
    sm = BorderlineSMOTE(random_state=42,kind="borderline-1")
    x, y = rus(x,y,2)
    x, y = sm.fit_resample(x, y)
    return x, y
X_b_smo_rus,y_b_smo_rus = b_smo_rus(X_train,y_train)


# In[197]:


def rus_ros(x,y,a=2,b=1.5):
    x,y = rus(x,y,a)#将多数类样本删去直到少数类的两倍
    x,y = ros(x,y,b)#复制少数类样本直到多数类样本是少数类的1.5倍
    return x,y
X_rus_ros,y_rus_ros=rus_ros(X_train,y_train,2.6,1.5)


# In[198]:


def ros_rus(x,y,a=2,b=1.5):
    x,y = ros(x,y,a)
    x,y = rus(x,y,b)
    return x,y
X_ros_rus,y_ros_rus=ros_rus(X_train,y_train,2.5,1.5)


# In[199]:


def b_smo_rus_ros(x,y):
    from imblearn.over_sampling import BorderlineSMOTE
    sm = BorderlineSMOTE(random_state=42,kind="borderline-1")
    x, y = rus_ros(x,y,2.5,1.5)
    x, y = sm.fit_resample(x, y)
    return x, y
X_b_smo_rus_ros,y_b_smo_rus_ros = b_smo_rus_ros(X_train,y_train)


# In[200]:


def b_smo_ros_rus(x,y):
    from imblearn.over_sampling import BorderlineSMOTE
    sm = BorderlineSMOTE(random_state=42,kind="borderline-1")
    x, y = ros_rus(x,y,2.5,1.5)
    x, y = sm.fit_resample(x, y)
    return x, y
X_b_smo_ros_rus,y_b_smo_ros_rus = b_smo_ros_rus(X_train,y_train)


# In[201]:


def clu_tom_rus_ros(x,y):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_clu, y_clu = cc.fit_sample(x, y)
    x = x.append(X_clu)
    y = y.append(y_clu)
    from imblearn.under_sampling import TomekLinks
    nm1 = TomekLinks(sampling_strategy='all')
    x, y = nm1.fit_sample(x, y)
    x,y = rus_ros(x,y)

    return x,y
X_clu_tom_rus_ros,y_clu_tom_rus_ros=clu_tom_rus_ros(X_train,y_train)


# In[202]:


def S_enn_rus(x,y):
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN(random_state = 0)
    x, y = rus(x,y,1.5)
    x, y = smote_enn.fit_sample(x, y)
    return x, y
X_S_enn_rus,y_S_enn_rus=S_enn_rus(X_train,y_train)


# In[203]:


def S_enn_ros(x,y):
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN(random_state = 0)
    x, y = ros(x,y,1.5)
    x, y = smote_enn.fit_sample(x, y)
    return x, y
X_S_enn_ros,y_S_enn_ros=S_enn_ros(X_train,y_train)


# In[204]:


def NRAS_ros(x,y):

    x, y = ros(x,y,1.5)
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(-1)
    oversampler = sv.NRAS()
    x, y= oversampler.sample(x, y)
    return x, y
X_NRAS_ros, y_NRAS_ros = NRAS_ros(X_train,y_train)


# In[205]:


def NRAS_rus(x,y):

    x, y = rus(x,y,1.5)
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(-1)
    oversampler = sv.NRAS()
    x, y= oversampler.sample(x, y)
    return x, y
X_NRAS_rus, y_NRAS_rus = NRAS_rus(X_train,y_train)


# In[206]:


def MWMOTE_ros(x,y):
    x=x.astype(float)
    x, y = ros(x,y,1.5)
    x = np.array(x) 
    y = np.array(y)
    y = y.reshape(-1)
    oversampler = sv.MWMOTE()
    x, y= oversampler.sample(x, y)
    return x, y
X_MWMOTE_ros, y_MWMOTE_ros=MWMOTE_ros(X_train,y_train)


# In[207]:


def MWMOTE_rus(x,y):
    x=x.astype(float)
    x, y = rus(x,y,1.5)
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(-1)
    oversampler = sv.MWMOTE()
    x, y= oversampler.sample(x, y)
    return x, y
X_MWMOTE_rus, y_MWMOTE_rus=MWMOTE_rus(X_train,y_train)


# In[355]:


import math
def MWMOTE_trans(x,y,max_ = 1000000,min_ = 1000):
    if len(x) > max_:
        a=1
    elif len(x) < min_:
        a=0
    else:
        a=(math.log10(len(x))-math.log10(min_))/(math.log10(max_)-math.log10(min_))
    x, y = a * MWMOTE_ros(x,y) + (1-a) * MWMOTE_rus(x,y)
    return x,y
X_MWMOTE_trans,y_MWMOTE_trans = MWMOTE_trans(X_train,y_train)


# In[357]:


len(X_MWMOTE_trans)


# In[350]:


x,y=1*MWMOTE_ros(X_train,y_train)+0*MWMOTE_rus(X_train,y_train)


# In[349]:


x


# In[208]:


#ROS+SMOTE代码
#策略：随机抽取k个样本，然后随机选择两个来，一个的权重是1，另一个的权重是（0,1）
def ros_smo_1(x,y,k):
    x_less,x_more,y_less,y_more=deter(x,y)
    generate_num = len(y_more)//2 - len(y_less)
    a=[]
    for i in range(generate_num):
        for n in range(k):
            no = random.randrange(0,len(y_less))#选择生成第几号的少数类
            a.append(no)
        chosen_num_1 = random.randrange(0,k)
        chosen_num_2 = random.randrange(0,k)
        
        x_gen = x_less.iloc[chosen_num_1] + random.random()*x_less.iloc[chosen_num_2]
        y_gen = y_less.iloc[chosen_num_1]
        #x_new = x_less.iloc[no]#把选择的少数类拿出来
        #y_new = y_less.iloc[no]#把选择的少数类拿出来
        x_less = x_less.append(x_gen,ignore_index=True)
        y_less = y_less.append(y_gen,ignore_index=True)
    x_new = x_less.append(x_more,ignore_index=True)
    y_new = y_less.append(y_more,ignore_index=True)
    
    return x_new,y_new
X_ros_smo_1,y_ros_smo_1 = ros_smo_1(X_train,y_train,5)


# In[209]:


#策略：随机抽取k个样本，以（0,1）的权重合成新的样本
def ros_smo_2(x,y,k):
    x_less,x_more,y_less,y_more=deter(x,y)
    generate_num = len(y_more)//2 - len(y_less)
    for i in range(generate_num):
        a=[]
        for n in range(k):
            no = random.randrange(0,len(y_less))#选择生成第几号的少数类
            a.append(no)
        x_gen = random.random()*x_less.iloc[a[len(a)-1]]
                                            
        for m in range(len(a)-1):           
            x_gen = x_gen + random.random()*x_less.iloc[a[m]]
        y_gen = y_less.iloc[1]
        #x_new = x_less.iloc[no]#把选择的少数类拿出来
        #y_new = y_less.iloc[no]#把选择的少数类拿出来
        x_less = x_less.append(x_gen,ignore_index=True)
        y_less = y_less.append(y_gen,ignore_index=True)
    x_new = x_less.append(x_more,ignore_index=True)
    y_new = y_less.append(y_more,ignore_index=True)
    
    return x_new,y_new
X_ros_smo_2,y_ros_smo_2 = ros_smo_2(X_train,y_train,5)


# In[210]:


#策略：先CLU生成新的样本加进数据集，再用ROS生成新的样本
def clu_ros(x,y):
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_clu, y_clu = cc.fit_sample(x, y)
    a1 = x.append(X_clu)
    a2 = y.append(y_clu)
    a1,a2 = ros(a1,a2)
    return a1,a2
X_clu_ros,y_clu_ros=clu_ros(X_train,y_train)


# In[211]:


#SMOTE 参数k = 5, p = 3
from imblearn.over_sampling import SMOTE
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_sample(X_train, y_train)


# In[212]:


#Borderline SMOTE 参数试试改成k = 5
from imblearn.over_sampling import BorderlineSMOTE
sm = BorderlineSMOTE(random_state=42,kind="borderline-1")
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[213]:


#TomekLinks,数据清洗方法，无法控制欠采样数量
from imblearn.under_sampling import TomekLinks
nm1 = TomekLinks(sampling_strategy='all')
X_tom, y_tom = nm1.fit_sample(X_train, y_train)


# In[214]:


#CondensedNearestNeighbour,数据清洗方法，无法控制欠采样数量
from time import time 
#这个方法耗时久一点,大概要11.5h

from imblearn.under_sampling import CondensedNearestNeighbour
renn = CondensedNearestNeighbour(random_state=0)

X_con, y_con = renn.fit_resample(X_train, y_train)


# In[215]:


#OSS-one side selection（单边选择）,数据清洗方法，无法控制欠采样数量

from imblearn.under_sampling import OneSidedSelection
oss = OneSidedSelection(random_state=0)
X_oss, y_oss = oss.fit_sample(X_train, y_train)


# In[216]:


#ClusterCentroids，使用KMeans对各类别样本进行聚类，然后将聚类中心作为新生成的样本，以此达到下采样的效果。

#这个方法耗时13mins左右
from imblearn.under_sampling import ClusterCentroids
from time import time 
cc = ClusterCentroids(random_state=0)

X_clu, y_clu = cc.fit_sample(X_train, y_train)


# In[217]:


#SMOTE+ENN（可选最近邻）参数k = 5
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_smote_enn, y_smote_enn = smote_enn.fit_sample(X_train, y_train)


# In[218]:


#ADASYN（自适应综合过采样 
from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state=42)
X_ada, y_ada = ada.fit_resample(X_train, y_train)


# In[219]:


#SMOTE+TomekLinks 参数k = 5
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_train, y_train)


# In[220]:


X_n=np.array(X_train)
y_n=np.array(y_train)
y_n=y_n.reshape(-1)


# In[221]:


from time import time
import smote_variants as sv
#以前的数据输入X_trian,y_train直接使用会报错，需要转成格式相符的数组array
oversampler= sv.Safe_Level_SMOTE()
start=time()
# supposing that X and y contain some the feature and target data of some dataset
X_n=X_n.astype(float)
X_sl_smote, y_sl_smote= oversampler.sample(X_n, y_n)
print("程序运行时间为：" + str(time()-start) + "秒")
X_sl_smote.shape


# In[222]:


oversampler= sv.MWMOTE(k2=3)

# supposing that X and y contain some the feature and target data of some dataset
X_mwsmote, y_mwsmote= oversampler.sample(X_n, y_n)


# In[223]:


oversampler= sv.NRAS()
start=time()
# supposing that X and y contain some the feature and target data of some dataset
X_NRAS, y_NRAS= oversampler.sample(X_train, y_n)
print("程序运行时间为：" + str(time()-start) + "秒")


# In[224]:


oversampler= sv.kmeans_SMOTE()
start=time()
# supposing that X and y contain some the feature and target data of some dataset
X_k_smote, y_k_smote= oversampler.sample(X_n, y_n)
print("程序运行时间为：" + str(time()-start) + "秒")


# In[225]:


import sklearn.metrics as metrics
from math import sqrt

def metric (precited,expected):
    precited=np.array(precited)
    expected = np.array(expected).flatten()
    res =(precited ^ expected)#亦或使得判断正确的为0,判断错误的为1

    r = np.bincount(res)
    tp_list = ((precited)&(expected))
    fp_list = (precited&(~expected))
    tp_list=tp_list.tolist()
    fp_list=fp_list.tolist()
    tp=tp_list.count(1)
    fp=fp_list.count(1)
    tn = r[0]-tp
    fn = r[1]-fp
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    Dom = TPR - TNR
    G_means=np.sqrt(TPR*TNR)
    IBA = (1+0.05*Dom )*G_means
    precision=tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score=metrics.f1_score(expected, precited, average='weighted')
    acc=(tp+tn)/(tp+tn+fp+fn)
    fpr, tpr, thresholds = metrics.roc_curve(expected, precited)
    auc=metrics.auc(fpr, tpr)
    metric_list=[]
    metric_list.append(acc)
    metric_list.append(precision)
    metric_list.append(recall)
    metric_list.append(f1_score)
    metric_list.append(auc)
    metric_list.append(IBA)
    return metric_list
def plot_fn(precited,expected):
    precited=np.array(precited)
    expected = np.array(expected).flatten()
    res =(precited ^expected)#亦或使得判断正确的为0,判断错误的为1
    r = np.bincount(res)
    tp_list = ((precited)&(expected))
    fp_list = (precited&(~expected))
    tp_list=tp_list.tolist()
    fp_list=fp_list.tolist()
    tp=tp_list.count(1)
    fp=fp_list.count(1)
    tn = r[0]-tp
    fn = r[1]-fp
    list_=[]
    list_.append(fp)
    list_.append(fn)
    list_.append(tp)
    list_.append(tn)
    return list_


# In[226]:


dataset=[]
dataset.append(X_train)
dataset.append(X_rus)
dataset.append(X_ros)
dataset.append(X_smo)
dataset.append(X_res)
dataset.append(X_tom)
dataset.append(X_con)
dataset.append(X_oss)
dataset.append(X_clu)
dataset.append(X_smote_enn)
dataset.append(X_ada)
dataset.append(X_smote_tomek)
dataset.append(X_sl_smote)
dataset.append(X_mwsmote)
dataset.append(X_NRAS)
dataset.append(X_k_smote)
dataset.append(X_ros_1)
dataset.append(X_ros_smo_1)
dataset.append(X_ros_smo_2)
dataset.append(X_clu_ros)

dataset.append(X_rus_1)
dataset.append(X_clu_rus)
dataset.append(X_clu_ros_tom)
dataset.append(X_clu_tom_ros)
dataset.append(X_clu_tom_rus)
dataset.append(X_b_smo_ros)
dataset.append(X_b_smo_rus)

dataset.append(X_S_enn_rus)
dataset.append(X_S_enn_ros)
dataset.append(X_NRAS_ros)
dataset.append(X_NRAS_rus)
dataset.append(X_MWMOTE_ros)
dataset.append(X_MWMOTE_rus)
dataset.append(X_rus_ros)
dataset.append(X_ros_rus)
dataset.append(X_b_smo_rus_ros)
dataset.append(X_b_smo_ros_rus)
dataset.append(X_clu_tom_rus_ros)


dataset.append(y_train)
dataset.append(y_rus)
dataset.append(y_ros)
dataset.append(y_smo)
dataset.append(y_res)
dataset.append(y_tom)
dataset.append(y_con)
dataset.append(y_oss)
dataset.append(y_clu)
dataset.append(y_smote_enn)
dataset.append(y_ada)
dataset.append(y_smote_tomek)
dataset.append(y_sl_smote)
dataset.append(y_mwsmote)
dataset.append(y_NRAS)
dataset.append(y_k_smote)
dataset.append(y_ros_1)
dataset.append(y_ros_smo_1)
dataset.append(y_ros_smo_2)
dataset.append(y_clu_ros)

dataset.append(y_rus_1)
dataset.append(y_clu_rus)
dataset.append(y_clu_ros_tom)
dataset.append(y_clu_tom_ros)
dataset.append(y_clu_tom_rus)
dataset.append(y_b_smo_ros)
dataset.append(y_b_smo_rus)

dataset.append(y_S_enn_rus)
dataset.append(y_S_enn_ros)
dataset.append(y_NRAS_ros)
dataset.append(y_NRAS_rus)
dataset.append(y_MWMOTE_ros)
dataset.append(y_MWMOTE_rus)
dataset.append(y_rus_ros)
dataset.append(y_ros_rus)
dataset.append(y_b_smo_rus_ros)
dataset.append(y_b_smo_ros_rus)
dataset.append(y_clu_tom_rus_ros)


# In[227]:


#创建绘制tp,fn等参数的列表
list_total=[]
for i in range(int(len(dataset)/2)):
    list_total.append([])
X_value = ["ini", "RUS", "ROS", "SMO", "B-SMO", "Tom", "Con", "OSS", "Clu", "S-Enn", "ADA", "S-Tom",'SL-SMO','MWSMOTE',
           'NRAS','K-smo','ros_1','ros_smo_1','ros_smo_2','clu_ros','rus_1','clu_rus','clu_ros_tom','clu_tom_ros','clu_tom_rus','b_smo_ros','b_smo_rus'
          ,'S_enn_rus','S_enn_ros','NRAS_ros','NRAS_rus','MWMOTE_ros','MWMOTE_rus','rus_ros','ros_rus','b_smo_rus_ros','b_smo_ros_rus','clu_tom_rus_ros']
title_list=['LogisticRegression','LogisticRegression_1','KNeighbors','GaussianNB','DecisionTree'
            ,'Bagging_LR','Bagging_tree','RandomForest','RandomForest_1','ExtraTrees','AdaBoost'
            ,'AdaBoost_1','GradientBoosting','vote_1','vote_2','LGBM','xgboost']


# In[228]:


from sklearn.linear_model import LogisticRegression
c=[]
for i in range(int(len(dataset)/2)):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='LogisticRegression')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
LogisticRegressiont_metric=c


# In[229]:


from sklearn.linear_model import LogisticRegression
c=[]
for i in range(int(len(dataset)/2)):
    #tol=0.00000001
    clf = LogisticRegression(tol=0.00000001)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图
df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='LogisticRegression_1')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
LogisticRegressiont_1_metric=c


# In[230]:


from sklearn.neighbors import KNeighborsClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图
df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='KNeighbors')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
KNeighbors_metric=c


# In[231]:


from sklearn.naive_bayes import GaussianNB
c=[]
for i in range(int(len(dataset)/2)):
    clf = GaussianNB()
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='GaussianNB')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
GaussianNB_metric=c


# In[232]:


#Classifier = J48, the confidence factor used for pruning = 0.25, the minimum number of instances per leaf = 2
#class_weight=0.25,min_weight_fraction_leaf=2
from sklearn import tree
c=[]
for i in range(int(len(dataset)/2)):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='DecisionTree')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
DecisionTree_metric=c


# In[233]:


from sklearn.ensemble import BaggingClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'), max_samples=0.5, max_features=0.5)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='Bagging_LR')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
Bagging_LR_metric=c


# In[234]:


#论文的
from sklearn.ensemble import BaggingClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = BaggingClassifier(tree.DecisionTreeClassifier(),n_estimators=40)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='Bagging_tree')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
Bagging_tree_metric=c


# In[235]:


from sklearn.ensemble import RandomForestClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=12, random_state=0)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='RandomForest')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
RandomForest_metric=c


# In[236]:


#论文的
from sklearn.ensemble import RandomForestClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = RandomForestClassifier(n_estimators=40)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='RandomForest_1')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
RandomForest_1_metric=c


# In[237]:


from sklearn.ensemble import ExtraTreesClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='ExtraTrees')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
ExtraTrees_metric=c


# In[238]:


from sklearn.ensemble import AdaBoostClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = AdaBoostClassifier(n_estimators=10)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='AdaBoost')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
AdaBoost_metric=c


# In[239]:


#论文的参数
from sklearn.ensemble import AdaBoostClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = AdaBoostClassifier(n_estimators=40)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='AdaBoost_1')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
AdaBoost_1_metric=c


# In[240]:


from sklearn.ensemble import GradientBoostingClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='GradientBoosting')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
GradientBoosting_metric=c


# In[242]:


#随便调下参
from sklearn.ensemble import GradientBoostingClassifier
c=[]
for i in range(int(len(dataset)/2)):
    clf = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1, max_depth=3, random_state=0)
    clf = clf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=clf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='GradientBoosting——')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c


# In[243]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
c=[]
clf1 = LogisticRegression(tol=0.00000001)
clf2 = RandomForestClassifier(n_estimators=40, random_state=0)
clf3 = GaussianNB()
for i in range(int(len(dataset)/2)):
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    eclf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=eclf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='vote_1')

#计算指标和
for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
vote_1_metric=c


# In[244]:


c=[]
clf1 = LogisticRegression(tol=0.00000001)
clf2 = RandomForestClassifier(n_estimators=40, random_state=0)
clf3 = GaussianNB()
for i in range(int(len(dataset)/2)):
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    eclf.fit(dataset[i], dataset[i+int(len(dataset)/2)])
    y_pred=eclf.predict(X_test)
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='vote_2')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
vote_2_metric=c


# In[245]:


import json
import lightgbm as lgb
import pandas as pd
 # 将数据保存到LightGBM二进制文件将使加载更快
lgb_train = lgb.Dataset(X_train, y_train) # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据

# 将参数写成字典下形式
params = {
    'learning_rate': 0.05,
    'lambda_l1':0.4,
    'lambda_l2':0.5,
    'num_iterations':10000,
    'max_depth':16,
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary', # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 35,   # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.6, # 建树的特征选择比例
    'bagging_fraction': 0.6, # 建树的样本采样比例
    'bagging_freq': 8,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
#learning_rate: 0.05,max_depth: 16,feature_fraction:0.6,bagging_fraction:0.6,bagging_freq:8,lambda_l1=0.4,cat_smooth:0,lambda_l2:0.5
#num_iterations:100,num_leaves:35
c=[]
for i in range(int(len(dataset)/2)):
    lgb_model = lgb.LGBMClassifier()
    lgb_train = lgb.Dataset(dataset[i].astype(float), dataset[i+int(len(dataset)/2)])
    lgb_model.fit(dataset[i].astype(float), dataset[i+int(len(dataset)/2)])
    y_pred = lgb_model.predict(X_test.astype(float))
    a=metric(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='LGBM')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
LGBM_metric=c


# In[246]:


import xgboost as xgb
#xgboost专属的计算评估指标函数
def metric_(pred,true):
    cm = metrics.confusion_matrix(true, pred, labels=range(2))
    #cm = cm.astype(np.float32)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)    
    TR = tp / (tp + fn)  
    TPR=TR[0]
    TNR=TR[1]
    #TNR = tn / (tn + fp)
    Dom = TPR - TNR
    G_means=np.sqrt(TPR*TNR)
    IBA = (1+0.05*Dom )*G_means
    metric_list=[]
    acc=metrics.accuracy_score(true, pred)
    precision=metrics.precision_score(true, pred)
    recall= metrics.recall_score(true, pred)
    f1_score=metrics.f1_score(true, pred, average='weighted') 
    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    auc=metrics.auc(fpr, tpr)
    metric_list.append(acc)
    metric_list.append(precision)
    metric_list.append(recall)
    metric_list.append(f1_score)
    metric_list.append(auc)
    metric_list.append(IBA)
    return metric_list
def plot_fn_(pred,true):
    cm = metrics.confusion_matrix(true, pred, labels=range(2))
    #cm = cm.astype(np.float32)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)   
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    fp_=fp
    fp = fp_[1]
    fn = fp_[0]
    tp_=tp
    tp = tp_[1]
    tn = tp_[0]
    list_=[]
    list_.append(fp)
    list_.append(fn)
    list_.append(tp)
    list_.append(tn)
    return list_
c=[]
for i in range(int(len(dataset)/2)):
    data_x=pd.DataFrame(dataset[i])
    data_x.columns=dataset[11].columns
    data_train = xgb.DMatrix(data_x,label=dataset[i+int(len(dataset)/2)])
    data_test = xgb.DMatrix(X_test,label=y_test)
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax','num_class': 3} # logitraw
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 7
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watchlist)  
    y_pred = bst.predict(data_test)
    a=metric_(y_pred,y_test)
    c.append(a)
    list_total[i]=plot_fn_(y_pred,y_test)
#开始画图

df = pd.DataFrame(list_total)
df.index=X_value
df.columns=['fp','fn','tp','tn']
df.plot(kind='bar',figsize=(20,10),title='xgboost')

for i in c:#对于c里的每个重采样方法
    d=0
    for m in i:#对于每个重采样方法的指标
        d=d+m  #加起来求和
    i.append(d)
c
xgboost_metric=c


# In[247]:


metric_total=[]
metric_total.append(LogisticRegressiont_metric)
metric_total.append(LogisticRegressiont_1_metric)
metric_total.append(KNeighbors_metric)
metric_total.append(GaussianNB_metric)
metric_total.append(DecisionTree_metric)
metric_total.append(Bagging_LR_metric)
metric_total.append(Bagging_tree_metric)
metric_total.append(RandomForest_metric)
metric_total.append(RandomForest_1_metric)
metric_total.append(ExtraTrees_metric)
metric_total.append(AdaBoost_metric)
metric_total.append(AdaBoost_1_metric)
metric_total.append(GradientBoosting_metric)
metric_total.append(vote_1_metric)
metric_total.append(vote_2_metric)
metric_total.append(LGBM_metric)
metric_total.append(xgboost_metric)


# In[248]:


#自动保留小数点后四位函数
def round_(data):
    for i in range(len(data)):
        for ii in range(len(data[i])):
            data[i][ii]=round(data[i][ii],4)
    return data


# In[249]:


index = ["*_*ini", "*_*RUS", "*_*ROS", "*_*SMO", "*_*B-SMO", "*_*Tom", "*_*Con", "*_*OSS", "*_*Clu", "*_*S-Enn", "*_*ADA", "*_*S-Tom",'*_*SL-SMO','*_*MWMOTE',
           '*_*NRAS','*_*K-smo','*_*ros_1','*_*ros_smo_1','*_*ros_smo_2','*_*clu_ros','*_*rus_1','*_*clu_rus','*_*clu_ros_tom','*_*clu_tom_ros','*_*clu_tom_rus','*_*b_smo_ros','*_*b_smo_rus'
          ,'*_*S_enn_rus','*_*S_enn_ros','*_*NRAS_ros','*_*NRAS_rus','*_*MWMOTE_ros','*_*MWMOTE_rus','*_*rus_ros','*_*ros_rus','*_*b_smo_rus_ros','*_*b_smo_ros_rus','*_*clu_tom_rus_ros']
col=['ACC','Precision','Recall','F1 score','AUC','IBA','total']
def textbf_each_col(table):
    table = table.T
    markEach = table.apply(lambda x: x.max() if x.name not in ('consumeTime', 'trainLoss', 'testLoss') else x.min(), axis=1)
    newTable = table.copy()
    for each in table.index:
        newTable.loc[each] = table.loc[each].apply(lambda x: '\\textbf{{{:f}}}'.format(x) if x==markEach[each] else '{:f}'.format(x))
    newTable = newTable.T
    return newTable


# In[250]:


for i in range(len(metric_total)):
    metric_total[i]=round_(metric_total[i])
    metric_total[i]=pd.DataFrame(metric_total[i])
    print(i)
    metric_total[i].columns=col
    metric_total[i].index=index
    metric_total[i]=textbf_each_col(metric_total[i])


# In[251]:


metric_total[0]


# In[252]:


metric_total


# In[254]:


a11=pd.concat(metric_total)


# In[255]:


a11.to_csv('german_2.21.csv')

