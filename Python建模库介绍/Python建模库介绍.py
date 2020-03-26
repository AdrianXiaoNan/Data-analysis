# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

'''
第一节：pandas与建模代码的结合
'''

data = pd.DataFrame({'x0':[1,2,3,4,5],
                     'x1':[0.01,-0.01,0.25,-4.1,0],
                     'y':[-1.5,0,3.6,1.3,-2]})
data.columns
data.values #使用values转换为numpy数组

df2 = pd.DataFrame(data.values,columns = ['one','two','three']) #把numpy数组转化为DF
df3 = data.copy()
df3['strings'] = ['a','b','c','d','e']
df3.values

model_cols = ['x0','x1'] #对DF进行切片，返回需要的列
data.loc[:,model_cols].values #转化成numpy数组narange对象
data['category'] = pd.Categorical(['a','b','a','a','b'],categories = ['a','b'])
dummies = pd.get_dummies(data.category,prefix = 'category')
data_with_dummies = data.drop('category',axis = 1).join(dummies)

'''
第二节：使用Patsy创建模型描述
'''

data = pd.DataFrame({'x0':[1,2,3,4,5],
                     'x1':[0.01,-0.01,0.25,-4.1,0],
                     'y':[-1.5,0,3.6,1.3,-2]})
import patsy
y,X = patsy.dmatrices('y~x0+x1',data)

np.asarray(y)
np.asarray(X) #将patsy模块分离的X转化为np数组

patsy.dmatrices('y~x0+x1+0',data)[1] #通过+0来加入截距
coef,resid, _, _ = np.linalg.lstsq(X,y) #最小二乘回归

coef = pd.Series(coef.squeeze(),index = X.design_info.column_names) #将coef的传入，并连接字段名
#你可以将Python代码混合到你的Patsy公式中，在执行公式时，Patsy库将尝试在封闭作用域中寻找你使用的函数
y,X = patsy.dmatrices('y~x0+np.log(np.abs(x1)+1)',data) 

#一些常用的变量转换包括标准化（对均值0和方差1）和居中（减去平均值）
y,X = patsy.dmatrices('y~standardize(x0) + center(x1)',data)

new_data = pd.DataFrame({'x0':[6,7,8,9],
                         'x1':[3.1,-0.5,0,2.3],
                         'y':[1,2,3,4]})
new_X = patsy.build_design_matrices([X.design_info],new_data)

#对数据集按照列名相加
y,X = patsy.dmatrices('y~I(x0+x1)',data)


#分类数据与Patsy
data = pd.DataFrame({'key1':['a','a','b','b','a','b','a','b'],
                     'key2':[0,1,0,1,0,1,0,0],
                     'v1':[1,2,3,4,5,6,7,8],
                     'v2':[-1,0,2.5,-0.5,4.0,-1.2,0.2,-1.7]})
y,X = patsy.dmatrices('v2~key1',data) #默认将key1列转换为虚拟变量.
y,X = patsy.dmatrices('v2~key1+0',data) #如果你忽略了模型的截距，每个类别值的列将会被包含在模型的设计矩阵中

y,X = patsy.dmatrices('v2~C(key2)',data) #数字类型列可以使用C函数解释为分类类型

#在模型中使用多个分类名词列时,例如，可用于方差分析（ANOVA）模型
data['key2'] = data['key2'].map({0:'zero',1:'one'})
y,X = patsy.dmatrices('y~key1+key2',data)

y,X = patsy.dmatrices('y~key1+key2+key1:key2',data) #按照key1/key2做分类。相同时为1


'''
第三节：statsmodels介绍
'''
import statsmodels.api as sm
import statsmodels.formula.api as smf
def dnorm(mean,variance,size=1):
  if isinstance(size,int):
    size = size,
  return mean + np.sqrt(variance) * np.random.randn(*size)
np.random.seed(12345)
N = 100
X = np.c_[dnorm(0,0.4,size = N),
          dnorm(0,0.6,size = N),
          dnorm(0,0.2,size = N)]
eps = dnorm(0,0.1,size = N)
beta = [0.1,0.3,0.5]
y = np.dot(X,beta)+eps

#sm.add_constant函数可以将截距列添加到现有矩阵
X_model = sm.add_constant(X)

#sm.OLS类可以拟合一个最小二乘线性回归
model = sm.OLS(y,X)
#模型的fit方法返回一个回归结果对象，该对象包含了估计的模型参数和其他的诊断
results = model.fit()
results.params
print(results.summary()) #打印模型诊断细节

data = pd.DataFrame(X,columns = ['col0','col1','col2'])
data['y'] = y
data[-5:]

results = smf.ols('y~col0+col1+col2',data=data).fit() #使用patsy公式建模
results.predict(data[:5]) #给定新的样本外数据后，你可以根据估计的模型参数计算预测值

#评估时间序列处理
init_x = 4
values = [init_x,init_x]
N = 1000
b0 = 0.8
b1 = -0.4
noise = dnorm(0,0.1,N)
for i in range(N):
  new_x = values[-1]*b0+values[-2]*b1+noise[i]
  values.append(new_x) #该数据具有参数为0.8和-0.4的AR（2）结构（两个滞后）
#model = sm.tsa.AR(values)
#results = model.fit(maxlag=None)
  
'''
第四节：scikit-learn介绍
'''
train = pd.read_csv('F:/利用Python进行数据分析数据集/datasets/titanic/train.csv')
test = pd.read_csv('F:/利用Python进行数据分析数据集/datasets/titanic/test.csv')

train.isnull().sum() #检查列是否有缺失值并计数
test.isnull().sum()

#缺失值填补（imputation）
impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value) 

#添加了一列IsFemale作为’Sex’列的编码版本
train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)

#决定一些模型变量并创建NumPy数组
predictors = ['Pclass','IsFemale','Age']
X_train = train[predictors].values #生成numpy数组
X_test = test[predictors].values
y_train = train['Survived'].values
X_train[:5]
y_train[:5]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
y_predict[:10]
#(y_true == y_predict).mean()拥有测试数据集的真实值，则可以计算精度百分比或其他一些错误指标

#LogisticRegressionCV类可以与一个参数一起使用，该参数表示网格搜索在模型正则化参数C上的细致度
from sklearn.linear_model import LogisticRegressionCV
model_cv = LogisticRegressionCV(10)
model_cv.fit(X_train,y_train)

#手动进行交叉验证，可以使用cross_val_score帮助函数，该函数处理数据拆分过程
from sklearn.model_selection import cross_val_score
model = LogisticRegression(C = 10)
scores = cross_val_score(model,X_train,y_train,cv=4)
scores
#默认评分指标是依赖于模型的，但可以选择明确的评分函数.经过交叉验证的模型需要更长时间的训练，但通常可以产生更好的模型性能。
