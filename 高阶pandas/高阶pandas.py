'''
第一节：分类数据
'''

import numpy as np
import pandas as pd

values = pd.Series(['apple','orange','apple','apple']*2)
pd.unique(values)
pd.value_counts(values)

values = pd.Series([0,1,0,0]*2)
dim = pd.Series(['apple','orange'])
dim.take(values)

#pandas中的Categorical类型
fruits = ['apple','orange','apple','apple']*2
N = len(fruits)
df = pd.DataFrame({'fruit':fruits,
                   'basket_id':np.arange(N),
                   'count':np.random.randint(3,15,size = N),
                   'weight':np.random.uniform(0,4,size = N)},columns = ['basket_id','fruit','count','weight'])

fruit_cat = df['fruit'].astype('category')
c = fruit_cat.values
c.categories #categories属性
c.codes #codes属性
df['fruit'] = df['fruit'].astype('category') #将数据集的fruit列转换为categories对象
df.fruit

my_categroies = pd.Categorical(['foo','bar','baz','foo','bar']) #从其他Python序列类型直接生成pandas.Categorical
my_categroies

categories = ['foo','bar','baz']
codes = [0,1,2,0,0,1]
my_cats_2 = pd.Categorical.from_codes(codes,categories)

#指定顺序
ordered_cat = pd.Categorical.from_codes(codes,categories,ordered = True)
#未排序的分类实例可以使用as_ordered进行排序
my_cats_2.as_ordered()
#分类数据可以不是字符串，一个分类数组可以包含任一不可变的值类型。

#使用Categorical对象进行计算
np.random.seed(12345)
draws = np.random.randn(1000)
draws[:5]

#提取四分位分箱
bins = pd.qcut(draws,4)
#给四分位命名
bins = pd.qcut(draws,4,labels = ['Q1','Q2','Q3','Q4'])
bins.codes[:10] #解析bins的codes属性
bins.categories[:10] #解析bins的categories属性
bins = pd.Series(bins,name = 'quantile') #将bins转化为series对象并命名为quantile
results = (pd.Series(draws).groupby(bins).agg(['count','min','max','mean','sum']).reset_index()) #将draws按照bins进行分组，并计算统计值

##使用分类获得更高性能
N = 10000000
draws = pd.Series(np.random.randn(N))
labels = pd.Series(['foo','bar','baz','qux']*(N//4))
categories = labels.astype('category')
labels.memory_usage()
categories.memory_usage()
#%time _ = labels.astype('category') #计算运行时间

'''
第二节：高阶GroupBy应用
'''

#分类方法
s = pd.Series(['a','b','c','d']*2)
cat_s = s.astype('category')
cat_s.cat.codes
cat_s.cat.categories

#使用set_categories改变类别
actual_categories = ['a','b','c','d','e']
cat_s2 = cat_s.cat.set_categories(actual_categories) #改变类别，将反映在使用他们的操作中
cat_s.value_counts()
cat_s2.value_counts()


#去除未观察到的类别
cat_s3 = cat_s[cat_s.isin(['a','b'])]
cat_s3.cat.remove_unused_categories()

#创建用于建模的虚拟变量

cat_s = pd.Series(['a','b','c','d']*2,dtype = 'category')
pd.get_dummies(cat_s) #pandas.get_dummies函数将一维的分类数据转换为一个包含虚拟变量的DataFrame

df = pd.DataFrame({'key':['a','b','c']*4,'value': np.arange(12.)})
g = df.groupby('key').value
g.mean()
g.transform(lambda x :x.mean()) #
g.transform('mean') #我们可以像GroupBy的agg方法一样传递一个字符串别名
g.transform(lambda x :x*2) #和apply一洋，transform可以与返回Series的函数一起使用
g.transform(lambda x :x.rank(ascending = False)) #对g表做降序排名

def normalize(x):
  return (x-x.mean())/x.std()
g.transform(normalize) #等价于g.apply(normalize)

#内建的聚合函数如’mean’或’sum’通常会比apply函数更快
g.transform('mean')
normalized = (df['value']-g.transform('mean'))/g.transform('std')


#分组的时间重新采样
N = 15
times = pd.date_range('2017-05-20',freq = '1min',periods = N)
df = pd.DataFrame({'time':times,'value':np.arange(N)})
df.set_index('time').resample('5min').count() #我们可以按’time’进行索引，然后重新采样

df2 = pd.DataFrame({'time':times.repeat(3),
                   'key':np.tile(['a','b','c'],N),
                   'value':np.arange(N*3)})
time_key = pd.Grouper(freq = '5min')
resampled = (df2.set_index('time').groupby(['key',time_key]).sum())
resampled.reset_index()

'''
第三节：方法链技术
'''