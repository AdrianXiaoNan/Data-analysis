'''
第一节：Groupby机制

'''
import pandas as pd
import numpy as np

df = pd.DataFrame({'key1':['a','a','b','b','a'],
                   'key2':['one','two','one','two','one'],
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})

grouped = df['data1'].groupby(df['key1']) #生成groupby的对象
means = df['data1'].groupby([df['key1'],df['key2']]).mean() #加入计算，生成具体的DataFrame

#分组键都是Series，尽管分组键也可以是正确长度的任何数组
states = np.array(['Ohio','California','California','Ohio','Ohio'])
years = np.array([2005,2005,2006,2005,2006])
df['data1'].groupby([states,years]).mean()

#分组信息作为你想要继续处理的数据，通常包含在同一个DataFrame中
df.groupby('key1').mean()
df.groupby(['key1','key2']).mean()
df.groupby(['key1','key2']).size() #返回groupby组大小信息的Series


#遍历各分组

for name,group in df.groupby('key1'):
  print(name)
  print(group)
  
#在多个分组键的情况下，元组中的第一个元素是键值的元组
for (k1,k2),group in df.groupby(['key1','key2']):
  print((k1,k2))
  print(group)
  
#计算出数据块的字典
pieces = dict(list(df.groupby('key1')))
pieces['b']

#根据dtype对我们的示例df的列进行分组
grouped = df.groupby(df.dtypes,axis = 1)
for dtype,group in grouped:
  print(dtype)
  print(group)
  
#用列名称或列名称数组进行索引
df.groupby('key1')['data1'] #等价于df['data1'].groupby(df['key1'])
df.groupby('key1')[['data2']] #等价于df[['data2']].groupby(df['key1'])

#选择一列或所有列的子集
s_grouped = df.groupby(['key1','key2'])['data2']
print(s_grouped)
print(s_grouped.mean())

#使用字典和Series分组
people = pd.DataFrame(np.random.randn(5,5),
                      columns = ['a','b','c','d','e'],
                      index = ['Joe','Steve','Wes','Jim','Travis'])
people.iloc[2:3,[1,2]] = np.nan #增加空值

#可传入字典进行聚合操作
mapping = {'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}
by_column = people.groupby(mapping,axis = 1)
by_column.sum()

#即使传入Series也可以达到与字典相同的目的
map_series = pd.Series(mapping)
people.groupby(map_series,axis = 1).count()



#使用函数进行分组
people.groupby(len).sum()
key_list = ['one','one','one','two','two']
people.groupby([len,key_list]).min()



#根据索引层级分组
columns = pd.MultiIndex.from_arrays([['US','US','US','JP','JP'],[1,3,5,1,3]],names = ['cty','tenor'])
hier_df = pd.DataFrame(np.random.randn(4,5),columns = columns)
hier_df.groupby(level = 'cty',axis = 1).count()


'''
第二节：数据聚合
'''

grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)
def peak_to_peak(arr):
  return arr.max()-arr.min()
grouped.agg(peak_to_peak)


#逐列及多函数应用
tips = pd.read_csv('F:/利用Python进行数据分析数据集/examples/tips.csv')
tips['tip_pct'] = tips['tip']/tips['total_bill']
tips[:6]
grouped = tips.groupby(['day','smoker'])
grouped_pct = grouped['tip_pct']
grouped_pct.agg('mean')
grouped_pct.agg(['mean','std',peak_to_peak])

#传入(name, function)元组的列表，将第一个参数显示为列名
grouped_pct.agg([('foo','mean'),('bar',np.std)]) 

#指定应用到所有列的函数列表
functions = ['count','mean','max']
#选择需要进行计算的列
result = grouped['tip_pct','total_bill'].agg(functions)

#可以传入具有自定义名称的元组列表
ftuples = [('Dur','mean'),('Abw',np.var)]
grouped['tip_pct','total_bill'].agg(ftuples)

#想要将不同的函数应用到一个或多个列上。要实现这个功能，需要将含有列名与函数对应关系的字典传递给agg
grouped.agg({'tip':np.max,'size':'sum'}) 
grouped.agg({'tip_pct':['min','max','mean','std'],'size':'sum'})

#通过向groupby传递as_index=False来禁用分组键作为索引的行为
tips.groupby(['day','smoker'],as_index = False).mean()


'''
第三节：应用：通用拆分-应用-联合
'''

#按组选出小费百分比（tip-pct）最高的五组
def top(df,n = 5,column = 'tip_pct'):
  return df.sort_values(by = column)[-n:]
top(tips,n=6)

tips.groupby('smoker').apply(top) #按照smoker分组计算最高的5组

#还可以在apply内传入其他参数
tips.groupby(['smoker','day']).apply(top,n=6,column = 'total_bill') 

#在groupby对象上调用describe方法
result = tips.groupby('smoker')['tip_pct'].describe()
result.unstack('smoker') #将结果转成带有多层索引的Series

#调用describe方法简写
f = lambda x :x.describe()
grouped.apply(f)

#压缩分组，在能看到元素对象索引的时候可以使用group_keys来禁用分组
tips.groupby('smoker',group_keys = False).apply(top)


frame = pd.DataFrame({'data1':np.random.randn(1000),
                      'data2':np.random.randn(1000)})
quartiles = pd.cut(frame.data1,4)
quartiles[:10]

def get_stats(group):
  return {'min':group.min(),
          'max':group.max(),
          'count':group.count(),
          'mean':group.mean()}
grouped = frame.data2.groupby(quartiles)
grouped.apply(get_stats).unstack()

#根据样本分位数计算出等大小的桶,则需要使用qcut。我们将传递labels=False来获得分位数数值
grouping = pd.qcut(frame.data1,10,labels = False)
grouped = frame.data2.groupby(grouping)
grouped.apply(get_stats).unstack()


s = pd.Series(np.random.randn(6))
s[::2] = np.nan #取间隔为1的空
s.fillna(s.mean()) #使用均值填充空值


states = ['Ohio','New York','Vermont','Florida','Oregon','Nevada','California','Idaho']
group_key = ['East']*4+['West']*4
data = pd.Series(np.random.randn(8),index = states)

data[['Vermont','Nevada','Idaho']] = np.nan
data.groupby(group_key).mean()

#使用分组的平均值来填充NA值
fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)

#已经在代码中卫每个分组预定了填充值
fill_values = {'East':0.5,'West':-1}
fill_func = lambda g:g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)

#红桃、黑桃、梅花、方块
#随机抽样
suits = ['H','S','C','D']
card_val = (list(range(1,11))+[10]*3)*4
base_names = ['A']+list(range(2,11))+['J','Q','K']
cards = []
for suit in ['H','S','C','D']:
  cards.extend(str(num)+suit for num in base_names)
deck = pd.Series(card_val,index = cards)

#拿出5张牌
def draw(deck,n=5):
  return deck.sample(n) #sample函数表示随机抽样，n代表个数
draw(deck)

get_suit = lambda card:card[-1]
deck.groupby(get_suit).apply(draw,n=2)
deck.groupby(get_suit,group_keys = False).apply(draw,n=2)


#分组加权平均和相关性
df = pd.DataFrame({'category':['a','a','a','a','b','b','b','b'],
                   'data':np.random.randn(8),
                   'weights':np.random.rand(8)})
grouped = df.groupby('category')
get_wavg = lambda g:np.average(g['data'],weights = g['weights'])
grouped.apply(get_wavg)


#分组相关性
close_px = pd.read_csv('F:/利用Python进行数据分析数据集/examples/stock_px_2.csv',
                       parse_dates = True,
                       index_col = 0)
close_px.info()  #查看文件信息

spx_corr = lambda x:x.corrwith(x['SPX']) #计算每列与SPX列成对关联的函数
rets = close_px.pct_change().dropna() #我们使用pct_change计算close-px百分比的变化
get_year = lambda x:x.year
by_year = rets.groupby(get_year)
by_year.apply(spx_corr)
by_year.apply(lambda g:g['AAPL'].corr(g['MSFT'])) #计算aapl与msft相关性


#逐组线性回归,调用最小二乘回归（OLS）
import statsmodels.api as sm
def regress(data,yvar,xvars):
  Y = data[yvar]
  X = data[xvars]
  X['intercept'] = 1
  result = sm.OLS(Y,X).fit()
  return result.params
by_year.apply(regress,'AAPL',['SPX']) #要计算AAPL在SPX回报上的年度线性回归


'''
第四节：数据透视表与交叉表
'''
tips.pivot_table(index = ['day','smoker'])
#假设我们只想在tip_pct和size上进行聚合，并根据time分组。我将把smoker放入表的列，而将day放入表的行
tips.pivot_table(['tip_pct','size'],index = ['time','day'],columns = 'smoker',margins=True) #通过传入margins = True来填充部分总计，All的值是均值，且该均值是不考虑吸烟者与非吸烟者（All列）或行分组中任何两级的（All行）
tips.pivot_table('tip_pct',index = ['time','size','smoker'],columns = 'day',aggfunc = 'mean',margins = True,fill_value = 0)


#交叉表
pd.crosstab([tips.time,tips.day],tips.smoker,margins = True)