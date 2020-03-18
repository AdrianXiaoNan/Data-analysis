import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Series绘制折线图
s = pd.Series(np.random.randn(10).cumsum(),index = np.arange(0,100,10))
s.plot()


#DataFrame绘制折线图
df = pd.DataFrame(np.random.randn(10,4).cumsum(0),
                  columns = ['A','B','C','D'],
                  index = np.arange(0,100,10))
df.plot()


#绘制水平和垂直柱状图
fig,axes = plt.subplots(2,1)
data = pd.Series(np.random.rand(16),index = list('abcdefghijklmnop'))
data.plot.bar(ax = axes[0],color = 'k',alpha = 0.4)
data.plot.barh(ax = axes[1],color = 'k',alpha = 0.7)


#绘制垂直柱状图，将每一行的值分组并排到柱子中的一组
df = pd.DataFrame(np.random.rand(6,4),
                  index = ['one','two','three','four','five','six'],
                  columns = pd.Index(['A','B','C','D'],name = 'Genus'))
df.plot.bar()
df.plot.barh(stacked= True,alpha = 0.5) #stacked=True来生成堆积柱状图

#
tips = pd.read_csv('F:/利用Python进行数据分析数据集/examples/tips.csv')
party_counts = pd.crosstab(tips['day'],tips['size'])
#print(party_counts)
party_counts = party_counts.loc[:,2:5]
party_pcts = party_counts.div(party_counts.sum(1),axis = 0) #标准化以确保每一行的值和为1
party_pcts.plot.bar()

import seaborn as sns

tips['tip_pct'] = tips['tip']/(tips['total_bill']-tips['tip'])
#print(tips.head())
sns.barplot(x = 'tip_pct', y = 'day',data=tips,orient='h')  #根据星期日期和时间计算的小费百分比
sns.barplot(x = 'tip_pct', y = 'day',data=tips,orient='h',hue = 'time') #根据time列分离数据
sns.set(style = 'whitegrid') # 在不同的绘图外观中进行切换
tips['tip_pct'].plot.hist(bins = 50) #直方图
tips['tip_pct'].plot.density() #密度图


#distplot绘制直方图和连续密度估计
comp1 = np.random.normal(0,1,size = 200)
comp2 = np.random.normal(10,2,size = 200)
values = pd.Series(np.concatenate([comp1,comp2]))
sns.distplot(values,bins = 100,color = 'k') #绘制直方图和连续密度估计


#散点图和点图
macro = pd.read_csv('F:/利用Python进行数据分析数据集/examples/macrodata.csv')
data = macro[['cpi','m1','tbilrate','unemp']]
trans_data = np.log(data).diff().dropna()
plt.title('Change in log %s versus log %s'%('m1','unemp')) #标题
sns.regplot('m1','unemp',data = trans_data,color = 'k') #绘制散点图，并拟合出一个条线性回归线
sns.pairplot(trans_data,diag_kind = 'kde',plot_kws = {'alpha':0.2}) #在对角线上放置每个变量的直方图或密度估计值，plot_ksw参数，这个参数使我们能够将配置选项传递给非对角元素上的各个绘图调用
sns.factorplot(x = 'day',y = 'tip_pct',hue = 'time',col = 'smoker',kind = 'bar',data = tips[tips.tip_pct<1]) #对数据进行分类画图
sns.factorplot(x = 'day',y = 'tip_pct',row = 'time',col = 'smoker',kind = 'bar',data = tips[tips.tip_pct<1]) #除了可以将time分配到hue参数，还可以通过time分配到row参数使每个值增加一行分面网格
sns.factorplot(x = 'tip_pct',y = 'day',kind = 'box',data = tips[tips.tip_pct<0.5]) #通过kind参数来选择你要用的图类型
plt.show()