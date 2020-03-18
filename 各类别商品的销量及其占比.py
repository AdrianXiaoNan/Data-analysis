# -*- coding: utf-8 -*-

import pandas as pd

file1 = 'F:\Python数据分析与挖掘实战（第2版）\chapter8\demo\data\GoodsOrder.csv'
file2 = 'F:\Python数据分析与挖掘实战（第2版）\chapter8\demo\data\GoodsTypes.csv'

data = pd.read_csv(file1,encoding='gbk')
types = pd.read_csv(file2,encoding='gbk')

group = data.groupby(['Goods']).count().reset_index()
sort = group.sort_values('id',ascending=False).reset_index()
data_nums = data.shape[0]
del sort['index']
sort_links = pd.merge(sort,types) #根据type合并两个data

#根据类别求和，每个商品类别的总和，并排序
sort_link = sort_links.groupby(['Types']).sum().reset_index()
sort_link = sort_link.sort_values('id',ascending=False).reset_index()
del sort_link['index'] #删除index列
#求百分比并更换列名，最后输出到文件
sort_link['count'] = sort_link.apply(lambda line:line['id']/data_nums,axis=1) #计算百分比
sort_link.rename(columns={'count':'percent'},inplace=True) #更改列名count为percent
print('各商品类别销售及其占比：\n',sort_link)
outfile1 = 'F:\Python数据分析与挖掘实战（第2版）\chapter8\demo\data\percent.csv'
sort_link.to_csv(outfile1,index=False,header=True,encoding='utf_8_sig') #保存文件


#画饼图展示每类商品的销量占比
import matplotlib.pyplot as plt
data = sort_link['percent']
labels = sort_link['Types']
plt.figure(figsize=(8,6)) #画布大小
plt.pie(data,labels=labels,autopct='%1.2f%%') #autopct标记百分比注释
plt.rcParams['font.sans-serif']='SimHei'
plt.title('每类商品销售占比')
plt.show() 


#非酒精饮料类型的商品然后求百分比并输出到文件
selected = sort_links.loc[sort_links['Types'] == '非酒精饮料']
#排序
child_sums = selected['id'].sum() #对所有非酒精饮料进行求和
selected['child_percent'] = selected.apply(lambda line:line['id']/child_sums,axis=1) #求百分比
selected.rename(columns = {'id':'count'},inplace=True)
print('非酒精饮料内部商品的销量及其占比：\n',selected)
outfile2 = 'F:\Python数据分析与挖掘实战（第2版）\chapter8\demo\data\child_percent.csv'
selected.to_csv(outfile2,index = False,header = True,encoding='utf_8_sig') #输出结果

#画饼图展示非酒精饮料内部各商品的销售占比
data = selected['child_percent']
labels = selected['Goods']
plt.figure(figsize = (8,6)) #画布大小
explode = (0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12) #设置每一块分隔的间隙大小
plt.pie(data,explode=explode,labels = labels,autopct = '%1.2f%%',pctdistance = 1.1,labeldistance = 1.2)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title('非酒精饮料内部销量占比')
plt.axis('equal')
plt.show()
