# -*- coding: utf-8 -*-

#销量前10的商品销量及其占比

import pandas as pd
data = 'F:\Python数据分析与挖掘实战（第2版）\chapter8\demo\data\GoodsOrder.csv'
data = pd.read_csv(data,encoding='gbk')

#对商品进行分类汇总
group_qty = data.groupby(['Goods']).count().reset_index()

#sort_values()函数是按照选中索引所在列的原素进行排序
sort = group_qty.sort_values('id',ascending=False)

#排序并查看前十的商品
print('销量排名前10的商品销量: \n',sort[:10])

#画条形图展示销量排名前十的产品销量
#导入绘图模块
import matplotlib.pyplot as plt
x = sort[:10]['Goods']
y = sort[:10]['id']
plt.figure(figsize=(8,4)) #设置画布大小
plt.barh(x,y) 
plt.rcParams['font.sans-serif']='SimHei'
#设置x轴标题
plt.xlabel('销量')
#设置y轴标题
plt.ylabel('商品类别')
#设置图表标题
plt.title('商品销量TOP10')
#plt.savefig('路径')   -- 保存作图
plt.show()
#销量排名前10的销量占比
data_nums = data.shape[0]
for index,row in sort[:10].iterrows():
  x = row['Goods']
  y = row['id']
  z = '{:.2%}'.format(row['id']/data_nums)
  print(x,y,z)