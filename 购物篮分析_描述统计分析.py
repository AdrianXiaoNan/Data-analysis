# -*- coding: utf-8 -*-
'''
1.构建零售商品的Apriori关联规则模型，分析商品之间的关联性
2.根据模型结果给出销售策略
'''

'''
购物篮关联规则挖掘的主要步骤如下：
1）对原始数据进行数据探索性分析，分析商品的热销情况与商品结构。
2）对原始数据进行数据预处理，转换数据形式，使之符合Apriori关联规则算法要求。
3）在步骤2得到的建模数据基础上，采用Apriori关联规则算法调整模型输入参数，完成商品关联性分析。
4）结合实际业务，对模型结果进行分析，根据分析结果给出销售建议，最后输出关联规则结果。
'''

import pandas as pd
import numpy as np
#输入数据文件
inputfile = 'F:\Python数据分析与挖掘实战（第2版）\chapter8\demo\data\GoodsOrder.csv'
#读取数据
data = pd.read_csv(inputfile,encoding='gbk')
#查看数据属性
#data.info()

data = data['id']

#依次计算总数、最小值、最大值
description = [data.describe()]

#将结果加入数据框，并进行转置
description = pd.DataFrame(description).T
print('描述性统计结果：\n',np.round(description))
