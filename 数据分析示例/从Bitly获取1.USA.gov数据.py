# -*- coding: utf-8 -*-



import json
path = 'F:/利用Python进行数据分析数据集/datasets/bitly_usagov/example.txt'
open(path).readline()

records = [json.loads(line) for line in open(path)] #读取文件并转化为json格式

time_zones = [rec['tz'] for rec in records if 'tz' in rec] #提取时区列表，在列表推导式的结尾添加一个检查if 'tz'in rec判断是否有值
#time_zones[:10]

##时区计数
#按照时区生成计数，第一种方法，使用字典存储计数
def get_counts(sequence):
  counts = {}
  for x in sequence:
    if x in counts:
      counts[x] += 1
    else:
      counts[x] = 1
  return counts

#使用标准库中的defaultdict进行计数
from collections import defaultdict
def get_counts2(sequence):
  counts = defaultdict(int) #值将会初始化为0
  for x in sequence:
    counts[x] += 1
  return counts

counts = get_counts(time_zones)
counts['America/New_York']
len(time_zones)

#取前十计数
def top_counts(count_dict,n = 10):
  value_key_pairs = [(count,tz) for tz,count in count_dict.items()]
  value_key_pairs.sort()
  return value_key_pairs[-n:]
top_counts(counts)

#使用标准库进行提取前10
#collections.Counter类
from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)

#使用pandas进行时区计数
import pandas as pd
frame = pd.DataFrame(records)
frame.info()
frame['tz'][:10]
tz_counts = frame['tz'].value_counts() #计数后进行降序排序
tz_counts[:10]

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'

#seaborn绘图
import seaborn as sns
subest = tz_counts[:10]
sns.barplot(y = subest.index,x = subest.values)

#获取设备信息
frame['a'][1]
frame['a'][50]
frame['a'][51][:50]

results = pd.Series([x.split()[0] for x in frame.a.dropna()])
results[:5]
results.value_counts()[:8]

#排除a列为空值的行
import numpy as np
cframe = frame[frame.a.notnull()]
cframe['os'] = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')

#根据时区以及操作系统对数据分组
by_tz_os = cframe.groupby(['tz','os'])
agg_counts = by_tz_os.size().unstack().fillna(0) #分组计数可以使用size计算

#在agg_counts中根据行的计数构造一个间接索引数组
#用于升序排列
indexer = agg_counts.sum(1).argsort()
#使用take方法按顺序选出行，之后再对最后10行进行切片（最大的10个值）
count_subset = agg_counts.take(indexer[-10:])
#pandas中有一个便捷的方法叫做nlargest可以做到同样的事情
agg_counts.sum(1).nlargest(10)


#对绘图数据重新排列
count_subset = count_subset.stack()
count_subset.name = 'total'
count_subset = count_subset.reset_index()
#绘图
sns.barplot(x = 'total',y = 'tz',hue = 'os',data = count_subset)

#将数据标准归一化为1,计算Windows在各时区中的占比
def norm_total(group):
  group['normed_total'] = group.total/group.total.sum()
  return group
results = count_subset.groupby('tz').apply(norm_total)
sns.barplot(x = 'normed_total',y = 'tz',hue = 'os',data = results)

#可以使用transform方法更有效的计算归一化之和
g = count_subset.groupby('tz')
results2 = count_subset.total/g.total.transform('sum')