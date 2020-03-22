# -*- coding: utf-8 -*-

'''
第一节：日期和时间数据的类型及工具
'''
import matplotlib.pyplot as plt
from datetime import datetime
now = datetime.now()
now.year,now.month,now.day
delta = datetime(2020,3,1)-datetime(2020,1,1,8,30)
delta.days

#日期加减
from datetime import timedelta
start = datetime(2011,1,7)
start+timedelta(12)
start - 2*timedelta(12)

#日期格式转换
stamp = datetime(2011,1,3)
str(stamp)
stamp.strftime('%Y-%m-%d')


#解析日期格式
from dateutil.parser import parse
parse('2011-01-03')
#注：dateutil.parser会将一些字符串识别为一些不想要的日期，例如：42将会被解析为2042年的当前日期


#使用pandas解析文本日期
import pandas as pd
datestrs = ['2011-07-06 12:00:00','2011-08-06 00:00:00']
pd.to_datetime(datestrs)

idx = pd.to_datetime(datestrs+[None])
pd.isnull(idx) #判断idx值是否有空值
#datetime对象日期特殊地区格式化
now.strftime('%a')
now.strftime('%A')
now.strftime('%b')
now.strftime('%B')
now.strftime('%c')
now.strftime('%p')
now.strftime('%x')
now.strftime('%X')
now.strftime('%X%p')

'''
第二节：时间序列基础
'''

from datetime import datetime
import numpy as np
dates = [datetime(2011,1,2),datetime(2011,1,5),
         datetime(2011,1,7),datetime(2011,1,8),
         datetime(2011,1,10),datetime(2011,1,12)]
ts = pd.Series(np.random.randn(6),index = dates) #这些datetime对象可以被放入datetimeIndex中
ts+ts[::2]#会将ts中每隔一个的元素选择出
stamp = ts.index[2]
ts[stamp]
ts['1/10/2011'] 
ts['20110110']#还可传入能解释为日期的字符串

#使用年份或年月来选择数据切片
longer_ts = pd.Series(np.random.randn(1000),
                      index = pd.date_range('1/1/2000',periods = 1000)) #periods参数表示往后推多少
longer_ts
longer_ts['2001'] #根据年份切片
longer_ts['2001-05'] #根据月份进行切片
ts[datetime(2011,1,7):] #根据datetime对象进行切片
ts['1/6/2011':'1/11/2011'] #使用不包含在时间序列中的时间戳进行切片
ts.truncate(after = '1/9/2011') #truncate实例可通过限制结束日期进行切片


#在DataFrame进行索引
dates = pd.date_range('1/1/2000',periods = 100,freq = 'W-WED')
long_df = pd.DataFrame(np.random.randn(100,4),
                       index = dates,
                       columns = ['Colorado','Texas','New York','Ohio'])
long_df['2000-05']


#含有重复索引的时间序列
dates = pd.DatetimeIndex(['1/1/2000','1/2/2000','1/2/2000','1/2/2000','1/3/2000'])
dup_ts = pd.Series(np.arange(5),index = dates)
dup_ts.index.is_unique #检测索引是否为唯一值
dup_ts['2000/1/3']
dup_ts['2000/1/2']

#聚合含有非唯一时间戳的数据
grouped = dup_ts.groupby(level = 0) #使用groupby并传递level = 0
grouped.mean()


'''
第三节：日期范围、频率和移位
'''
resampler = ts.resample('D') #将ts中每日频率空缺的日期补齐，并把数值列展现为Nan
index = pd.date_range('2012-04-01','2012-06-01')
pd.date_range(start = '2020-01-01',periods = 30) #输入一个开始时间将去periods天的所有日期,输入end日期则往前推periods天的每日日期

pd.date_range('2000-01-01','2000-12-01',freq = 'M') #取月底最后一天日期
pd.date_range('2000-01-01','2000-12-01',freq = 'BM') #取月底工作日最后一天日期
pd.date_range('2012-05-02 12:56:21',periods = 5,normalize = True) #将时间格式化为标准零时


#频率和日期偏置
from pandas.tseries.offsets import Hour,Minute
hour = Hour()
minute = Minute()

pd.date_range('2000-01-01','2000-01-03 23:59',freq = '4h') #创建间隔为4h的时间频率
pd.date_range('2000-01-01',periods = 10,freq = '1h30min') #创建间隔为1.5h的时间频率为10的所有时间
rng = pd.date_range('2012-01-01','2012-09-01',freq = 'WOM-3FRI') #获取每月第三个星期五的日期
list(rng)


ts = pd.Series(np.random.randn(4),
               index = pd.date_range('2000/1/1',periods = 4,freq = 'M'))
ts.shift(2) #整数为向后移动位数，负数代表向前移动位数，仅移动数据，并不会移动index
ts/ts.shift(1)-1 #时间序列环比写法

ts.shift(2,freq = 'M') #加入freq参数则数据往后填充
ts.shift(2,freq = 'D') #datetimeIndex表示往后推2天
ts.shift(1,freq = '90T') #表示往后推1个90分钟，T代表分钟

#使用偏置进行移位日期

from pandas.tseries.offsets import Day,MonthEnd
now = datetime(2011,11,17)
now+3*Day() #表示往后推3天日期
now+MonthEnd() #表示本月月底日期
now+MonthEnd(2) #表示下月月底日期

offset = MonthEnd()
offset.rollforward(now)
offset.rollback(now)

#将位移方法与groupby一起使用是日期偏置的一种创造性用法
ts = pd.Series(np.random.randn(20),
               index = pd.date_range('2000/1/15',periods = 20,freq = '4d'))
ts.groupby(MonthEnd().rollforward).mean() #按照每月月底进行本月平均值
ts.resample('M').mean()

'''
第四节：时区处理(略)
'''
import pytz
pytz.common_timezones[-5:]

tz = pytz.timezone('America/New_York') 
rng = pd.date_range('2012/3/9 9:30',periods = 6,freq = 'D')
ts = pd.Series(np.random.randn(len(rng)),index = rng)

pd.date_range('2012/3/9 9:30',periods = 10,freq = 'D',tz = 'UTC') #通过时区集合生成日期
ts_utc = ts.tz_localize('UTC') #tz_localize方法可以从简单时区转换到本地化时区
ts_utc.index

ts_utc.tz_convert('America/New_York')
ts_eastern = ts.tz_localize('America/New_York')
ts_eastern.tz_convert('UTC')

'''
第五节：时间区间和区间算术
'''
p = pd.Period(2007,freq = 'A-DEC') #表示2007年整年
#使用period_range函数构造规则区间序列
rng = pd.period_range('2000-01-01','2000-6-30',freq = 'M')
pd.Series(np.random.randn(6),index = rng)
values = ['2001Q3','2002Q2','2003Q1']
index = pd.PeriodIndex(values,freq = 'Q-DEC') #将字符串数组转换成季度数组

#区间频率转换
p = pd.Period('2007',freq = 'A-DEC')
p.asfreq('M',how = 'start') #表示2017-01
p.asfreq('M',how = 'end') #表示2017-12

p = pd.Period('2007',freq = 'A-JUN')
p=pd.Period('Aug-2007','M') 
p.asfreq('A-JUN')

rng = pd.period_range('2006','2009',freq = 'A-DEC')
ts = pd.Series(np.random.randn(len(rng)),index = rng)
ts.asfreq('M',how = 'start') #转换为每年第一个月
ts.asfreq('B',how = 'end') #每年最后一个工作日

p = pd.Period('2012Q4',freq = 'Q-DEC') #表示以12月结束的财务年度
p.asfreq('D',how = 'start')
p.asfreq('D',how = 'end')

#获取在季度倒数第二个工作日下午4点的时间戳
p4pm = (p.asfreq('B','e')-1).asfreq('T','s')+16*60 
p4pm.to_timestamp()

#计算日期区间，返回导数第二个工作日下午4点，并更改为ts索引
rng = pd.period_range('2011Q3','2012Q4',freq = 'Q-DEC')
ts = pd.Series(np.arange(len(rng)),index = rng)
new_rng = (rng.asfreq('B','e')-1).asfreq('T','s')+16*60
ts.index = new_rng.to_timestamp()

#将时间戳转换为区间（以及逆转换）
rng = pd.date_range('2000-01-01',periods = 3,freq = 'M')
ts = pd.Series(np.random.randn(3),index = rng)
pts = ts.to_period()
rng = pd.date_range('2000-1-29',periods = 6,freq = 'D')
ts2 = pd.Series(np.random.randn(6),index = rng)
ts2.to_period('M') #转换为月份区间
pts = ts2.to_period() 
pts.to_timestamp(how = 'end') #将区间转换为时间戳


#从数组生成PeriodIndex
data = pd.read_csv('F:/利用Python进行数据分析数据集/examples/macrodata.csv')
data.head(5) #查看数据前5行
data.year
data.quarter
index = pd.PeriodIndex(year = data.year,quarter = data.quarter,freq = 'Q-DEC')
data.index = index
data.infl #读取infl列


'''
第六节：重新采样与频率转换
'''
#日期转换
rng = pd.date_range('2000-01-01',periods = 100,freq = 'D')
ts = pd.Series(np.random.randn(len(rng)),index = rng)
ts.resample('M').mean() #以月底最后一天日期显示的月度汇总
ts.resample('M',kind = 'period').mean() #以月份显示的月度汇总

#向下采样
rng = pd.date_range('2000-01-01',periods = 12,freq = 'T')
ts = pd.Series(np.arange(12),index = rng)
ts.resample('5min',closed = 'right').sum() #所有时间往前推5分钟，并使用开始时间作为index
ts.resample('5min',closed = 'left').sum() #每隔5分钟进行汇总，并使用开始时间作为index
ts.resample('5min',closed = 'right',label = 'right').sum() #所有时间往前推5分钟，并使用最后时间作为index
ts.resample('5min',closed = 'right',label = 'left').sum() #所有时间往前推5分钟，并使用开始时间作为index
ts.resample('5min',closed = 'right',label = 'right',loffset = '-1s').sum() #s使用loffset方法是PeriodIndex向下偏置1s,还可以在结果上应用shift达到同样的效果
ts.resample('5min').ohlc() #ohlc聚合函数获得开端、结束、峰值、谷值

#向上采样与插值
frame = pd.DataFrame(np.random.randn(2,4),
                     index = pd.date_range('2000-1-1',periods = 2,freq = 'W-WED'),
                     columns = ['Colorado','Texas','New York','Ohio'])
df_daily = frame.resample('D').asfreq()
frame.resample('D').ffill()  #填充每天的数值并向上填充
frame.resample('D').ffill(limit = 2) #填充每天的数值并前两列空向上填充
frame.resample('W-THU').ffill() #从每个周三转化为每个周四并向上填充

#使用区间进行重新采样
frame = pd.DataFrame(np.random.randn(24,4),
                     index = pd.period_range('2000-1','2001-12',freq = 'M'),
                     columns = ['Colorado','Texas','New York','Ohio'])
annual_frame = frame.resample('A-DEC').mean() #按照12月结束年份进行聚合平均
annual_frame.resample('Q-DEC').ffill() #Q-DEC每季度，年末在12月
annual_frame.resample('Q-DEC',convention = 'end').ffill()
annual_frame.resample('Q-MAR').ffill()

'''
第七章：移动窗口函数
'''
close_px_all = pd.read_csv('F:/利用Python进行数据分析数据集/examples/stock_px_2.csv',
                           parse_dates = True,index_col = 0)
close_px = close_px_all[['AAPL','MSFT','XOM']]
close_px = close_px.resample('B').ffill()
close_px.AAPL.plot()
close_px.AAPL.rolling(250).mean().plot() #表达式rolling(250)与groupby的行为类似，但是它创建的对象是根据250日滑动窗口分组的而不是直接分组
appl_std250 = close_px.AAPL.rolling(250,min_periods=10).std()
appl_std250[5:12]
appl_std250.plot() #为了计算扩展窗口均值，使用expanding算子，而不是rolling。扩展均值从时间序列的起始位置开始时间窗口，并增加窗口的大小，直到它涵盖整个序列。apple_std250的扩展均值窗口如下：
expanding_mean = appl_std250.expanding().mean() #
close_px.rolling(60).mean().plot(logy = True)
close_px.rolling('20D').mean() #计算20天的滚动平均值

#指数加权函数
aapl_px = close_px.AAPL['2006':'2007']
ma60 = aapl_px.rolling(30,min_periods=20).mean()
ewma60 = aapl_px.ewm(span = 30).mean()
ma60.plot(style = 'k--',label = 'Simple MA')
ewma60.plot(style='k-',label = 'EW MA')
plt.legend()

#二元移动窗口函数
spx_px = close_px_all['SPX']
spx_rets = spx_px.pct_change()
returns = close_px.pct_change()
corr = returns.AAPL.rolling(125,min_periods=100).corr(spx_rets)  #min_periods表示窗口需要最少个数为100个
corr.plot()

from scipy.stats import percentileofscore
score_at_2percent = lambda x :percentileofscore(x,0.02)
result = returns.AAPL.rolling(250).apply(score_at_2percent)
result.plot()
