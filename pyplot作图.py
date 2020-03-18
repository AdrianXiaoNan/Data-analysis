import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

plt.plot(np.random.randn(50).cumsum(),'k--') #最后一图增加数据
_ = ax1.hist(np.random.randn(100),bins = 20,color = 'k',alpha = 0.3) #1图增加数据
ax2.scatter(np.arange(30),np.arange(30)+3*np.random.randn(30)) #2图增加数据
plt.show()

# subplots_adjust(left = None,
#                 bottom = None,
#                 right = None,
#                 top = None,
#                 wspace = None,
#                 hspace = None) #wspace和hspace分别控制的是图片的宽度和高度百分比，以用作子图间的间距

fig,axes = plt.subplots(2,2,sharex=True,sharey=True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500),bins = 50,color = 'g',alpha = 0.5)
plt.subplots_adjust(wspace = 0,hspace = 0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum(),color = 'k') #设置折线图颜色为黑色
ticks = ax.set_xticks([0,250,500,750,1000]) #设置X轴刻度
labels = ax.set_xticklabels(['one','two','three','four','five'],rotation = 30,fontsize = 'small') #设置X轴刻度
yticks = ax.set_yticks([-50,-36,-22,-8,6,20]) #设置Y轴刻度
props = {
    'title':'My first matplotlib plot',
    'xlabel':'Stages',
    'ylabel':'Number'
} #设置图表标题和X、Y轴标题
ax.set(**props)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum(),'k',label = 'one')
ax.plot(np.random.randn(1000).cumsum(),'k--',label = 'two')
ax.plot(np.random.randn(1000).cumsum(),'k.',label = 'three')
ax.legend(loc = 'best') #添加图例
plt.show()


from datetime import datetime
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
data = pd.read_csv('F:/利用Python进行数据分析数据集/examples/spx.csv',index_col = 0,parse_dates = True)
spx = data['SPX']
spx.plot(ax=ax,style = 'k-')
#添加注释的时间及注释内容
crisis_data = [
    (datetime(2007,10,11),'Peak of bull market'),
    (datetime(2008,3,12),'Bear Stearns Fails'),
    (datetime(2008,9,15),'Lehman Bankruptcy')
]
for date,label in crisis_data:
    ax.annotate(label,
                xy = (date,spx.asof(date)+75),
                xytext = (date,spx.asof(date)+225),
                arrowprops = dict(facecolor = 'black',
                                  headwidth = 4,
                                  width = 2,
                                  headlength = 4),
                horizontalalignment = 'left',
                verticalalignment = 'top') #给图表添加注释
#放大2007年到2010年
ax.set_xlim(['1/1/2007','1/1/2011'])
ax.set_ylim([600,1800])
ax.set_title('Important dates in the 2008-2009 financial crisis')
plt.savefig('crisis.png',dpi = 400,bbox_inches = 'tight') #dpi分辨率400dpi,bbox_inches修剪实际图像的空白
plt.show()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rect = plt.Rectangle((0.2,0.75),0.4,0.15,color = 'k',alpha = 0.3) #矩形
circ = plt.Circle((0.7,0.2),0.15,color = 'b',alpha = 0.3) #圆形
pgon = plt.Polygon([[0.15,0.15],[0.35,0.4],[0.2,0.6]],color = 'g',alpha = 0.5) #三角形
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
plt.show()