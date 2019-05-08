# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:47:01 2019

@author: shaowu
"""

import gc
import os
import re
import h5py
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.rcParams['font.size'] = 20 #坐标刻度字体大小
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tqdm import *
from sklearn import preprocessing
seed =42
np.random.seed = seed

#读入数据：
print('读入原始数据...')
train_path='赛题1数据集/'
train1 = pd.read_excel(train_path+'产品.xlsx')
train2 = pd.read_excel(train_path+'工商基本信息表.xlsx')
train3 = pd.read_excel(train_path+'购地-地块公示.xlsx')
train4 = pd.read_excel(train_path+'购地-房地产大地块出让情况.xlsx')
train5 = pd.read_excel(train_path+'购地-房地产大企业购地情况.xlsx')
train6 = pd.read_excel(train_path+'购地-结果公告.xlsx')
train7 = pd.read_excel(train_path+'购地-市场交易-土地抵押.xlsx')
train8 = pd.read_excel(train_path+'购地-市场交易-土地转让.xlsx')
train9 = pd.read_excel(train_path+'海关进出口信用.xlsx')
train10 = pd.read_excel(train_path+'竞品.xlsx')
train11 = pd.read_excel(train_path+'纳税A级年份.xlsx')
train12 = pd.read_excel(train_path+'年报-的对外提供保证担保信息.xlsx')
train13 = pd.read_excel(train_path+'年报-对外投资信息.xlsx')
train14 = pd.read_excel(train_path+'年报-股东（发起人）及出资信息.xlsx')
train15 = pd.read_excel(train_path+'年报-股东股权转让.xlsx')
train16 = pd.read_excel(train_path+'年报-企业基本信息.xlsx')
train17 = pd.read_excel(train_path+'年报-企业资产状况信息.xlsx')
train18 = pd.read_excel(train_path+'年报-社保信息.xlsx')
train19 = pd.read_excel(train_path+'年报-网站或网点信息.xlsx')
train20 = pd.read_excel(train_path+'融资信息.xlsx')
train21 = pd.read_excel(train_path+'软著著作权.xlsx')
train22 = pd.read_excel(train_path+'商标.xlsx')
train23 = pd.read_excel(train_path+'上市公司财务信息-每股指标.xlsx')
train24 = pd.read_excel(train_path+'上市信息财务信息-财务风险指标.xlsx')
train25 = pd.read_excel(train_path+'上市信息财务信息-成长能力指标.xlsx')
train26 = pd.read_excel(train_path+'上市信息财务信息-利润表.xlsx')
train27 = pd.read_excel(train_path+'上市信息财务信息-现金流量表.xlsx')
train28 = pd.read_excel(train_path+'上市信息财务信息盈利能力指标.xlsx')
train29 = pd.read_excel(train_path+'上市信息财务信息运营能力指标.xlsx')
train30 = pd.read_excel(train_path+'上市信息财务信息资产负债表.xlsx')
train31 = pd.read_excel(train_path+'项目信息.xlsx')
train32 = pd.read_excel(train_path+'一般纳税人.xlsx')
train33 = pd.read_excel(train_path+'债券信息.xlsx')
train34 = pd.read_excel(train_path+'招投标.xlsx')
train35 = pd.read_excel(train_path+'专利.xlsx')
train36 = pd.read_excel(train_path+'资质认证.xlsx')
train37 = pd.read_excel(train_path+'作品著作权.xlsx')
label= pd.read_excel(train_path+'企业评分.xlsx')
label=label.drop_duplicates().reset_index(drop=True)  ##训练集标签
print('the size of label is:',len(label))
train_id_list=list(label['企业编号'])
final_submit= pd.read_excel('赛题1结果_团队名.xlsx',header=None) #读入待提交示例
final_submit.columns=['企业编号','企业总评分']
final_submit['企业总评分']=-1
all_label=pd.concat([label,final_submit],axis=0)

test_path='赛题1测试数据集/'
test1 = pd.read_excel(test_path+'产品.xlsx')
test2 = pd.read_excel(test_path+'工商基本信息表.xlsx')
test3 = pd.read_excel(test_path+'购地-地块公示.xlsx')
test4 = pd.read_excel(test_path+'购地-房地产大地块出让情况.xlsx')
test5 = pd.read_excel(test_path+'购地-房地产大企业购地情况.xlsx')
test6 = pd.read_excel(test_path+'购地-结果公告.xlsx')
test7 = pd.read_excel(test_path+'购地-市场交易-土地抵押.xlsx')
test8 = pd.read_excel(test_path+'购地-市场交易-土地转让.xlsx')
test9 = pd.read_excel(test_path+'海关进出口信用.xlsx')
test10 = pd.read_excel(test_path+'竞品.xlsx')
test11 = pd.read_excel(test_path+'纳税A级年份.xlsx')
test12 = pd.read_excel(test_path+'年报-的对外提供保证担保信息.xlsx')
test13 = pd.read_excel(test_path+'年报-对外投资信息.xlsx')
test14 = pd.read_excel(test_path+'年报-股东（发起人）及出资信息.xlsx')
test15 = pd.read_excel(test_path+'年报-股东股权转让.xlsx')
test16 = pd.read_excel(test_path+'年报-企业基本信息.xlsx')
test17 = pd.read_excel(test_path+'年报-企业资产状况信息.xlsx')
test18 = pd.read_excel(test_path+'年报-社保信息.xlsx')
test19 = pd.read_excel(test_path+'年报-网站或网点信息.xlsx')
test20 = pd.read_excel(test_path+'融资信息.xlsx')
test20.columns=['企业总评分']+list(test20.columns[1:])
test21 = pd.read_excel(test_path+'软著著作权.xlsx')
test21.columns=['企业总评分']+list(test21.columns[1:])
test22 = pd.read_excel(test_path+'商标.xlsx')
test22.columns=['企业总评分']+list(test22.columns[1:])
test23 = pd.read_excel(test_path+'上市公司财务信息-每股指标.xlsx')
test23.columns=['企业总评分']+list(test23.columns[1:])
test24 = pd.read_excel(test_path+'上市信息财务信息-财务风险指标.xlsx')
test24.columns=['企业总评分']+list(test24.columns[1:])
test25 = pd.read_excel(test_path+'上市信息财务信息-成长能力指标.xlsx')
test25.columns=['企业总评分']+list(test25.columns[1:])
test26 = pd.read_excel(test_path+'上市信息财务信息-利润表.xlsx')
test26.columns=['企业总评分']+list(test26.columns[1:])
test27 = pd.read_excel(test_path+'上市信息财务信息-现金流量表.xlsx')
test27.columns=['企业总评分']+list(test27.columns[1:])
test28 = pd.read_excel(test_path+'上市信息财务信息盈利能力指标.xlsx')
test28.columns=['企业总评分']+list(test28.columns[1:])
test29 = pd.read_excel(test_path+'上市信息财务信息运营能力指标.xlsx')
test29.columns=['企业总评分']+list(test29.columns[1:])
test30 = pd.read_excel(test_path+'上市信息财务信息资产负债表.xlsx')
test30.columns=['企业总评分']+list(test30.columns[1:])
test31 = pd.read_excel(test_path+'项目信息.xlsx')
test31.columns=['企业总评分']+list(test31.columns[1:])
test32 = pd.read_excel(test_path+'一般纳税人.xlsx')
test32.columns=['企业总评分']+list(test32.columns[1:])
test33 = pd.read_excel(test_path+'债券信息.xlsx')
test33.columns=['企业总评分']+list(test33.columns[1:])
test34 = pd.read_excel(test_path+'招投标.xlsx')
test34.columns=['企业总评分']+list(test34.columns[1:])
test35 = pd.read_excel(test_path+'专利.xlsx')
test35.columns=['企业总评分']+list(test35.columns[1:])
test36 = pd.read_excel(test_path+'资质认证.xlsx')
test36.columns=['企业总评分']+list(test36.columns[1:])
test37 = pd.read_excel(test_path+'作品著作权.xlsx')
test37.columns=['企业总评分']+list(test37.columns[1:])
print('结合训练测试数据一起挖掘特征...')

train1=pd.concat([train1,test1],axis=0).reset_index(drop=True)
train2=pd.concat([train2,test2],axis=0).reset_index(drop=True)
train3=pd.concat([train3,test3],axis=0).reset_index(drop=True)
train4=pd.concat([train4,test4],axis=0).reset_index(drop=True)
train5=pd.concat([train5,test5],axis=0).reset_index(drop=True)
train6=pd.concat([train6,test6],axis=0).reset_index(drop=True)
train7=pd.concat([train7,test7],axis=0).reset_index(drop=True)
train8=pd.concat([train8,test8],axis=0).reset_index(drop=True)
train9=pd.concat([train9,test9],axis=0).reset_index(drop=True)
train10=pd.concat([train10,test10],axis=0).reset_index(drop=True)
train11=pd.concat([train11,test11],axis=0).reset_index(drop=True)
train12=pd.concat([train12,test12],axis=0).reset_index(drop=True)
train13=pd.concat([train13,test13],axis=0).reset_index(drop=True)
train14=pd.concat([train14,test14],axis=0).reset_index(drop=True)
train15=pd.concat([train15,test15],axis=0).reset_index(drop=True)
train16=pd.concat([train16,test16],axis=0).reset_index(drop=True)
train17=pd.concat([train17,test17],axis=0).reset_index(drop=True)
train18=pd.concat([train18,test18],axis=0).reset_index(drop=True)
train19=pd.concat([train19,test19],axis=0).reset_index(drop=True)
train20=pd.concat([train20,test20],axis=0).reset_index(drop=True)
train21=pd.concat([train21,test21],axis=0).reset_index(drop=True)
train22=pd.concat([train22,test22],axis=0).reset_index(drop=True)
train23=pd.concat([train23,test23],axis=0).reset_index(drop=True)
train24=pd.concat([train24,test24],axis=0).reset_index(drop=True)
train25=pd.concat([train25,test25],axis=0).reset_index(drop=True)
train26=pd.concat([train26,test26],axis=0).reset_index(drop=True)
train27=pd.concat([train27,test27],axis=0).reset_index(drop=True)
train28=pd.concat([train28,test28],axis=0).reset_index(drop=True)
train29=pd.concat([train29,test29],axis=0).reset_index(drop=True)
train30=pd.concat([train30,test30],axis=0).reset_index(drop=True)
train31=pd.concat([train31,test31],axis=0).reset_index(drop=True)
train32=pd.concat([train32,test32],axis=0).reset_index(drop=True)
train33=pd.concat([train33,test33],axis=0).reset_index(drop=True)
train34=pd.concat([train34,test34],axis=0).reset_index(drop=True)
train35=pd.concat([train35,test35],axis=0).reset_index(drop=True)
train36=pd.concat([train36,test36],axis=0).reset_index(drop=True)
train37=pd.concat([train37,test37],axis=0).reset_index(drop=True)
print('结合数据完毕！准备特征提取...')

def one_hot_col(col):
    '''标签编码'''
    lbl = preprocessing.LabelEncoder()
    lbl.fit(col)
    return lbl
def feat1(data):
    '''对train1提特征'''
    data_id=pd.DataFrame(list(set(data['企业编号'])),columns=['企业编号'])
    zz1=data.groupby(['企业编号'],as_index=False)['产品类型'].agg({'feat1_count':'count',
                   'feat1_nunique':'nunique'})
    zz=data.groupby(['企业编号','产品类型'],as_index=False)['产品类型'].agg({'feat1_count':'count'})
    m=set(train1['产品类型'])
    for i in m:
        mm=zz[zz['产品类型']==i]
        mm.columns=['企业编号','产品类型','feat1_count_'+i]
        data_id=pd.merge(data_id,mm[['企业编号','feat1_count_'+i]],how='left',on='企业编号')
    data_id=pd.merge(data_id,zz1,how='left',on='企业编号')
    del zz1,zz
    return data_id
def feat2(data):
    '''对train2提特征'''
    for i,row in data.iterrows():
        if row['注册资本币种(正则)']=='美元':
            data.loc[i,'注册资本币种(正则)']='人民币'
            data.loc[i,'注册资本（万元）']=data.loc[i,'注册资本（万元）']*6.7
    mm=data[(~data['成立日期'].isnull())&(~data['经营期限自'].isnull())]
    mm['成立日期']=mm['成立日期'].apply(lambda x:int(x[:4]))
    mm['经营期限自']=mm['经营期限自'].apply(lambda x:int(x[:4]))
    mm['shijian']=mm['经营期限自']-mm['成立日期']
    data=data.drop(['注销时间','成立日期'],axis=1)
    for col in ['注册资本币种(正则)','经营状态','行业大类（代码）','行业小类（代码）','类型','省份代码',\
                '城市代码','地区代码','是否上市','登记机关区域代码']:
        data[col]=one_hot_col(data[col].astype(str)).transform(data[col].astype(str))
    data=pd.concat([pd.get_dummies(data['行业大类（代码）'],prefix='行业大类（代码）'),\
              data.drop(['行业大类（代码）'],axis=1)
              ],axis=1)
    #data=pd.concat([pd.get_dummies(data['行业小类（代码）'],prefix='行业小类（代码）'),\
    #          data.drop(['行业小类（代码）'],axis=1)
    #          ],axis=1)
    data=data.fillna(0)
    m=[]
    for i,row in data.iterrows():
        if row['注销原因']==0:
            m.append(0)
        else:
            m.append(1)
        #if row['发照日期']!=0 and row['经营期限自']!=0:
        #    data['shijian2']=row['发照日期'].apply(lambda x:int(x[:4]))-row['经营期限自'].apply(lambda x:int(x[:4]))
    data['注销原因']=m
    data=data.drop(['发照日期','经营期限自','经营期限至'],axis=1)
    data=pd.merge(data,mm[['企业编号','shijian']],how='left',on=['企业编号'])
    return data
def feat4(data):
    '''对train4提特征'''
    prefix='feat3'
    agg_func = {
        '供地总面积': ['median','mean', 'max', 'min', 'std','skew'],
        '成交价款（万元）': ['median','mean', 'max', 'min', 'std','skew'],
        #'行政区': ['nunique'],
        '土地用途': ['nunique'],
    }
    agg_trans = data.groupby(['企业编号']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (data.groupby('企业编号')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_trans = pd.merge(agg_trans,df,on='企业编号', how='left')
    return agg_trans
def feat9(data):
    '''对train9提特征'''
    data_id=pd.DataFrame(list(set(data['企业编号'])),columns=['企业编号'])
    m=data.groupby(['企业编号'],as_index=False)['企业编号'].agg({'train9_count':'count'})
    
    z=data[['企业编号','信用等级']].drop_duplicates()
    z=z.fillna(0)
    z=z[z['信用等级']!=0]
    def str_to_int(x):
        if x=='高级认证企业':
            return 4
        elif x=='一般认证企业':
            return 3
        elif x=='一般信用企业':
            return 2
        else:
            return 1
    z['信用等级']=z['信用等级'].apply(lambda x:str_to_int(x))
    z=z.groupby(['企业编号'],as_index=False)['信用等级'].agg({'信用等级':'mean'})
    data_id=pd.merge(data_id,z,how='left',on='企业编号')
    data_id=pd.merge(data_id,m,how='left',on='企业编号')
    return data_id
def feat10(data):
    '''对train10提特征'''
    data_id=pd.DataFrame(list(set(data['企业编号'])),columns=['企业编号'])
    m1=data.groupby(['企业编号'],as_index=False)['企业编号'].agg({'train10_count':'count'})
    m2=data.groupby(['企业编号'],as_index=False)['竞品的标签'].agg({'竞品的标签_count':'count',
                                                                   '竞品的标签_nunique':'nunique'})
    m3=data.groupby(['企业编号'],as_index=False)['竞品轮次'].agg({'竞品轮次_count':'count',
                                                                  '竞品轮次_nunique':'nunique'})
    m4=data.groupby(['企业编号'],as_index=False)['竞品详细地址'].agg({'竞品详细地址_count':'count',
                                                                  '竞品详细地址_nunique':'nunique'})
    m5=data.groupby(['企业编号'],as_index=False)['竞品运营状态'].agg({'竞品运营状态_count':'count',
                                                                  '竞品运营状态_nunique':'nunique'})
    data_id=pd.merge(data_id,m1,how='left',on='企业编号')
    data_id=pd.merge(data_id,m2,how='left',on='企业编号')
    data_id=pd.merge(data_id,m3,how='left',on='企业编号')
    data_id=pd.merge(data_id,m4,how='left',on='企业编号')
    data_id=pd.merge(data_id,m5,how='left',on='企业编号')
    return data_id
def feat14(data):
    '''对train14提特征'''
    data=data.drop_duplicates()
    mm=data.groupby(['企业编号'],as_index=False)['企业编号'].agg({'feat14_count':'count'})
    
    m=data[~data['认缴出资信息'].isnull()].reset_index(drop=True)
    m1=data[~data['实缴出资信息'].isnull()].reset_index(drop=True)
    def find_int(x):
        z=re.findall('（万元）：\d+.?\d*万美元|（万元）：\d+.?\d*人民币万|（万元）：\d+.?\d*万人民币|（万元）：\d+.?\d*|（万元）：\d+\d*',x)
        if len(z)>0:
            if '万美元' in z[0][5:]:
                return float(z[0][5:][:-3])*6.7
            elif '人民币万' in z[0][5:]:
                return float(z[0][5:][:-4])
            elif '万人民币' in z[0][5:]:
                return float(z[0][5:][:-4])
            elif '万' in z[0][5:]:
                return float(z[0][5:][:-1])
            else:
                #print(x)
                return float(z[0][5:])
        else:
            return -1
    
    m['认缴出资信息']=m['认缴出资信息'].apply(lambda x:find_int(x))
    m1['实缴出资信息']=m1['实缴出资信息'].apply(lambda x:find_int(x))
    m=m[['企业编号','认缴出资信息']].drop_duplicates().reset_index(drop=True)
    m1=m1[['企业编号','实缴出资信息']].drop_duplicates().reset_index(drop=True)
    
    return mm,m,m1

f1=feat1(train1)
f2=feat2(train2)

f3=train3.groupby(['企业编号'],as_index=False)['企业编号'].agg({'feat3_count':'count'})
f3_1=train3.groupby(['企业编号'],as_index=False)['行政区'].agg({'行政区_nunique':'nunique'})

f4=feat4(train4)

f6=train6.groupby(['企业编号'],as_index=False)['企业编号'].agg({'feat6_count':'count'})
f6_1=train6.groupby(['企业编号'],as_index=False)['总面积'].agg({'总面积_mean':'mean'})

f9=feat9(train9)
zz=train9[['企业编号','经济区划']].drop_duplicates()
f9_1=pd.DataFrame(list(set(zz['企业编号'])),columns=['企业编号'])
cols=list(set(zz[~zz['经济区划'].isnull()]['经济区划'])) ##没有空值
for col in cols:
    m=zz[zz['经济区划']==col].groupby(['企业编号'],as_index=False)['经济区划'].agg({str(col)+'_count':'count'})
    f9_1=pd.merge(f9_1,m,how='left',on=['企业编号']).fillna(0)

f10=feat10(train10)
f11=train11.groupby(['企业编号'],as_index=False)['纳税A级年份'].agg({'纳税A级年份_count':'count'})
f12=train12.groupby(['企业编号'],as_index=False)['企业编号'].agg({'feat12_count':'count'})

f13=train13.groupby(['企业编号'],as_index=False)['企业编号'].agg({'feat13_count':'count'})
f13_1=pd.DataFrame(list(set(train13['企业编号'])),columns=['企业编号'])
cols=list(set(train13[~train13['年报年份'].isnull()]['年报年份'])) ##没有空值
for col in cols:
    m=train13[train13['年报年份']==col].groupby(['企业编号'],as_index=False)['年报年份'].agg({str(int(col))+'_count':'count'})
    f13_1=pd.merge(f13_1,m,how='left',on=['企业编号']).fillna(0)
f13_1.drop(['2018_count','2012_count'],axis=1,inplace=True)
f13_1['mean']=np.log(f13_1[['2013_count','2014_count','2015_count','2016_count','2017_count']].mean(axis=1))
f13_1['max']=np.log(f13_1[['2013_count','2014_count','2015_count','2016_count','2017_count']].max(axis=1))
#f13_1['min']=f13_1[['2013_count','2014_count','2015_count','2016_count','2017_count']].min(axis=1)
f13_1['std']=np.log(f13_1[['2013_count','2014_count','2015_count','2016_count','2017_count']].std(axis=1))
f13_1.drop(['2013_count','2014_count','2015_count','2016_count','2017_count'],axis=1,inplace=True)

f14,m,m1=feat14(train14)
f14_1=m[m['认缴出资信息']!=-1].groupby(['企业编号'],as_index=False)['认缴出资信息'].agg({#'认缴出资信息_nunique':'nunique',
                                                       #  '认缴出资信息_max':'max',
                                                       #  '认缴出资信息_min':'min',
                                                         '认缴出资信息_mean':'mean',
                                                         })
f14_2=m1[m1['实缴出资信息']!=-1].groupby(['企业编号'],as_index=False)['实缴出资信息'].agg({#'实缴出资信息_nunique':'nunique',
                                                       #  '实缴出资信息_max':'max',
                                                       #  '实缴出资信息_min':'min',
                                                         '实缴出资信息_mean':'mean',
                                                         })
f14_1['认缴出资信息_mean']=np.log(f14_1['认缴出资信息_mean']+1)
f14_2['实缴出资信息_mean']=np.log(f14_2['实缴出资信息_mean']+1)

f15=train15.drop_duplicates().groupby(['企业编号'],as_index=False)['企业编号'].agg({'feat15_count':'count'})

f16=train16.drop_duplicates().groupby(['企业编号'],as_index=False)['企业编号'].agg({'feat16_count':'count'})
m=train16.drop_duplicates()
m=m[(~m['是否有网站或网点'].isnull())&(~m['企业是否有投资信息或购买其他公司股权'].isnull())& \
    (~m['有限责任公司本年度是否发生股东股权转'].isnull())&(~m['是否提供对外担保'].isnull())]
cols=['是否有网站或网点', '企业是否有投资信息或购买其他公司股权',\
       '有限责任公司本年度是否发生股东股权转', '是否提供对外担保']
for col in cols:
    m[col]=m[col].apply(lambda x:1 if x in ['是','有'] else 0)
m['从业人数']=m['从业人数'].apply(lambda x:0 if x[0]=='企' else 1)
m=m[['企业编号','从业人数']+cols]
m=m.drop_duplicates().reset_index(drop=True)
cols=[col for col in m.columns if col!='企业编号']
f16_1=m.groupby(['企业编号'],as_index=False)[cols].sum()

f19=train19.drop_duplicates().groupby(['企业编号'],as_index=False)['企业编号'].agg({'feat19_count':'count'})
import time
def get_date(x):
    y,m,d=x.split('-')
    return int(y),int(m),int(y)+(int(m)*30+int(d))/365

f22=train22.drop_duplicates().groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat22_count':'count'})
f22.columns=['企业编号','feat22_count']
#f22_1=train22.drop_duplicates().groupby(['企业总评分'],as_index=False)['商标状态'].agg({'商标状态_nunique':'nunique'})
#f22_1.columns=['企业编号','商标状态_nunique']
f22_1=pd.DataFrame(list(set(train22['企业总评分'])),columns=['企业编号'])
cols=list(set(train22[~train22['商标状态'].isnull()]['商标状态']))
train22=train22.drop_duplicates()
for col in cols:
    m=train22[train22['商标状态']==col].groupby(['企业总评分'],as_index=False)['商标状态'].agg({col+'_count':'count'})
    m.columns=['企业编号',col+'_count'] 
    f22_1=pd.merge(f22_1,m,how='left',on=['企业编号']).fillna(0)
##下面求train22中申请日期的特征：
m=train22[~train22['申请日期'].isnull()]
m['申请年']=m['申请日期'].apply(lambda x:get_date(x)[0])
m['申请日']=m['申请日期'].apply(lambda x:get_date(x)[1])
m['申请日期']=m['申请日期'].apply(lambda x:get_date(x)[2])
f22_3=m.groupby(['企业总评分'],as_index=False)['申请年'].agg({'申请年_nunique':'nunique',
                                                         '申请年_max':'max',
                                                         '申请年_min':'min',})
f22_3.columns=['企业编号']+list(f22_3.columns[1:])
f22_3['申请年_max-申请年_min']=f22_3['申请年_max']-f22_3['申请年_min']

f22_4=m.groupby(['企业总评分'],as_index=False)['申请日'].agg({'申请日_nunique':'nunique',
                                                         '申请日_max':'max',
                                                         '申请日_min':'min',
                                                         '申请日_mean':'mean',
                                                         '申请日_median':'median',})
f22_4.columns=['企业编号']+list(f22_4.columns[1:])
f22_2=pd.DataFrame(list(set(m['企业总评分'])),columns=['企业编号'])
f=[]
for i,row in tqdm(f22_2.iterrows()):
    mm=m[m['企业总评分']==row['企业编号']].sort_values(by=['申请日期']).reset_index(drop=True)
    mmm=mm['申请日期'].diff()
    f.append([np.mean(mmm),np.max(mmm),np.std(mmm),mmm.skew(),mmm.kurt()])
f=pd.DataFrame(f,columns=['申请日期mean','申请日期max','申请日期std','申请日期skew','申请日期kurt'])
f22_2=pd.concat([f22_2,f],axis=1)
print('提取train22申请日期特征完成')

f23=train23.drop_duplicates().groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat23_count':'count'})
f23.columns=['企业编号','feat23_count']
##求每股的收益：
train23=train23.fillna(-999)
m=[]
for i,row in train23.iterrows():
    if row['基本每股收益(元)'] not in ['--',-999]:
        m.append(float(row['基本每股收益(元)']))
    elif row['扣非每股收益(元)'] not in ['--',-999]:
        m.append(float(row['扣非每股收益(元)']))
    elif row['稀释每股收益(元)'] not in ['--',-999]:
        m.append(float(row['稀释每股收益(元)']))
    else:
        m.append(-999)
train23['每股收益']=m
f23_1=train23[train23['每股收益']!=-999].groupby(['企业总评分'],as_index=False)['每股收益'].agg({'每股收益_mean':'mean',
                   '每股收益_std':'std',
                 #  '每股收益max':'max',
                 #  '每股收益_min':'min',
                   })
f23_1.columns=['企业编号']+list(f23_1.columns)[1:]
col='每股净资产(元)'
mm=train23[(train23[col]!='--')&(train23[col]!=-999)]
mm[col]=mm[col].astype(float)
f23_2=mm.groupby(['企业总评分'],as_index=False)[col].agg({col+'_mean':'mean',
                   col+'_std':'std',
                  # col+'_max':'max',
                  # col+'_min':'min',
                   })
f23_2.columns=['企业编号']+list(f23_2.columns)[1:] 
col='每股公积金(元)'
mm=train23[(train23[col]!='--')&(train23[col]!=-999)]
mm[col]=mm[col].astype(float)
f23_3=mm.groupby(['企业总评分'],as_index=False)[col].agg({col+'_mean':'mean',
                   col+'_std':'std',
                  # col+'_max':'max',
                  # col+'_min':'min',
                   })
f23_3.columns=['企业编号']+list(f23_3.columns)[1:] 
col='每股未分配利润(元)'
mm=train23[(train23[col]!='--')&(train23[col]!=-999)]
mm[col]=mm[col].astype(float)
f23_4=mm.groupby(['企业总评分'],as_index=False)[col].agg({col+'_mean':'mean',
                   col+'_std':'std',
                  # col+'_max':'max',
                  # col+'_min':'min',
                   })
f23_4.columns=['企业编号']+list(f23_4.columns)[1:] 
col='每股经营现金流(元)'
mm=train23[(train23[col]!='--')&(train23[col]!=-999)]
mm[col]=mm[col].astype(float)
f23_5=mm.groupby(['企业总评分'],as_index=False)[col].agg({col+'_mean':'mean',
                   col+'_std':'std',
                 #  col+'_max':'max',
                 #  col+'_min':'min',
                   })
f23_5.columns=['企业编号']+list(f23_5.columns)[1:] 

train24['资产负债率(%)']=train24['资产负债率(%)'].apply(lambda x:x[:-1])
train24['流动负债/总负债(%)']=train24['流动负债/总负债(%)'].apply(lambda x:x[:-1])
col='资产负债率(%)'
mm=train24[(train24[col]!='--')&(train24[col]!=-999)]
mm[col]=mm[col].astype(float)
f24=mm.groupby(['企业总评分'],as_index=False)[col].agg({col+'_mean':'mean',
                   col+'_std':'std',
                 #  col+'_max':'max',
                 #  col+'_min':'min',
                   })
f24.columns=['企业编号']+list(f24.columns)[1:]
col='流动负债/总负债(%)'
mm=train24[(train24[col]!='--')&(train24[col]!=-999)]
mm[col]=mm[col].astype(float)
f24_1=mm.groupby(['企业总评分'],as_index=False)[col].agg({col+'_mean':'mean',
                   col+'_std':'std',
                 #  col+'_max':'max',
                 #  col+'_min':'min',
                   })
f24_1.columns=['企业编号']+list(f24_1.columns)[1:]
col='流动比率'
mm=train24[(train24[col]!='--')&(train24[col]!=-999)]
mm[col]=mm[col].astype(float)
f24_2=mm.groupby(['企业总评分'],as_index=False)[col].agg({col+'_mean':'mean',
                   col+'_std':'std',
                 #  col+'_max':'max',
                 #  col+'_min':'min',
                   })
f24_2.columns=['企业编号']+list(f24_2.columns)[1:]
col='速动比率'
mm=train24[(train24[col]!='--')&(train24[col]!=-999)]
mm[col]=mm[col].astype(float)
f24_3=mm.groupby(['企业总评分'],as_index=False)[col].agg({col+'_mean':'mean',
                   col+'_std':'std',
                 #  col+'_max':'max',
                 #  col+'_min':'min',
                   })
f24_3.columns=['企业编号']+list(f24_3.columns)[1:]

def train25_deal(x):
    if x[-2:]=='万亿':
        return float(x[:-2])*10000
    elif x[-1]=='万':
        return float(x[:-1])*0.0001
    elif x[-1]=='亿':
        return float(x[:-1])
    else:
        return -99  
train25['营业总收入(元)']=train25['营业总收入(元)'].apply(lambda x:train25_deal(x))
train25.columns=['企业编号']+list(train25.columns[1:])
#train25=pd.merge(train25,label,how='left',on=['企业编号'])
f25=train25[train25['营业总收入(元)']!=-99].groupby(['企业编号'],as_index=False)['营业总收入(元)'].agg({'营业总收入(元)_mean':'mean'})

def train26_deal(x):
    if x==-99:
        return x
    elif str(x[-1])=='亿':
        #print(x)
        return float(x[:-1])*10**8
    elif str(x[-1])=='万':
        return float(x[:-1])*10**4
    else:
        return float(x)
f26=train26.drop_duplicates().groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat26_count':'count'})
f26.columns=['企业编号','feat26_count']
m=train26.drop_duplicates().fillna(-99).copy()
m['营业收入(元)1']=m['营业收入(元)'].apply(lambda x:train26_deal(x))

f28=train28.drop_duplicates().groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat28_count':'count'})
f28.columns=['企业编号','feat28_count']

f29=train29.groupby(['企业总评分'],as_index=False)['总资产周转率(次)'].agg({'总资产周转率(次)_mean':'mean'})
f29.columns=['企业编号','总资产周转率(次)_mean']
m=train29[train29['应收账款周转天数(天)']!='--']
m['应收账款周转天数(天)']=m['应收账款周转天数(天)'].astype(float)
f29_1=m.groupby(['企业总评分'],as_index=False)['应收账款周转天数(天)'].agg({'应收账款周转天数(天)_mean':'mean'})
f29_1.columns=['企业编号','应收账款周转天数(天)_mean']
m=train29[train29['存货周转天数(天)']!='--']
m['存货周转天数(天)']=m['存货周转天数(天)'].astype(float)
f29_2=m.groupby(['企业总评分'],as_index=False)['存货周转天数(天)'].agg({'存货周转天数(天)_mean':'mean'})
f29_2.columns=['企业编号','存货周转天数(天)_mean']


f30=train30.drop_duplicates().groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat30_count':'count'})
f30.columns=['企业编号','feat30_count']
train30=train30.drop_duplicates()
m=train30[(~train30['资产:货币资金(元)'].isnull())&(train30['资产:货币资金(元)']!='--')][['企业总评分','资产:货币资金(元)']]
def str_to_num(x):
    if x[-1]=='万':
        return float(x[:-1])
    elif x[-1]=='亿':
        return float(x[:-1])
    else:
        return float(x)
m['资产:货币资金(元)']=m['资产:货币资金(元)'].apply(lambda x:str_to_num(x))
f30_1=m.groupby(['企业总评分'],as_index=False)['资产:货币资金(元)'].agg({'资产:货币资金(元)_mean':'mean'})
f30_1.columns=['企业编号','资产:货币资金(元)_mean']
m=train30[(~train30['资产:固定资产(元)'].isnull())&(train30['资产:固定资产(元)']!='--')][['企业总评分','资产:固定资产(元)']]
m['资产:固定资产(元)']=m['资产:固定资产(元)'].apply(lambda x:str_to_num(x))
f30_2=m.groupby(['企业总评分'],as_index=False)['资产:固定资产(元)'].agg({'资产:固定资产(元)_mean':'mean'})
f30_2.columns=['企业编号','资产:固定资产(元)_mean']
m=train30[(~train30['资产:无形资产(元)'].isnull())&(train30['资产:无形资产(元)']!='--')][['企业总评分','资产:无形资产(元)']]
m['资产:无形资产(元)']=m['资产:无形资产(元)'].apply(lambda x:str_to_num(x))
f30_3=m.groupby(['企业总评分'],as_index=False)['资产:无形资产(元)'].agg({'资产:无形资产(元)_mean':'mean'})
f30_3.columns=['企业编号','资产:无形资产(元)_mean']
m=train30[(~train30['资产:资产总计(元)'].isnull())&(train30['资产:资产总计(元)']!='--')][['企业总评分','资产:资产总计(元)']]
m['资产:资产总计(元)']=m['资产:资产总计(元)'].apply(lambda x:str_to_num(x))
f30_4=m.groupby(['企业总评分'],as_index=False)['资产:资产总计(元)'].agg({'资产:资产总计(元)_mean':'mean'})
f30_4.columns=['企业编号','资产:资产总计(元)_mean']
m=train30[(~train30['负债:负债合计(元)'].isnull())&(train30['负债:负债合计(元)']!='--')][['企业总评分','负债:负债合计(元)']]
m['负债:负债合计(元)']=m['负债:负债合计(元)'].apply(lambda x:str_to_num(x))
f30_5=m.groupby(['企业总评分'],as_index=False)['负债:负债合计(元)'].agg({'负债:负债合计(元)_mean':'mean'})
f30_5.columns=['企业编号','负债:负债合计(元)_mean']

f31=train31.drop_duplicates().groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat31_count':'count'})
f31.columns=['企业编号','feat31_count']
f31_1=train31.drop_duplicates().groupby(['企业总评分'],as_index=False)['标签'].agg({'标签_nunique':'nunique'})
f31_1.columns=['企业编号','标签_nunique']

f32=train32.drop_duplicates().groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat32_count':'count'})
f32.columns=['企业编号','feat32_count']

f33=train33.groupby(['企业总评分'],as_index=False)['计划发行总额（亿元）'].agg({'计划发行总额（亿元）_mean':'mean'})
f33.columns=['企业编号','计划发行总额（亿元）_mean']
f33_1=train33.groupby(['企业总评分'],as_index=False)['票面利率（%）'].agg({'票面利率（%）_mean':'mean'})
f33_1.columns=['企业编号','票面利率（%）_mean']
f33_2=train33.groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat33_count':'count'})
f33_2.columns=['企业编号','feat33_count']

f34=train34.groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat34_count':'count'})
f34.columns=['企业编号','feat34_count'] ##有用特征
f34_1=train34.groupby(['企业总评分'],as_index=False)['省份'].agg({'省份_nunique':'nunique'})
f34_1.columns=['企业编号','省份_nunique'] ##有用特征

f35=train35.groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat35_count':'count'})
f35.columns=['企业编号','feat35_count'] ##有用特征,强特
f35_1=pd.DataFrame(list(set(train35['企业总评分'])),columns=['企业编号'])
cols=list(set(train35[~train35['专利类型'].isnull()]['专利类型']))
for col in cols:
    m=train35[train35['专利类型']==col].groupby(['企业总评分'],as_index=False)['专利类型'].agg({col+'_count':'count'})
    m.columns=['企业编号',col+'_count'] 
    f35_1=pd.merge(f35_1,m,how='left',on=['企业编号']).fillna(0)
##删除重复项：
train35['授权公告日']=train35['授权公告日'].apply(lambda x:x[:-1] if x[-1]=='\xad' else x)
train35['申请日']=train35['申请日'].apply(lambda x:x[4:] if x[:4]=='公告日：' else x)
train35=train35.drop_duplicates().reset_index(drop=True)
f35_2=train35.groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat35_count':'count'})
f35_2.columns=['企业编号','feat35_count'] ##有用特征,强特
f35_3=pd.DataFrame(list(set(train35['企业总评分'])),columns=['企业编号'])
cols=list(set(train35[~train35['专利类型'].isnull()]['专利类型']))
for col in cols:
    m=train35[train35['专利类型']==col].groupby(['企业总评分'],as_index=False)['专利类型'].agg({col+'_count':'count'})
    m.columns=['企业编号',col+'_count'] 
    f35_3=pd.merge(f35_3,m,how='left',on=['企业编号']).fillna(0)


f36=train36.groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat36_count':'count'})
f36.columns=['企业编号','feat36_count'] ##有用特征,强特
f36_1=train36.groupby(['企业总评分'],as_index=False)['证书名称'].agg({'证书名称_nunique':'nunique'})
f36_1.columns=['企业编号','证书名称_nunique'] ##有用特征,强特 3.145->3.1217
f36_2=pd.DataFrame(list(set(train36['企业总评分'])),columns=['企业编号'])
cols=list(set(train36[~train36['状态'].isnull()]['状态']))
for col in cols:
    m=train36[train36['状态']==col].groupby(['企业总评分'],as_index=False)['状态'].agg({col+'_count':'count'})
    m.columns=['企业编号',col+'_count']
    f36_2=pd.merge(f36_2,m,how='left',on=['企业编号']).fillna(0)

#s=f36_2.drop(['企业编号'],axis=1).sum(axis=1)
#for col in [i for i in f36_2.columns if i!='企业编号']:
#    f36_2[col]=f36_2[col]/s
#f36_2['sum']=s
#f36_1=train36.groupby(['企业总评分','状态'],as_index=False)['状态'].agg({'状态_count':'count'})

f37=train37.groupby(['企业总评分'],as_index=False)['企业总评分'].agg({'feat37_count':'count'})
f37.columns=['企业编号','feat37_count']
print('所有特征提取结束！')

print('特征结合...')
new_train=pd.merge(all_label,f1,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f2,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f3,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f3_1,how='left',on=['企业编号']) #+
new_train=pd.merge(new_train,f4,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f6,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f6_1,how='left',on=['企业编号']) #+
new_train=pd.merge(new_train,f9,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f9_1,how='left',on=['企业编号'])   #+

new_train=pd.merge(new_train,f10,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f11,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f12,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f13,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f13_1,how='left',on=['企业编号']) #+

new_train=pd.merge(new_train,f14,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f14_1,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f14_2,how='left',on=['企业编号']) #+

new_train=pd.merge(new_train,f15,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f16,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f16_1,how='left',on=['企业编号'])

new_train=pd.merge(new_train,f19,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f22,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f22_1,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f22_2,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f22_3,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f22_4,how='left',on=['企业编号']) #+
new_train=pd.merge(new_train,f22_3.drop(['申请年_max','申请年_min'],axis=1),how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f22_4[['企业编号','申请日_mean']],how='left',on=['企业编号']) #+

new_train=pd.merge(new_train,f23,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f23_1,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f23_2,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f23_3,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f23_4,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f23_5,how='left',on=['企业编号']) #+

#new_train=pd.merge(new_train,f24,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f24_1,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f24_2,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f24_3,how='left',on=['企业编号']) #+

#new_train=pd.merge(new_train,f25,how='left',on=['企业编号'])  #+

new_train=pd.merge(new_train,f26,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f28,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f29,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f29_1,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f29_2,how='left',on=['企业编号']) #+

new_train=pd.merge(new_train,f30,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f30_1,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f30_2,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f30_3,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f30_4,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f30_5,how='left',on=['企业编号']) #+

new_train=pd.merge(new_train,f31,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f31_1,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f32,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f33,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f33_1,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f33_2,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f34,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f34_1,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f35,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f35_1,how='left',on=['企业编号'])
#new_train=pd.merge(new_train,f35_2,how='left',on=['企业编号']) #+
#new_train=pd.merge(new_train,f35_3,how='left',on=['企业编号']) #+

new_train=pd.merge(new_train,f36,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f36_1,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f36_2,how='left',on=['企业编号'])
new_train=pd.merge(new_train,f37,how='left',on=['企业编号'])
print('特征结合完毕！')

#准备划分数据：
train_data=new_train[:len(label)]  #最终训练集
test_data=new_train[len(label):].reset_index(drop=True) #最终测试集
zz=new_train.describe()

#========================平滑处理===============================================
for i in zz.columns:
    
    if zz.ix['max',i]>10000 and i!='企业编号':
        #print(i)
        train_data[i]=np.log(train_data[i])
        test_data[i]=np.log(test_data[i])
    
    train_data.loc[(train_data[i]==-np.inf)|(train_data[i]==np.inf),i]=-99
    test_data.loc[(test_data[i]==-np.inf)|(test_data[i]==np.inf),i]=-99

#================================特征选择========================================
print('特征选择中...')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
sm = SelectFromModel(GradientBoostingRegressor(random_state=42), threshold=0.001)
cols=[i for i in new_train.columns if i not in ['企业编号','企业总评分']]
train_data= sm.fit_transform(train_data.drop(['企业编号','企业总评分'], axis=1).fillna(-99).astype(np.float), train_data['企业总评分'])
test_data= sm.transform(test_data.drop(['企业编号','企业总评分'], axis=1).fillna(-99).astype(np.float))
use_cols=[cols[i] for i in sm.get_support([0])]

train_data=pd.DataFrame(train_data,columns=use_cols)
test_data=pd.DataFrame(test_data,columns=use_cols)
print('特征选择后的特征数：',train_data.shape)
train_data=pd.concat([label,train_data],axis=1)
test_data=pd.concat([final_submit[['企业编号']],test_data],axis=1) ##这里的new_train应该是new_test
print('特征选择完毕！')


#train_data=new_train.copy()
#test_data=new_train.copy()
def display_importances(feature_importance_df_):
    '''特征重要性可视化'''
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(16, 10))
    print(best_features.columns)
    print(best_features['feature'])
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    #plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    
#================================lgb模型5折交叉验证训练预测================================
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 31,
    'verbose': -1,
    'max_depth': -1,
    'reg_alpha':2.2,
    'reg_lambda':1.4,
    'nthread': 8
}
print('模型训练...')
from sklearn.model_selection import KFold
cv_pred_all = 0
en_amount = 3

oof_lgb1=np.zeros(len(train_data))
prediction_lgb1=np.zeros(len(test_data))
all_tr_lgb=0
for seed in range(en_amount):
    NFOLDS = 5
    train_label = train_data['企业总评分']
    
    kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
    kf = kfold.split(train_data, train_label)

    train_data_use = train_data.drop(['企业编号','企业总评分'], axis=1)
    test_data_use = test_data[train_data_use.columns]


    cv_pred = np.zeros(test_data.shape[0])
    valid_best_l2_all = 0
    
    feature_importance_df = pd.DataFrame()
    count = 0
    oof_lgb11=np.zeros(len(train_data)) ##用于计算每个seed对应的train，方便计算每个seed的rmse
    for i, (train_fold, validate) in enumerate(kf):
        print('fold: ',i, ' training')
        X_train, X_validate, label_train, label_validate = \
        train_data_use.iloc[train_fold, :], train_data_use.iloc[validate, :], \
        train_label[train_fold], train_label[validate]
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=30000, valid_sets=dvalid, verbose_eval=500,early_stopping_rounds=250)
        cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
        valid_best_l2_all += bst.best_score['valid_0']['l1']
        
        oof_lgb1[validate]=bst.predict(X_validate,num_iteration=bst.best_iteration)
        oof_lgb11[validate]=bst.predict(X_validate,num_iteration=bst.best_iteration)
        prediction_lgb1+=bst.predict(test_data_use,num_iteration=bst.best_iteration)/kfold.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(X_train.columns)
        fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        fold_importance_df["fold"] = count + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        count += 1
    #计算每个seed的误差：
    print("lgb每个seed的score: {:<8.8f}".format(np.sqrt(mean_squared_error(oof_lgb11, train_label))))
    all_tr_lgb+=oof_lgb11
    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    
    cv_pred_all += cv_pred
#计算所有seed的平均误差
#print("lgb所有seed的平均score: {:<8.8f}".format(np.sqrt(mean_squared_error(oof_lgb1/en_amount, train_label))))
cv_pred_all /= en_amount
all_tr_lgb/=en_amount
print("lgb所有seed的平均score: {:<8.8f}".format(np.sqrt(mean_squared_error(all_tr_lgb, train_label))))
prediction_lgb1/=en_amount
print('cv score for valid is: ', 1/(1+valid_best_l2_all))

'''
===========================特征重要性可视化======================================
# 2. 将特征按分数 从大到小 排序
named_scores = zip(feature_importance_df["feature"],feature_importance_df["importance"])
sorted_named_scores = sorted(named_scores, key=lambda z: z[1], reverse=True)
m=pd.DataFrame()
m['sorted_scores']=[each[1] for each in sorted_named_scores]
m['sorted_names']=[each[0] for each in sorted_named_scores]
m=m.drop_duplicates('sorted_names').reset_index(drop=True)
##只画前十的特征：
sorted_scores = m['sorted_scores'][:10]
sorted_names = m['sorted_names'][:10]
print(m['sorted_names'][:10])
qian10=['Registered_Capital(Ten_Thousand_Yuan)',\
 'Current_Liabilities/Total_Liabilities(%)_std',\
 'Total_Asset_Turnover(Times)_mean',\
 'Earnings_Per_Share_std',\
 'Provident_Fund_Per_Share(Yuan)_mean',\
 'Current_Liabilities/Total_Liabilities(%)_mean',\
 'Days_of_Receivables_Turnover(Days)_mean',\
 'Inventory_Turnover_Days(Days)',\
 'Total_Revenue(Yuan)',\
 'Application_Date_std']
sorted_names=qian10
y_pos = np.arange(10) # 从上而下的绘图顺序
# 3. 绘图
#ax = plt.figure(figsize=(16, 10))
#plt.figure(figsize=(16, 10))
fig, ax = plt.subplots(figsize=(16, 10))
ax.barh(y_pos, sorted_scores, height=0.6,align='center', tick_label=sorted_names)
# ax.set_yticklabels(sorted_names)      # 也可以在这里设置 条条 的标签~
ax.set_yticks(y_pos)
ax.set_xlabel('Feature Importance',fontsize=20)
ax.set_ylabel('Feature Name',fontsize=20)
ax.invert_yaxis()
#ax.set_title('F_classif scores of the features.')
# 4. 添加每个 条条 的数字标签
for score, pos in zip(sorted_scores, y_pos):
    ax.text(score + 30, pos, '%.1f' % score, ha='center', va='bottom', fontsize=20)
plt.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('importance_10.tiff', dpi=300)
plt.show()
'''


#===================================xgb模型5折交叉验证训练=============================
xgb_params={'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 8}
from sklearn.model_selection import KFold
cv_pred_allxgb=0
en_amount=3
oof_xgb1=np.zeros(len(train_data))
prediction_xgb1=np.zeros(len(test_data))
all_tr_xgb=0
for seed in range(en_amount):
    NFOLDS=5
    train_label=train_data['企业总评分']
    kfold=KFold(n_splits=NFOLDS, shuffle=True, random_state=seed+2019)
    kf=kfold.split(train_data,train_label)
    
    train_data_use = train_data.drop(['企业编号','企业总评分'], axis=1)
    test_data_use = test_data[train_data_use.columns]
    
    cv_pred = np.zeros(train_data.shape[0])
    valid_best_l2_all = 0
    
    feature_importance_df = pd.DataFrame()
    count = 0
    oof_xgb11=np.zeros(len(train_data))
    for i, (train_fold, validate) in enumerate(kf):
        print('fold: ',i, ' training')
        X_train, X_validate, label_train, label_validate = train_data_use.iloc[train_fold, :], train_data_use.iloc[validate, :], train_label[train_fold], train_label[validate]
        dtrain = xgb.DMatrix(X_train, label_train)
        dvalid = xgb.DMatrix(X_validate, label_validate)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
        bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=500, verbose_eval=300, params=xgb_params)
        cv_pred += bst.predict(xgb.DMatrix(train_data_use), ntree_limit=bst.best_ntree_limit)
        oof_xgb1[validate]=bst.predict(xgb.DMatrix(X_validate),ntree_limit=bst.best_ntree_limit)
        oof_xgb11[validate]=bst.predict(xgb.DMatrix(X_validate),ntree_limit=bst.best_ntree_limit)
        prediction_xgb1+=bst.predict(xgb.DMatrix(test_data_use),ntree_limit=bst.best_ntree_limit)/kfold.n_splits
        count += 1
    print("xgb每个seed的score: {:<8.8f}".format(np.sqrt(mean_squared_error(oof_xgb11, train_label))))
    all_tr_xgb+=oof_xgb11
    cv_pred /= NFOLDS
    cv_pred_allxgb+=cv_pred
#print("xgb所有seed的平均score: {:<8.8f}".format(np.sqrt(mean_squared_error(oof_xgb1/en_amount, train_label))))
cv_pred_allxgb /= en_amount
all_tr_xgb/=en_amount
print("xgb所有seed的平均score: {:<8.8f}".format(np.sqrt(mean_squared_error(all_tr_xgb, train_label))))
prediction_xgb1/=en_amount

from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

#=======================将lgb和xgb的结果进行stacking===============================
train_stack = np.vstack([all_tr_lgb,all_tr_xgb]).transpose()
test_stack = np.vstack([prediction_lgb1, prediction_xgb1]).transpose()

folds_stack = KFold(n_splits=5, random_state=2019)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])
target=train_data['企业总评分']

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
    
    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 5
print("stacking的score: {:<8.8f}".format(np.sqrt(mean_squared_error(oof_stack, train_label))))

#======================================保存第二个结果===============================
final_submit['企业总评分']=predictions
#保存第二个模型的结果：
final_submit[['企业编号','企业总评分']].to_excel('output/model2_result.xlsx', index=False,encoding='gbk')
