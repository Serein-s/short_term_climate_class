import numpy as np
import cmaps
import pandas as pd
import xarray as xr
import matplotlib
import proplot as pplt
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import warnings
#忽略警告
warnings.filterwarnings("ignore")
#防止中文出错
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
f_6 = pd.read_csv(r"D:\sx08\r1606.txt", sep="\s+", header=None)
prec_6 = np.array(f_6).reshape(71, 160)
f_7 = pd.read_csv(r"D:\sx08\r1607.txt", sep="\s+", header=None)
prec_7 = np.array(f_7).reshape(71, 160)
f_8 = pd.read_csv(r"D:\sx08\r1608.txt", sep="\s+", header=None)
prec_8 = np.array(f_8).reshape(71, 160)
sta = pd.read_csv(r"D:\sx08\zd.txt",
                  sep="\s+",
                  header=None,
                  names=['station', 'lat', 'lon'])
f_index = pd.read_csv(r"D:\sx08\74index_195101-201612.txt",
                      sep="\s+",
                      header=None)
f_index = np.array(f_index).reshape(100, 12, 74)
nino = pd.read_csv(r"D:\sx08\nino3.4_195001-202112.txt",
                   sep="\s+",
                   header=None,
                   usecols=np.arange(1, 13))
nino = np.array(nino).reshape(72, 12)

choes_nino = nino[29:61, 5:8].mean(1)
s1 = pd.DataFrame(choes_nino)
'''
10.北美大西洋副高面积指数(110W-20W)
16.西太平洋副高强度指数(110E-180)
25.北非大西洋北美副高脊线(110W-60E)
33.太平洋副高脊线(110E-115W)
'''
index = np.array([10, 16, 25, 33]) - 1
chose_index = f_index[28:60, 5:8, :].mean(1)
s2 = pd.DataFrame((chose_index[:, index]))
X = np.array(pd.concat([s1, s2], axis=1))
prec = (prec_6 + prec_7 + prec_8)[28:60, :]
ave_prec = (prec[2:32, :]).mean(0)
Y = (prec - ave_prec) / ave_prec
k = np.zeros((160, 5))  #存放系数
for i in range(Y.shape[1]):
    k[i] = np.linalg.lstsq(X, Y[:, i], rcond=None)[0]
s11 = pd.DataFrame(nino[61:66, 5:8, ].mean(1))
chose_pre_index = f_index[60:65, 5:8, :].mean(1)
s22 = pd.DataFrame(chose_pre_index[:, index])
X_pre = np.array(pd.concat([s11, s22], axis=1))
#预测
prec_pre = np.zeros((5, 160))
for i in range(5):
    prec_pre[i] = (k * X_pre[i]).sum(1)
#观测
prec_obs = ((prec_6 + prec_7 + prec_8)[60:65, :] - ave_prec) / ave_prec
def darw_pre(sta, prec_pre, prec_obs, title):
    def c_map(ax):
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        ax.add_feature(cfeature.LAKES.with_scale('10m'))
        ax.add_geometries(Reader(r"D:\data\map\bou2_4l.shp").geometries(),
                          ccrs.PlateCarree(),
                          facecolor='none',
                          edgecolor='k',
                          linewidth=0.7)

    fig = pplt.figure(figsize=(12, 5), suptitle=title[0])
    ax = fig.subplot(121, proj='cyl', title=title[1])
    c1 = ax.scatter(sta["lon"],
                    sta["lat"],
                    vmin=-45,
                    vmax=45,
                    s=15,
                    c=prec_pre * 100,
                    cmap='jet',
                    transform=ccrs.PlateCarree())
    cbar = ax.colorbar(c1, loc='b', tickdir='in', shrink=0.75)
    cbar.set_ticks(np.arange(-45, 45.1, 10))
    c_map(ax)
    ax.format(
        coast=True,
        borders=True,
        latlines=10,
        lonlines=10,
        lonlabels=True,
        latlabels=True,
        lonlim=(70, 140),
        latlim=(15, 65),
        longrid=True,
        latgrid=True,
        abc='a.',
    )
    ax_1 = fig.add_axes([0.368, 0.14, 0.15, 0.25], proj='cyl')
    ax_1.scatter(sta["lon"],
                 sta["lat"],
                 vmin=-45,
                 vmax=45,
                 s=8,
                 c=prec_pre * 100,
                 cmap='jet')
    ax_1.format(
        lonlabels=False,
        latlabels=False,
        lonlim=(105, 122),
        latlim=(0, 24),
    )
    c_map(ax_1)

    ax2 = fig.subplot(122, proj='cyl', title=title[2])
    c2 = ax2.scatter(sta["lon"],
                     sta["lat"],
                     vmin=-45,
                     vmax=45,
                     s=15,
                     c=prec_obs * 100,
                     cmap='jet',
                     transform=ccrs.PlateCarree())
    c_map(ax2)
    cbar = ax2.colorbar(c2, loc='b', tickdir='in', shrink=0.75)
    cbar.set_ticks(np.arange(-45, 45.1, 10))
    ax2.format(
        coast=True,
        borders=True,
        latlines=10,
        lonlines=10,
        lonlabels=True,
        latlabels=True,
        lonlim=(70, 140),
        latlim=(15, 65),
        longrid=True,
        latgrid=True,
        abc='a.',
    )
    ax_2 = fig.add_axes([0.86, 0.14, 0.15, 0.25], proj='cyl')
    ax_2.scatter(sta["lon"],
                 sta["lat"],
                 vmin=-45,
                 vmax=45,
                 s=8,
                 c=prec_obs * 100,
                 cmap='jet',
                 transform=ccrs.PlateCarree())
    ax_2.format(
        lonlabels=False,
        latlabels=False,
        lonlim=(105, 122),
        latlim=(0, 24),
    )
    c_map(ax_2)
    
for i in range(5):
    title = ['%0.f年夏季中国降水距平百分率' % (2011 + i), '预测', '观测']
    darw_pre(sta, prec_pre[i], prec_obs[i], title)
    plt.savefig(r'D:\sx08\%0.f.jpg' % (2011 + i), dpi=300, bbox_inches='tight')

#预测准确率 P 
cal_p=prec_pre*prec_obs 
N=160 
P=pd.DataFrame(np.sum(cal_p>0,axis = 1)/N*100,columns={'P'}) 
#预测评分 Ps 
No=np.zeros((5)) 
n1=No 
n2=No 
for i in range(5): 
 for j in range(N): 
 if 
(cal_p[i,j]>0)or((abs(prec_pre[i,j])<=0.13)&(abs(prec_obs[i,j])<=0.13)&(cal_p[i
,j]<0)): 
 No[i]+=1 
 elif (abs(prec_pre[i,j])>=0.5)&(abs(prec_obs[i,j])>=0.5): 
 n1[i]+=0 
 elif (abs(prec_pre[i,j])>=0.2)&(abs(prec_obs[i,j])>=0.2): 
 n2[i]+=0 
f1=5 
f2=2 
Ps=pd.DataFrame((No+f1*n1+f2*n2)/(N+f1*n1+f2*n2)*100,columns={'Ps'}) 
#异常级评分 TS 
Nf=np.sum(abs(prec_pre)>=0.2,axis = 1) 
No=np.sum(abs(prec_obs)>=0.2,axis = 1) 
Nc=np.zeros((5)) 
9 
for i in range(5): 
 for j in range(N): 
 if (abs(prec_pre[i,j])>=0.2)&(abs(prec_obs[i,j])>=0.2): 
 Nc[i]+=1 
TS=pd.DataFrame(Nc/(No+Nf-Nc)*100,columns={'Ts'}) 
#距平相关系数 ACC 
Rf=prec_pre-prec_pre.mean(0) 
ave_Rf=prec_pre.mean(1) 
Ro=prec_obs-prec_obs.mean(0) 
ave_Ro=prec_obs.mean(1) 
Sum_1=np.zeros((5)) 
Sum_2=np.zeros((5)) 
for i in range(5): 
 for j in range(160): 
 Sum_1[i]+=(Rf[i,j]-ave_Rf[i])*(Ro[i,j]-ave_Ro[i]) 
for i in range(5): 
 a1=0 
 b1=0 
 for j in range(160): 
 a1+=((Rf[i,j]-ave_Rf[i])**2) 
 b1+=((Ro[i,j]-ave_Ro[i])**2) 
 Sum_2[i]=np.sqrt(a1*b1) 
 
ACC=pd.DataFrame(Sum_1/Sum_2*100,columns={'ACC'}) 
#转化为表格 
sheet=pd.concat([P,Ps,TS,ACC],axis=1).T# 
sheet.columns = ["2011", "2012", "2013","2014", "2015"] 
sheet.columns.names=['%'] 
sheet.to_excel(r'D:/data/short_term_climate_class/sx08/sheet.xlsx') 
