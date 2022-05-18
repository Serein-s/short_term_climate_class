import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
from eofs.standard import Eof

#读取nc数据集
f_z = xr.open_dataset("D:\data\short_term_climate_class\sx01\hgt.mon.mean.nc", drop_variables=["time_bnds"])

#索取1979-2020年1月欧亚地区（20-70N, 40-140E）500hPa高度的hgt
z = f_z['hgt'].loc[f_z.time.dt.month.isin([1])].loc['1979-01-01':'2020-02-01'].loc[:,500,70:20,40:140]

#索取1991-2020年1月欧亚地区（20-70N, 40-140E）500hPa高度的hgt
z1 = f_z['hgt'].loc[f_z.time.dt.month.isin([1])].loc['1991-01-01':'2020-02-01'].loc[:,500,70:20,40:140]

#求平均位势高度场（即气候态）
ave_hgt = np.array(z1).reshape(30, 21, 41).mean(0) 

#提取纬度
lat_z = f_z['lat'].loc[70:20]

#提取经度
lon_z = f_z['lon'].loc[40:140]

#权重处理
coslat = np.cos(np.deg2rad(np.array(lat_z))).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]

#eof实例
eof = Eof(np.array(z-ave_hgt), weights=wgts)#数据距平后
k=5#模态数,此次取5次
z_eof = eof.eofsAsCorrelation(neofs=k)#特征向量
z_pc = eof.pcs(npcs=k, pcscaling=1)#时间序列，1：除特征值平方根
z_var = eof.varianceFraction(neigs=k)#方差系数

#检验函数
def test_north(a,b,k):
    for i in range (0,k-1):
        if (b[i+1]+a[i+1])<(b[i]-a[i]):
            print('第%d模态和第%d模态显著分离，通过north检验'%(i+1,i+2))
        else:
            print('第%d模态和第%d模态分离不显著，没有通过north检验'%(i+1,i+2))
    for i in range (0,4):
            print('第%d模态：%.2f'%(i+1,z_var[i]*100)+'%')

#底图投影绘制函数
def contour_map(fig_ax,img_extent,spec,eof_name,data_z_var,size,lon,lat,data_eof):
    #投影设置
    fig_ax.set_extent(img_extent, crs=ccrs.PlateCarree())
    #填加海岸线
    fig_ax.add_feature(cfeature.COASTLINE.with_scale('50m')) 
    #添加湖泊
    fig_ax.add_feature(cfeature.LAKES, alpha=0.5)
    #添加经纬度设置
    fig_ax.set_xticks(np.arange(leftlon,rightlon+spec,spec), crs=ccrs.PlateCarree())
    fig_ax.set_yticks(np.arange(lowerlat,upperlat+spec,spec), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    fig_ax.xaxis.set_major_formatter(lon_formatter)
    fig_ax.yaxis.set_major_formatter(lat_formatter)
    #添加标题设置
    fig_ax.set_title(eof_name,loc='left',fontsize =size)
    fig_ax.set_title( '%.2f%%' % (data_z_var*100),loc='right',fontsize =size)
    #绘制填色图
    ax_colorbar=fig_ax.contourf(lon,lat,data_eof,levels=np.arange(-0.9,1.0,0.1), extend = 'both',
                                transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r)
    #绘制等值线
    fig_ax.contour(lon,lat,data_eof,levels=np.arange(-0.9,1.0,0.1),colors='black',
                   transform=ccrs.PlateCarree())
    #shrink控制colorbar长度，pad控制colorbar和图的距离
    plt.rcParams['axes.unicode_minus'] = False##负号显示问题
    plt.colorbar(ax_colorbar, shrink=0.6, pad=0.03)

#条形图底图绘制函数
def bar_map(fig_ax,pc_name,data_z_var,size,x,data_pc,start_year,end_year):
    c_color=[]#条形图颜色：红+，蓝-
    for i in range(start_year,end_year):
        if data_pc[i-start_year] >=0:
              c_color.append('red')
        elif data_pc[i-start_year] <0:
              c_color.append('blue')
    #设置标题
    fig_ax.set_title('(b) PC1',loc='left',fontsize =size)
    fig_ax.set_title('%.2f%%' % (data_z_var*100),loc='right',fontsize=size)
    #设置轴范围
    fig_ax.set_ylim(-3.1,3.1)
    #y=0设置为虚线
    fig_ax.axhline(0,linestyle="--")
    #设置刻度值大小
    plt.xticks(size=size)
    plt.yticks(size = size)
    #绘制条形图
    fig_ax.bar(x,data_pc,color=c_color)


#公共设置    
start_year,end_year=(1979,2020)#起始年份
years=range(start_year,end_year+1)
proj = ccrs.PlateCarree(central_longitude=90)#投影设置(中心为90°E)
leftlon, rightlon, lowerlat, upperlat = (40,140,20,70)#经纬度范围
img_extent = [leftlon, rightlon, lowerlat, upperlat]
a=eof.northTest(neigs=k)#north检验
b=eof.eigenvalues(neigs=k)#特征根

#打印检验结果结果
test_north(a,b,k)

#设置画布
fig = plt.figure(figsize=(15,15))
#子图11
fig_ax11 = fig.add_axes([0.05, 0.75, 0.61, 0.405],projection = proj)
contour_map(fig_ax11,img_extent,10,'(a) EOF1',z_var[0],15,lon_z,lat_z,z_eof[0,:,:])
#子图12
fig_ax12 = fig.add_axes([0.65, 0.821, 0.42, 0.258])
bar_map(fig_ax12,'(b) PC1',z_var[0],15,years,z_pc[:,0],1979,2020)
#子图21
fig_ax11 = fig.add_axes([0.05, 0.40, 0.61, 0.405],projection = proj)
contour_map(fig_ax11,img_extent,10,'(c) EOF2',z_var[1],15,lon_z,lat_z,-1*z_eof[1,:,:])
#子图22
fig_ax12 = fig.add_axes([0.65, 0.472, 0.42, 0.258])
bar_map(fig_ax12,'(d) PC1',z_var[1],15,years,-1*z_pc[:,1],1979,2020)
#子图31
fig_ax11 = fig.add_axes([0.05, 0.10, 0.61, 0.405],projection = proj)
contour_map(fig_ax11,img_extent,10,'(e) EOF3',z_var[2],15,lon_z,lat_z,-1*z_eof[2,:,:])
#子图32
fig_ax12 = fig.add_axes([0.65, 0.172, 0.42, 0.258])
bar_map(fig_ax12,'(f) PC1',z_var[2],15,years,-1*z_pc[:,2],1979,2020)
#存图
plt.savefig(r'D:\data\short_term_climate_class\sx02\eof.jpg',dpi=300,bbox_inches='tight')
plt.show()
