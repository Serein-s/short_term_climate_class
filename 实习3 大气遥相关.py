import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import maskout31
from scipy.signal import detrend 
from scipy.stats import pearsonr
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import cartopy.mpl.ticker as cticker
#读取nc
tem=xr.open_dataset(r"D:\data\short_term_climate_class\sx03\air.2m.mon.mean.nc",
                    drop_variables=["time_bnds"])
air= tem.air.loc[tem.time.dt.month.isin([1])].loc['1979-01-01':'2020-02-01',2.0]
z=xr.open_dataset(r"D:\data\short_term_climate_class\sx01\hgt.mon.mean.nc", 
                  drop_variables=["time_bnds"])
z_500= z.hgt.loc[z.time.dt.month.isin([1])].loc['1979-01-01':'2020-02-01', 500]
z_60_155=z_500.loc[:,60,155]
z_30_155=z_500.loc[:,30,155]
#去趋势函数
z_1=detrend(z_60_155,axis=0,type='linear')
z_2=detrend(z_30_155,axis=0,type='linear')
det_air=detrend(air,axis=0,type='linear',overwrite_data=False)
det_z=detrend(z_500,axis=0,type='linear',overwrite_data=False)
#标准化
WP=(z_1-z_2)/2.
WP_nor=(WP-WP.mean())/(WP.std())
#求相关系数
def cor(x,y,data_1,data_2 ):
        r=np.zeros(shape=(y,x))
        p=np.zeros(shape=(y,x))
        test_r=np.zeros(shape=(y,x))
        sum=0.0
        for i in range(y):
              for j in range(x):
                    r[i,j],p[i,j]=pearsonr(data_1,data_2[:,i,j])
                    if abs(p[i,j])>=0.01:
                            test_r[i,j]=np.NaN
                    else:
                            test_r[i,j]=r[i,j]
        
        return r,test_r 
    
#填色底图
def c_map(ax,img_extent,spec,a):           
        proj=ccrs.PlateCarree()
        ax.set_extent(img_extent,crs = proj)
        if a==1:
            ax.set_xticks(np.arange(img_extent[0], 
                                    img_extent[1] + spec, spec), crs = proj)
            ax.set_yticks(np.arange(img_extent[2], 
                                    img_extent[3] + spec, spec),crs = proj)
            lon_formatter = cticker.LongitudeFormatter()
            lat_formatter = cticker.LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
        else:
            
            pass

        ax.add_feature(cfeature.OCEAN.with_scale('50m'))#海洋
        ax.add_feature(cfeature.LAND.with_scale('50m'))#陆地
        ax.add_feature(cfeature.LAKES.with_scale('50m'))#湖泊                        
        ax.add_geometries(Reader(r'D:\data\china_map\river1.shp').geometries(),
                          ccrs.PlateCarree(),facecolor='none',edgecolor='b',linewidth=0.4)#长江黄河
        ax.add_geometries(Reader(r'D:\data\china_map\china1.shp').geometries(),
                          ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.5)#详细国界中国国界
        ax.add_geometries(Reader(r'D:\data\china_map\china2.shp').geometries(),
                          ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.35)#省界
        ax.add_geometries(Reader(r'D:\data\china_map\ne_10m_land.shp').geometries(),
                          ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)#海岸线
        ax.add_geometries(Reader(r'D:\data\china_map\ne_50m_lakes.shp').geometries(),
                          ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)#湖泊
        #ax.add_geometries(Reader(r"D:\data\china_map\china0").geometries(),
            #ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.2)#简单国界
        ax.add_geometries(Reader(r"D:\data\china_map\country1.shp").geometries(),
                          ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.1)


#WP与500hPa的相关
r,test_r=cor(144,73,WP_nor,det_z)
#WP与中国同期气温的相关
r_t,test_r_t=cor(192,94,WP_nor,det_air)

#条形图底图
def bar_map(fig_ax,size,data_pc,start_year,end_year):
    c_color=[]
    for i in range(start_year,end_year+1):
        if data_pc[i-start_year] >0:
              c_color.append('red')
        elif data_pc[i-start_year]<=0:
              c_color.append('blue')
    #设置轴范围
    fig_ax.set_ylim(-3.1,3.1)
    # y=0设置为虚线
    fig_ax.axhline(0,linestyle="--")
    #设置刻度值大小
    plt.xticks(size=size)
    plt.yticks(size = size)
    #绘制条形图
    fig_ax.bar(range(start_year,end_year+1),data_pc,color=c_color)
    
#北半球极地投影底图
def NPS_map(ax,extent):
    ax.set_extent(extent, ccrs.PlateCarree())       
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))#海岸线
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))#海洋
    ax.add_feature(cfeature.LAND.with_scale('50m'))#陆地
    ax.add_feature(cfeature.LAKES.with_scale('50m'))#湖泊
    ax.add_geometries(Reader(r'D:\data\china_map\country1.shp').geometries(),
          ccrs.PlateCarree(),facecolor='none',edgecolor='black',linewidth=0.5)
    #经纬网
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='k', alpha=0.85, linestyle='--')
    # 调节字体大小
    gl.xlabel_style={'size':15}
    gl.ylabel_style={'size':15}

proj =ccrs.NorthPolarStereo(central_longitude=150)
leftlon, rightlon, lowerlat, upperlat = (-180,180,10,90)
img_extent = [leftlon, rightlon, lowerlat, upperlat]

#画布
fig = plt.figure(figsize=(12,12))

#子图1
f_ax1 = fig.add_axes([0.05, 0.75, 0.5, 0.5],projection = proj)
#注意此处添加了projection = ccrs.NorthPolarStereo()北半球极地投影                      
NPS_map(f_ax1,img_extent)
#填色
ax_colorbar=f_ax1.contourf(z.lon,z.lat,test_r,levels =np.arange(-1,1.001,0.1),
                           extend = 'both',transform=ccrs.PlateCarree(),cmap='RdBu_r')
f_ax1.contour(ax_colorbar,levels =np.arange(-1,1.001,0.1),linewidths=0.5,
              colors='black',transform=ccrs.PlateCarree())
#绘制填色，需要说明的是：虽然是极地投影，但是我们的数据仍是按圆柱投影计算的，
#所以数据的坐标转换仍为transform=ccrs.PlateCarree()
#色标
plt.colorbar(ax_colorbar, shrink=0.68, pad=0.15)
#标题
f_ax1.set_title('WP and 500hPa Corr_Index',fontsize =15)
f_ax1.set_title('(a)', loc='left', fontsize=15)


#子图2
#WP--条形图
f_ax2 =fig.add_axes([0.6, 0.85, 0.45, 0.35])
bar_map(f_ax2,15,WP_nor,1979,2020)
f_ax2.set_title('1979-2020 January WP Index',fontsize =15)
f_ax2.set_title('(b)', loc='left', fontsize=15)


#子图3
f_ax3 = fig.add_axes([0.05, 0.3, 0.5, 0.5],projection = proj)
#注意此处添加了projection = ccrs.NorthPolarStereo()北半球极地投影                      
NPS_map(f_ax3,img_extent)
#填色
ax3_colorbar=f_ax3.contourf(air.lon,air.lat,test_r_t,levels =np.arange(-1,1.001,0.1),
                            extend = 'both',transform=ccrs.PlateCarree(),cmap='RdBu_r')
f_ax3.contour(ax3_colorbar,levels =np.arange(-1,1.001,0.1),linewidths=0.5,colors='black',
              transform=ccrs.PlateCarree())
plt.colorbar(ax3_colorbar, shrink=0.68, pad=0.15)
#标题
f_ax3.set_title('WP and China_tem Test_Corr_Index',fontsize =15)
f_ax3.set_title('(c)', loc='left', fontsize=15)


#子图4
f_ax4 = fig.add_axes([0.6, 0.33, 0.45, 0.5],projection = ccrs.PlateCarree())
#填色
c11=f_ax4.contourf(air.lon,air.lat, r_t, levels=np.arange(-1,1.1,0.1),extend='both',
                   transform=ccrs.PlateCarree(),cmap='seismic')
c12=f_ax4.contour(c11,levels=np.arange(-1,1.1,0.1),transform=ccrs.PlateCarree(),
                  colors='k',linewidths=0.45)
#白化
clip1=maskout31.shp2clip(c11,f_ax4,r"D:\data\china_map\china0.shp")
clip11=maskout31.shp2clip(c12,f_ax4,r"D:\data\china_map\china0.shp")
c_map(f_ax4,[70, 140, 15, 55],10,1)
#标题
f_ax4.set_title('WP and China_tem Corr_Index',fontsize =15) 
f_ax4.set_title('(d)', loc='left', fontsize=15)
#色标
cbar=plt.colorbar(c11,shrink=0.75,orientation='horizontal',pad=0.05) 
cbar.ax.tick_params(labelsize=12, direction='in')

#添加南海
ax2 = fig.add_axes([0.838, 0.43, 0.37, 0.085],projection=ccrs.PlateCarree())
c_map(ax2,[105, 122,0,26],10,0)

plt.savefig(r'D:\data\short_term_climate_class\sx03\WP.jpg',dpi=300,bbox_inches='tight')
plt.show()
