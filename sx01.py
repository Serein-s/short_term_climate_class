from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker  # 刻度
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import matplotlib

# 防止中文出错
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus'] = False

f = xr.open_dataset(r"D:\data\short_term_climate_class\sx01\hgt.mon.mean.nc",
                    drop_variables=["time_bnds"])
lon = f.lon
lat = f.lat

# 索引1991-2020年之间1月的数据 500hPa
z = f.hgt.loc[f.time.dt.month.isin([1])].loc['1991-01-01':'2020-02-01', 500]
# 索引2008年1月的数据 500hPa
z2008 = f.hgt.loc[f.time.dt.month.isin([1])].loc['2008-01-01', 500]

# 求平均位势高度场（即气候态）
ave_hgt = np.array(z).reshape(30, 73, 144).mean(0)

# 求距平
dep = z2008 - ave_hgt  # dep 距平

# 求纬偏
ave_lon = np.array(z2008).reshape(73, 144).mean(1)  # 纬圈平均值
ave_lons = np.array([ave_lon] * 144).T  # 转置矩阵，使其维度与z2008相同
wp = z2008 - ave_lons  # wp 纬偏值

# 经纬度范围
leftlon, rightlon, lowerlat, upperlat = (-180, 180, -90, 90)
img_extent = [leftlon, rightlon, lowerlat, upperlat]


# 底图
def contour_map(fig, img_extent, spec):  # 画布，经纬度范围，步长
    fig.set_extent(img_extent, crs=ccrs.PlateCarree())
    fig.add_feature(cfeature.COASTLINE.with_scale('50m'))  # 添加海岸线
    fig.add_feature(cfeature.LAKES, alpha=0.5)  # 添加湖泊
    # 添加经纬度
    fig.set_xticks(np.arange(leftlon, rightlon + spec, spec),
                   crs=ccrs.PlateCarree())
    fig.set_yticks(np.arange(lowerlat, upperlat + spec, spec),
                   crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    fig.xaxis.set_major_formatter(lon_formatter)
    fig.yaxis.set_major_formatter(lat_formatter)


# 绘制1991-2020年平均1月500hPa位势高度图
# 除去0和360处空白(使用循环)
chgt, cycle_lon = add_cyclic_point(ave_hgt, coord=lon)
LON1, LAT1 = np.meshgrid(cycle_lon, lat)

# 生成画布
fig1 = plt.figure(figsize=(12, 5))
ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 绘制等值线图
c1 = ax1.contourf(LON1,
                  LAT1,
                  chgt / 10.0,
                  levels=np.arange(500, 600, 6),
                  cmap='jet',
                  transform=ccrs.PlateCarree())
c11 = ax1.contour(LON1,
                  LAT1,
                  chgt / 10.,
                  colors='black',
                  levels=np.arange(500, 600, 6),
                  linewidths=0.9)
contour_map(ax1, img_extent, 30)
# ax.clabel(contour)  #添加等值线的值
ax1.set_title('1991-2020年平均1月500hPa位势高度')  # 设置标签
plt.xticks(fontsize=13)  # 设置标签大小
plt.yticks(fontsize=13)
plt.colorbar(c1)  # 色标
plt.savefig('D:\data\short_term_climate_class\sx01\图1.jpg',
            dpi=300,
            bbox_inches='tight')  # 存图

# 绘制2008年1月500hPa位势高度距平图
# 除去0和360处空白
cdep, cycle_lon = add_cyclic_point(dep, coord=lon)
LON2, LAT2 = np.meshgrid(cycle_lon, lat)

# 生成画布
fig2 = plt.figure(figsize=(12, 5))
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 绘制等值线图
c2 = ax2.contourf(LON2,
                  LAT2,
                  cdep / 10.0,
                  levels=np.arange(-20, 20, 2),
                  cmap='seismic',
                  transform=ccrs.PlateCarree())
c21 = ax2.contour(LON2,
                  LAT2,
                  cdep / 10.,
                  colors='black',
                  levels=np.arange(-20, 20, 2),
                  linewidths=0.9)
contour_map(ax2, img_extent, 30)
# ax.clabel(contour)  #添加等值线的值
ax2.set_title('2008年1月500hPa位势高度距平（即相对于气候态的偏差）')  # 设置标签
plt.xticks(fontsize=13)  # 设置标签大小
plt.yticks(fontsize=13)

plt.colorbar(c2)  # 色标
plt.savefig('D:\data\short_term_climate_class\sx01\图2.jpg',
            dpi=300,
            bbox_inches='tight')  # 存图

# 绘制2008年1月500hPa位势高度纬偏值图
# 除去0和360处空白
cwp, cycle_lon = add_cyclic_point(wp, coord=lon)
LON3, LAT3 = np.meshgrid(cycle_lon, lat)

# 生成画布
fig3 = plt.figure(figsize=(12, 5))
ax3 = fig3.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 绘制等值线图
c3 = ax3.contourf(LON3,
                  LAT3,
                  cwp / 10.0,
                  levels=np.arange(-30, 30, 3),
                  cmap='RdBu_r',
                  transform=ccrs.PlateCarree())
c31 = ax3.contour(LON3,
                  LAT3,
                  cwp / 10.,
                  colors='black',
                  levels=np.arange(-30, 30, 3),
                  linewidths=1.0)
contour_map(ax3, img_extent, 30)
# ax.clabel(contour)  #添加等值线的值
ax3.set_title('2008年1月500hPa位势高度纬偏值')  # 设置标签
plt.xticks(fontsize=13)  # 设置标签大小
plt.yticks(fontsize=13)

#shrink控制colorbar长度，pad控制colorbar和图的距离
plt.rcParams['axes.unicode_minus'] = False  ##负号显示问题
plt.colorbar(c3, shrink=1.0, pad=0.03)

#plt.colorbar(c3)  # 色标
# position = fig.add_axes([0.91, 0.28, 0.02, 0.45])  # 坐标＋长宽
# cbar = fig.colorbar(contour, cax=position, orientation='vertical')
# 水平horizontal
plt.savefig('D:\data\short_term_climate_class\sx01\图3.jpg',
            dpi=300,
            bbox_inches='tight')  # 存图

plt.show()
