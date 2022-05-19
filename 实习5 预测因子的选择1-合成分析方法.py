from scipy.stats.mstats import ttest_ind
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
#防止中文出错
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

#6月份降水
f_pre_6 = pd.read_csv(r"D:\data\short_term_climate_class\sx05\r1606.txt",
                      sep="\s+",
                      header=None)
pre_6 = np.array(f_pre_6).reshape(71, 160)
#7月份降水
f_pre_7 = pd.read_csv(r"D:\data\short_term_climate_class\sx05\r1607.txt",
                      sep="\s+",
                      header=None)
pre_7 = np.array(f_pre_7).reshape(71, 160)
#8月份降水
f_pre_8 = pd.read_csv(r"D:\data\short_term_climate_class\sx05\r1608.txt",
                      sep="\s+",
                      header=None)
pre_8 = np.array(f_pre_8).reshape(71, 160)

pre = ((pre_6 + pre_7 + pre_8) / 3.)[0:55, :]
#高度
f_hgt = xr.open_dataset(
    r"D:\data\short_term_climate_class\sx05\hgt.mon.mean.nc")
#500hPa
z = f_hgt.hgt.loc[f_hgt.time.dt.month.isin(
    [12, 1,2])].loc['1950-12-01':'2005-02-01',500, 90:0, :]
#站点
sta = pd.read_csv(r"D:\data\short_term_climate_class\sx05\zd.txt",
                  sep="\s+",
                  header=None,
                  names=['station', 'lat', 'lon'])
#三类雨型年
type_1_year=np.array([1953,1958,1959,1960,1961,1964,1966,1967,1973,1976,
                      1977,1978,1981,1985,1988,1992,1994,1995,2001,2004])
type_2_year=np.array([1956,1957,1962,1963,1965,1971,1972,1975,1979,1982,
                      1984,1989,1990,1991,2000,2003,2005])
type_3_year=np.array([1951,1952,1954,1955,1968,1969,1970,1974,1980,1983,
                      1986,1987,1993,1996,1997,1998,1999,2002])

#计算雨型
def cal_pre(type_year, pre):
    type_year_pre = pre[type_year - 1951]
    ave_pre = np.tile((pre[0:50, :]).mean(0), (type_year_pre.shape[0], 1))
    pre_per = ((type_year_pre - ave_pre) / ave_pre).mean(0)
    return pre_per


pre_per_1 = cal_pre(type_1_year, pre)
pre_per_2 = cal_pre(type_2_year, pre)
pre_per_3 = cal_pre(type_3_year, pre)


#计算500hPa合成 并 T检验
def cal_hgt(type_year, z):
    ave_z = (np.array(z).reshape(-1, 3, 37, 144).mean(1)).mean(0)
    hgt = z.loc[((z.time.dt.month.isin([1, 2])) &
                 (z.time.dt.year.isin(type_year))) |
                ((z.time.dt.month.isin([12])) &
                 (z.time.dt.year.isin(type_year - 1)))]
    hgt_winter = (np.array(hgt).reshape(-1, 3, 37, 144).mean(1)).mean(0)
    hgt_jp = hgt_winter - ave_z
    #T检验
    _, p_hgt = ttest_ind(hgt, z, equal_var=False)
    return hgt_jp, p_hgt

hgt_1_winter, p_hgt_1 = cal_hgt(type_1_year, z)
hgt_2_winter, p_hgt_2 = cal_hgt(type_2_year, z)
hgt_3_winter, p_hgt_3 = cal_hgt(type_3_year, z)

#散点底图
def Lbt_map(ax, extent, mark=1):
    ax.set_extent(extent)
    ax.add_feature(cfeature.COASTLINE, lw=0.3)
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.add_geometries(Reader(r"D:\data\china_map\river1.shp").geometries(),
                      ccrs.PlateCarree(),
                      facecolor='none',
                      edgecolor='b',
                      linewidth=0.6)
    ax.add_geometries(Reader(r'D:\data\china_map\china1.shp').geometries(),
                      ccrs.PlateCarree(),
                      facecolor='none',
                      edgecolor='k',
                      linewidth=0.5)
    ax.add_geometries(Reader(r"D:\data\map\bou2_4l.shp").geometries(),
                      ccrs.PlateCarree(),
                      facecolor='none',
                      edgecolor='k',
                      linewidth=0.7)
    ax.add_geometries(
        Reader(r'D:\data\china_map\ne_10m_land.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='k',
        linewidth=0.5)
    ax.add_geometries(
        Reader(r'D:\data\china_map\ne_50m_lakes.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='k',
        linewidth=0.5)
    if mark == 1:
        lb = ax.gridlines(draw_labels=None,
                          x_inline=False,
                          y_inline=False,
                          linewidth=0.5,
                          color='gray',
                          alpha=0.5,
                          linestyle='--')
        lb.xlocator = mticker.FixedLocator(range(0, 180, 10))
        lb.ylocator = mticker.FixedLocator(range(0, 90, 10))
        lb = ax.gridlines(draw_labels=True,
                          x_inline=False,
                          y_inline=False,
                          linewidth=0.5,
                          color='gray',
                          alpha=0.5,
                          linestyle='--')
        lb.top_labels = False
        lb.right_labels = None
        lb.xlocator = mticker.FixedLocator(range(90, 130, 10))
        lb.ylocator = mticker.FixedLocator(range(10, 60, 10))
        lb.ylabel_style = {'size': 15, 'color': 'k'}
        lb.xlabel_style = {'size': 15, 'color': 'k'}
        lb.rotate_labels = False
    else:
        pass

def contour_map(fig, img_extent, spec):  # 画布，经纬度范围，步长
    fig.set_ylim((lowerlat, upperlat))
    fig.set_xlim((leftlon, rightlon))
    #fig.set_extent(img_extent, crs=ccrs.PlateCarree())
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
    plt.xticks(fontsize=15)  # 设置标签大小
    plt.yticks(fontsize=15)


def cycle_data(hgt_jp, z):
    chgt, cycle_lon = add_cyclic_point(hgt_jp, coord=z.lon)
    LON, LAT = np.meshgrid(cycle_lon, z.lat)
    return LON, LAT, chgt

c_hgt_1_lon, c_hgt_1_lat, c_hgt_1 = cycle_data(hgt_1_winter, z)
c_hgt_2_lon, c_hgt_2_lat, c_hgt_2 = cycle_data(hgt_2_winter, z)
c_hgt_3_lon, c_hgt_3_lat, c_hgt_3 = cycle_data(hgt_3_winter, z)

leftlon, rightlon, lowerlat, upperlat = (-180, 180, 0, 90)
img_extent = [leftlon, rightlon, lowerlat, upperlat]
lev = np.arange(-60, 60, 3)
#画布
fig = plt.figure(figsize=(15, 15))
#降水
map = ccrs.LambertConformal(central_longitude=105)

ax1 = fig.add_subplot(4, 3, 1, projection=map)
Lbt_map(ax1, [80, 130, 15, 55], 1)
c11 = ax1.scatter(sta["lon"],
                  sta["lat"],
                  vmin=-45,
                  vmax=45,
                  s=15,
                  c=pre_per_1 * 100,
                  cmap='jet',
                  transform=ccrs.PlateCarree())
ax1.set_title('Ⅰ类雨型降水距平百分率分布(北方型)', loc='right', fontsize=15)
ax1.set_title('(a)', loc='left', fontsize=15)

ax2 = fig.add_subplot(4, 3, 2, projection=map)
Lbt_map(ax2, [80, 130, 15, 55], 1)
ax2.scatter(sta["lon"],
            sta["lat"],
            vmin=-45,
            vmax=45,
            s=15,
            c=pre_per_2 * 100,
            cmap='jet',
            transform=ccrs.PlateCarree())
ax2.set_title('Ⅱ类雨型降水距平百分率分布(中间型)', loc='right', fontsize=15)
ax2.set_title('(b)', loc='left', fontsize=15)

ax3 = fig.add_subplot(4, 3, 3, projection=map)
Lbt_map(ax3, [80, 130, 15, 55], 1)
ax3.scatter(sta["lon"],
            sta["lat"],
            vmin=-45,
            vmax=45,
            s=15,
            c=pre_per_3 * 100,
            cmap='jet',
            transform=ccrs.PlateCarree())
ax3.set_title('Ⅲ类雨型降水距平百分率分布(南方型)', loc='right', fontsize=15)
ax3.set_title('(c)', loc='left', fontsize=15)

ax = fig.add_axes([0.98, 0.778, 0.012, 0.18])
cbar = plt.colorbar(c11, cax=ax)
cbar.ax.tick_params(labelsize=12, direction='in')
cbar.set_ticks(np.arange(-45, 45.1, 10))
cbar.ax.set_title('%', fontsize=20)

#南海
ax_nh1 = fig.add_axes([0.208, 0.770, 0.14, 0.06],
                      projection=ccrs.PlateCarree())
ax_nh1.scatter(sta["lon"],
               sta["lat"],
               vmin=-45,
               vmax=45,
               s=5,
               c=pre_per_1 * 100,
               cmap='jet',
               transform=ccrs.PlateCarree())
Lbt_map(ax_nh1, [106, 122, 0, 24], 0)

ax_nh2 = fig.add_axes([0.537, 0.770, 0.14, 0.06],
                      projection=ccrs.PlateCarree())
ax_nh2.scatter(sta["lon"],
               sta["lat"],
               vmin=-45,
               vmax=45,
               s=5,
               c=pre_per_2 * 100,
               cmap='jet',
               transform=ccrs.PlateCarree())
Lbt_map(ax_nh2, [106, 122, 0, 24], 0)

ax_nh3 = fig.add_axes([0.866, 0.770, 0.14, 0.06],
                      projection=ccrs.PlateCarree())
ax_nh3.scatter(sta["lon"],
               sta["lat"],
               vmin=-45,
               vmax=45,
               s=5,
               c=pre_per_3 * 100,
               cmap='jet',
               transform=ccrs.PlateCarree())
Lbt_map(ax_nh3, [106, 122, 0, 24], 0)

#500hPa
proj=ccrs.PlateCarree(central_longitude=180)
f_ax1 = fig.add_subplot(4, 1, 2, projection=proj)
c1 = f_ax1.contourf(c_hgt_1_lon,
                    c_hgt_1_lat,
                    c_hgt_1,
                    levels=lev,
                    cmap='bwr',
                    transform=ccrs.PlateCarree())
f_ax1.contour(c1, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
c1p = f_ax1.contourf(z.lon,
                     z.lat,
                     p_hgt_1,
                     levels=[0, 0.05, 1],
                     hatches=['.', None],
                     colors="none",
                     transform=ccrs.PlateCarree())
contour_map(f_ax1, img_extent, 30)
f_ax1.set_title('Ⅰ类雨型前期冬季 500hPa 高度距平合成', fontsize=15)
f_ax1.set_title('(d)', loc='left', fontsize=15)

f_ax2 = fig.add_subplot(4, 1, 3, projection=proj)
c2 = f_ax2.contourf(c_hgt_2_lon,
                    c_hgt_2_lat,
                    c_hgt_2,
                    levels=lev,
                    cmap='bwr',
                    transform=ccrs.PlateCarree())
f_ax2.contour(c2, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
c2p = f_ax2.contourf(z.lon,
                     z.lat,
                     p_hgt_2,
                     levels=[0, 0.05, 1],
                     hatches=['.', None],
                     colors="none",
                     transform=ccrs.PlateCarree())
contour_map(f_ax2, img_extent, 30)
f_ax2.set_title('Ⅱ类雨型前期冬季 500hPa 高度距平合成', fontsize=15)
f_ax2.set_title('(e)', loc='left', fontsize=15)

f_ax3 = fig.add_subplot(4, 1, 4, projection=proj)
c3 = f_ax3.contourf(c_hgt_3_lon,
                    c_hgt_3_lat,
                    c_hgt_3,
                    levels=lev,
                    cmap='bwr',
                    transform=ccrs.PlateCarree(),
                    extend='both')
f_ax3.contour(c3, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
c3p = f_ax3.contourf(z.lon,
                     z.lat,
                     p_hgt_3,
                     levels=[0, 0.05, 1],
                     hatches=['..', None],
                     colors="none",
                     transform=ccrs.PlateCarree())
contour_map(f_ax3, img_extent, 30)
f_ax3.set_title('Ⅲ类雨型前期冬季 500hPa 高度距平合成', fontsize=15)
f_ax3.set_title('(f)', loc='left', fontsize=15)

axc = fig.add_axes([0.13, -0.02, 0.75, 0.012])
cbar = plt.colorbar(c3, cax=axc, orientation='horizontal')
cbar.ax.tick_params(labelsize=12, direction='in')
cbar.set_ticks(np.arange(-60, 60.1, 10))
cbar.ax.set_title('m', fontsize=20)

plt.tight_layout()

plt.show()
