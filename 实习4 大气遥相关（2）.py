import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.util import add_cyclic_point
from scipy.stats import pearsonr
from scipy.signal import detrend
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr
import maskout31
import matplotlib.pyplot as plt
import matplotlib.path as mpath

#读取nc，除去多余变量
prec = xr.open_dataset(
    r"D:\data\short_term_climate_class\sx04\precip.mon.mean.nc",
    drop_variables=["time_bnds", 'lat_bnds', 'lon_bnds'])
tem = xr.open_dataset(
    r"D:\data\short_term_climate_class\sx03\air.2m.mon.mean.nc",
    drop_variables=["time_bnds"])
z = xr.open_dataset(
    r"D:\data\short_term_climate_class\sx01\hgt.mon.mean.nc",
    drop_variables=["time_bnds"])

#提取数据
#500hPa
z_500 = z.hgt.loc[z.time.dt.month.isin([1])].loc[
         '1979-01-01':'2020-02-01',500, 90:0, 0:360]
#同期中国气温
air = tem.air.loc[tem.time.dt.month.isin([1])].loc[
              '1979-01-01':'2020-02-01',2.0, 55:0, 70:140]
#同期中国降水
pre = prec.precip.loc[prec.time.dt.month.isin([7])].loc[
           '1979-01-01':'2020-12-01', 0:55, 70:140]


#去趋势函数det()
def det(data, lon, lat, time):
    det_data = detrend(data, axis=0, type='linear')

    new_data = xr.DataArray(det_data,
                            dims=["time", "lat", "lon"],
                            coords={
                                'time': time,
                                'lat': lat,
                                'lon': lon
                            })
    return new_data


#求相关系数函数cor()
def cor(x, y, data_1, data_2):
    r = np.zeros(shape=(y, x))
    p = np.zeros(shape=(y, x))
    test_r = np.zeros(shape=(y, x))
    for i in range(y):
        for j in range(x):
            r[i, j], p[i, j] = pearsonr(data_1, data_2[:, i, j])
            if abs(p[i, j]) >= 0.05:
                test_r[i, j] = np.NaN
            else:
                test_r[i, j] = r[i, j]

    return r, p, test_r


#计算
new_z = det(z_500, z_500.lon, z_500.lat, z_500.time)
new_air = det(air, air.lon, air.lat, air.time)
new_pre = det(pre, pre.lon, pre.lat, pre.time)
#EU指数
EU = -new_z.loc[:, 55, 20] / 4. + new_z.loc[:, 55,75] / 2. - new_z.loc[:, 40,145] / 4.
#标准化
EU_nor = (EU - EU.mean(axis=0)) / (EU.std(axis=0))
EU_nor
#计算相关系数
r_z, p_z, test_r_z = cor(144, 37, EU_nor, new_z)
r_air, p_air, test_r_air = cor(37, 29, EU_nor, new_air)
r_pre, p_pre, test_r_pre = cor(28, 22, EU_nor, new_pre)
#南海
nh_air = new_air.loc[:, 25:0, 105:125]
nh_pre = new_pre.loc[:, 0:25, 105:125]
nh_r_air, nh_p_air, nh_test_r_air = cor(11, 13, EU_nor, nh_air)
nh_r_pre, nh_p_pre, nh_test_r_pre = cor(8, 10, EU_nor, nh_pre)


#PC序列底图
def bar_map(fig_ax, size, data_pc, start_year, end_year):
    c_color = []
    for i in range(start_year, end_year + 1):
        if data_pc[i - start_year] > 0:
            c_color.append('red')
        elif data_pc[i - start_year] <= 0:
            c_color.append('blue')
    fig_ax.set_ylim(-3.1, 3.1)
    fig_ax.axhline(0, linestyle="--")
    plt.xticks(size=size)
    plt.yticks(size=size)
    fig_ax.bar(range(start_year, end_year + 1), data_pc, color=c_color)


#填色底图
def c_map(ax, img_extent, spec, a, size=None):
    proj = ccrs.PlateCarree()
    ax.set_extent(img_extent, crs=proj)
    if a == 1:
        ax.set_xticks(np.arange(img_extent[0], img_extent[1] + spec, spec),
                      crs=proj)
        ax.set_yticks(np.arange(img_extent[2], img_extent[3] + spec, spec),
                      crs=proj)
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.xticks(fontsize=size)  # 设置标签大小
        plt.yticks(fontsize=size)
    else:

        pass

    #ax.add_feature(cfeature.OCEAN.with_scale('50m'))#海洋
    #ax.add_feature(cfeature.LAND.with_scale('50m'))#陆地
    ax.add_feature(cfeature.LAKES.with_scale('50m'))  #湖泊
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))

    ax.add_geometries(Reader(r'D:\data\china_map\river1.shp').geometries(),
                      ccrs.PlateCarree(),
                      facecolor='none',
                      edgecolor='b',
                      linewidth=0.4)  #长江黄河
    ax.add_geometries(Reader(r'D:\data\china_map\china1.shp').geometries(),
                      ccrs.PlateCarree(),
                      facecolor='none',
                      edgecolor='k',
                      linewidth=0.5)  #详细国界中国国界
    ax.add_geometries(Reader(r'D:\data\china_map\china2.shp').geometries(),
                      ccrs.PlateCarree(),
                      facecolor='none',
                      edgecolor='k',
                      linewidth=0.2)  #省界
    ax.add_geometries(
        Reader(r'D:\data\china_map\ne_10m_land.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='k',
        linewidth=0.4)  #海岸线
    ax.add_geometries(
        Reader(r'D:\data\china_map\ne_50m_lakes.shp').geometries(),
        ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='k',
        linewidth=0.3)  #湖泊


#北极点极地投影底图
def NPS_map(ax):
    ax.coastlines(linewidths=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent([-180, 180, 0, 90], ccrs.PlateCarree())

    # specifying xlocs/ylocs yields number of meridian/parallel lines
    dmeridian = 30  # spacing for lines of meridian
    dparallel = 15  # spacing for lines of parallel
    num_merid = 360 // dmeridian + 1
    num_parra = 90 // dparallel + 1
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      xlocs=np.linspace(-180, 180, num_merid),
                      ylocs=np.linspace(0, 90, num_parra),
                      linestyle="--",
                      linewidth=1,
                      color='k',
                      alpha=0.5)

    theta = np.linspace(0, 2 * np.pi, 120)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    center, radius = [0.5, 0.5], 0.5
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle,
                    transform=ax.transAxes)  #without this; get rect bound

    # for label alignment
    va = 'center'  # also bottom, top
    ha = 'center'  # right, left
    degree_symbol = u'\u00B0'

    # for locations of (meridional/longitude) labels
    lond = np.linspace(0, 360, num_merid)
    latd = np.zeros(len(lond))

    for (alon, alat) in zip(lond, latd):
        projx1, projy1 = ax.projection.transform_point(alon, alat,
                                                       ccrs.Geodetic())
        if alon > 0 and alon < 180:
            ha = 'left'
            va = 'center'
        if alon > 180 and alon < 360:
            ha = 'right'
            va = 'center'
        if np.abs(alon - 180) < 0.01:
            ha = 'center'
            va = 'bottom'
        if alon == 0.:
            ha = 'center'
            va = 'top'
        if (alon < 360.):
            txt = ' {0} '.format(str(int(alon))) + degree_symbol
            ax.text(projx1, projy1, txt, va=va, ha=ha, color='g')

    # for locations of (meridional/longitude) labels
    # select longitude: 225 for label positioning
    lond2 = 225 * np.ones(len(lond))
    latd2 = np.linspace(0, 90, num_parra)
    va, ha = 'center', 'center'
    for (alon, alat) in zip(lond2, latd2):
        projx1, projy1 = ax.projection.transform_point(alon, alat,
                                                       ccrs.Geodetic())
        txt = ' {0} '.format(str(int(alat))) + degree_symbol
        ax.text(projx1, projy1, txt, va=va, ha=ha, color='r')


fig = plt.figure(figsize=[12, 8])
#图(a)
proj = ccrs.NorthPolarStereo(central_longitude=0)
f_ax1 = fig.add_axes([0.32, 0.78, 0.51, 0.52], projection=proj)
c_data, cycle_lon = add_cyclic_point(test_r_z, coord=new_z.lon)
LON, LAT = np.meshgrid(cycle_lon, new_z.lat)
NPS_map(f_ax1)
cf1 = f_ax1.contourf(LON,
                     LAT,
                     c_data,
                     levels=np.arange(-1, 1.1, 0.2),
                     extend='both',
                     transform=ccrs.PlateCarree(),
                     cmap='RdBu_r')
#打点: 显著性水平为 95%
f_ax1.contourf(new_z.lon,
               new_z.lat,
               p_z, [0, 0.05, 1],
               zorder=1,
               hatches=['..', None],
               colors="none",
               transform=ccrs.PlateCarree())
f_ax1.set_title('EU and 500hPa Corr_Index', fontsize=15)
f_ax1.set_title('(a)', loc='left', fontsize=15)
position = fig.add_axes([0.45, 0.73, 0.25, 0.02])
cbar = plt.colorbar(cf1, cax=position, orientation='horizontal', format='%.1f')
cbar.ax.tick_params(labelsize=12, direction='in')

#图(b)
f_ax2 = fig.add_axes([0.98, 0.84, 0.49, 0.45])
bar_map(f_ax2, 15, EU_nor, 1979, 2020)
f_ax2.set_title('1979-2020 January EU Index', fontsize=15)
f_ax2.set_title('(b)', loc='left', fontsize=15)

#图(c)
f_ax3 = fig.add_axes([0.3, 0.02, 0.55, 0.55], projection=ccrs.PlateCarree())
cf3 = f_ax3.contourf(new_air.lon,
                     new_air.lat,
                     r_air,
                     levels=np.arange(-0.8, 0.5, 0.1),
                     extend='both',
                     transform=ccrs.PlateCarree(),
                     cmap='jet')
cf31 = f_ax3.contourf(new_air.lon,
                      new_air.lat,
                      p_air,
                      [np.min(p_air), 0.05, np.max(p_air)],
                      hatches=['..', None],
                      colors="none",
                      transform=ccrs.PlateCarree())
clip3 = maskout31.shp2clip(cf3, f_ax3, r"D:\data\china_map\china0.shp")
clip31 = maskout31.shp2clip(cf31, f_ax3, r"D:\data\china_map\china0.shp")
c_map(f_ax3, [70, 140, 15, 55], 10, 1, 14)
f_ax3.set_title('EU and China_tem Corr_Index', fontsize=15)
f_ax3.set_title('(c)', loc='left', fontsize=15)
cbar = plt.colorbar(cf3, shrink=0.65, orientation='horizontal', pad=0.09)
cbar.ax.tick_params(labelsize=12, direction='in')

#图(d)
f_ax4 = fig.add_axes([0.95, 0.02, 0.55, 0.55], projection=ccrs.PlateCarree())
cf4 = f_ax4.contourf(new_pre.lon,
                     new_pre.lat,
                     r_pre,
                     levels=np.arange(-0.4, 0.6, 0.1),
                     extend='both',
                     transform=ccrs.PlateCarree(),
                     cmap='jet')
cf41 = f_ax4.contourf(new_pre.lon,
                      new_pre.lat,
                      p_pre, [0, 0.05, 1],
                      zorder=1,
                      hatches=['..', None],
                      colors="none",
                      transform=ccrs.PlateCarree())
clip4 = maskout31.shp2clip(cf4, f_ax4, r"D:\data\china_map\china0.shp")
clip41 = maskout31.shp2clip(cf41, f_ax4, r"D:\data\china_map\china0.shp")
c_map(f_ax4, [70, 140, 15, 55], 10, 1, 14)
f_ax4.set_title('EU and China_July_pre Corr_Index', fontsize=15)
f_ax4.set_title('(d)', loc='left', fontsize=15)
cbar = plt.colorbar(cf4, shrink=0.65, orientation='horizontal', pad=0.09)
cbar.ax.tick_params(labelsize=12, direction='in')

#南海
ax1 = fig.add_axes([0.5825, 0.1535, 0.4, 0.15], projection=ccrs.PlateCarree())
c_map(ax1, [105, 122, 0, 24], 10, 0)
cf5 = ax1.contourf(nh_air.lon,
                   nh_air.lat,
                   nh_r_air,
                   levels=np.arange(-0.8, 0.5, 0.1),
                   extend='both',
                   transform=ccrs.PlateCarree(),
                   cmap='jet')
cf51 = ax1.contourf(nh_air.lon,
                    nh_air.lat,
                    nh_p_air, [0.0, 0.05, 1],
                    hatches=['..', None],
                    colors="none",
                    transform=ccrs.PlateCarree())
clip5 = maskout31.shp2clip(cf5, ax1, r"D:\data\china_map\china0.shp")
clip51 = maskout31.shp2clip(cf51, ax1, r"D:\data\china_map\china0.shp")

ax2 = fig.add_axes([1.236, 0.15, 0.4, 0.15], projection=ccrs.PlateCarree())
c_map(ax2, [106, 122, 0, 24], 10, 0)
cf6 = ax2.contourf(nh_pre.lon,
                   nh_pre.lat,
                   nh_r_pre,
                   levels=np.arange(-0.4, 0.6, 0.1),
                   extend='both',
                   transform=ccrs.PlateCarree(),
                   cmap='jet')
cf61 = ax1.contourf(nh_pre.lon,
                    nh_pre.lat,
                    nh_p_pre, [0.0, 0.05, 1],
                    hatches=['..', None],
                    colors="none",
                    transform=ccrs.PlateCarree())
clip5 = maskout31.shp2clip(cf6, ax2, r"D:\data\china_map\china0.shp")
clip51 = maskout31.shp2clip(cf61, ax2, r"D:\data\china_map\china0.shp")
plt.savefig(r'D:\data\short_term_climate_class\sx04\EU.jpg',
            dpi=300,
            bbox_inches='tight')
plt.show()
