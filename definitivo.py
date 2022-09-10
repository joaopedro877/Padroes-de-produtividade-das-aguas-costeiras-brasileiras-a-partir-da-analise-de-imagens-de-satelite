# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 16:32:42 2022

@author: jdebr
"""

'''script definitivo dos resultados do trabalho'''


'''verificando a porcentagem de dados disponíveis para cada pixel'''
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import pandas as pd

'''acessando os dados de clorofila'''
fn = 'C:/Users/jdebr/Documents/clorofila_bahia/chl_ba_monthly.nc'
ds = nc.Dataset(fn)

'''definindo as variaveis'''
chl=ds['chlor_a'][:]
lat=ds['lat'][:]
lon=ds['lon'][:]

''' cada pixel possui dados em quantos % do tempo
ex : chl[56,18] tem 200 dados de clorofila, de um total de 228 dados possiveis'''
valid=np.zeros_like(chl[0])

for i in range(0,164):
    for j in range(0,102):
        valid[i,j]=(np.isnan(chl[:,i,j])).tolist().count(False)*100/228
valid[valid==np.nan]=[]
valid[valid==0]=np.nan
'''calculando a disponibilidade media de dados para a area'''
med_valid=round(np.nanmean(valid),2)

'''calculando quantos pixels possuem valor igual ou inferior a média'''
contador=0
for i in range(0,164):
    for j in range(0,102):
        if valid[i,j]<=med_valid:
            contador = contador + 1
menor_qmed=round((contador*100/(164*102)),1)      
'''visualisando a imagem'''
levels=np.linspace(0,100,11)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lon, lat,valid,60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet')
'''adicionando lat e lon'''
g = ax.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False
ax.coastlines(resolution = '10m')
fname = r'C:/Users/jdebr/Documents/clorofila_bahia/ne_10m_bathymetry_K_200.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='black',linestyle='--')
ax.add_feature(shape_feature, facecolor='none')
cb=plt.colorbar( ticks=[0,10,20,30,40,50,60,70,80,90,100])
plt.suptitle('Dados de clorofila-a válidos')
cb.ax.set_xlabel('% do total de 228')
plt.subplots_adjust(top=0.945,bottom=0.041,left=0.01,right=0.735,hspace=0.195,wspace=0.2)


#########################################################################################
'''excluindo pixels com baixa disponibilidade de dados(<90) e outros, como os da BTS'''
#########################################################################################
for i in range(0,164):
    for j in range(0,102):
        if valid[i,j]<90:
            chl[:,i,j]=np.nan

'''removendo a bts e outros'''
chl[:,29:36,22:31]=np.nan
chl[:,147,13]=np.nan
chl[:,126:128,14]=np.nan 
chl[:,114:116,17]=np.nan
chl[:,28,24:26]=np.nan   
chl[:,149,14]=np.nan  
chl[:,154,8]=np.nan
#########################################################################################
''' maiores, e  menores concentrações de clorofila  e média'''
#########################################################################################
#max
val_max=round(np.nanmax(chl),3)
max_con = np.where(chl==(np.nanmax(chl)))
#-13.781372(lat); -38.979168(lon)
#min
val_min= round(np.nanmin(chl),3)
min_con = np.where(chl==(np.nanmin(chl)))
#-16.779907 s; -36.479168 o 
#med e desvio
med_chl=round(np.nanmean(chl),3)
des_pad=round(np.nanstd(chl),3)

fig = plt.figure()
plt.suptitle('Concentração de clorofila x Tempo',fontsize=14)
ax = fig.add_subplot()
tempo=pd.date_range(start="2003-01-01",end="2021-12-31",freq='M')
ax.plot(tempo,chl[:,55,18],'-',color='red',label='13,781372°S; 38,979168°O')
ax.plot(tempo,chl[:,127,78],'-',color='blue',label='16,779907°S; 36,479168°O')
ax.legend()
ax.set_xlabel('Tempo')
ax.set_ylabel('Concentração de clorofila-a em mg/m³')

#########################################################################################
'''variação temporal da clorofila'''
#########################################################################################

#definindo os meses
chl_jan= chl[0:228:12]
chl_fev=chl[1:228:12]
chl_mar=chl[2:228:12]
chl_abr=chl[3:228:12]
chl_mai=chl[4:228:12]
chl_jun=chl[5:228:12]
chl_jul=chl[6:228:12]
chl_ago=chl[7:228:12]
chl_set=chl[8:228:12]
chl_out=chl[9:228:12]
chl_nov=chl[10:228:12]
chl_dez=chl[11:228:12]
#medias climatologicas
chl_jan_med = np.mean(chl_jan,axis=0)
chl_fev_med = np.mean(chl_fev,axis=0)
chl_mar_med = np.mean(chl_mar,axis=0)
chl_abr_med = np.mean(chl_abr,axis=0)
chl_mai_med = np.mean(chl_mai,axis=0)
chl_jun_med = np.mean(chl_jun,axis=0)
chl_jul_med = np.mean(chl_jul,axis=0)
chl_ago_med = np.mean(chl_ago,axis=0) 
chl_set_med = np.mean(chl_set,axis=0)
chl_out_med = np.mean(chl_out,axis=0)
chl_nov_med = np.mean(chl_nov,axis=0)
chl_dez_med = np.mean(chl_dez,axis=0)

#media da area para cada mes, com base na media climatologica
med_area=[]
med_area=[np.mean(chl_jan_med),np.mean(chl_fev_med),np.mean(chl_mar_med),np.mean(chl_abr_med),np.mean(chl_mai_med),np.mean(chl_jun_med),np.mean(chl_jul_med),np.mean(chl_ago_med),np.mean(chl_set_med),np.mean(chl_out_med),np.mean(chl_nov_med),np.mean(chl_dez_med)]

for i in range(0,12):
    med_area[i]=round(med_area[i],3)
plt.plot(med_area)

#desvio padrao da area para cada mes, com base na media climatologica
std_area=[np.std(chl_jan_med),np.std(chl_fev_med),np.std(chl_mar_med),np.std(chl_abr_med),np.std(chl_mai_med),np.std(chl_jun_med),np.std(chl_jul_med),np.std(chl_ago_med),np.std(chl_set_med),np.std(chl_out_med),np.std(chl_nov_med),np.std(chl_dez_med)]
for i in range(0,12):
    std_area[i]=round(std_area[i],3)

'''#meses de onde ocorrem as maiores concentrações, com base na média 
chl_ano_clim = np.empty((12,164,102))
chl_ano_clim[0]=chl_jan_med
chl_ano_clim[1]=chl_fev_med
chl_ano_clim[2]=chl_mar_med
chl_ano_clim[3]=chl_abr_med
chl_ano_clim[4]=chl_mai_med
chl_ano_clim[5]=chl_jun_med
chl_ano_clim[6]=chl_jul_med
chl_ano_clim[7]=chl_ago_med
chl_ano_clim[8]=chl_set_med
chl_ano_clim[9]=chl_out_med
chl_ano_clim[10]=chl_nov_med
chl_ano_clim[11]=chl_dez_med

mes_max=np.zeros_like(chl_abr_med)
mes_max[mes_max==0]=np.nan
def max_chl(x,y):
    return np.max(chl_ano_clim[:,x,y])

for i in range(0,164):
    for j in range(0,102):
        if max_chl(i,j)!= 0 : 
            mes_max[i,j]=(int((np.where(chl_ano_clim[:,i,j]==max_chl(i,j)))[0])+1)'''
#esta apresentando erro, nao consegui resolver ainda
          
##########################################################################################
'''variaçõa temporal e espacial - mapa das medias'''
##########################################################################################           

'''valores maximos e mínimos de media, para usar o mesmo colorbar'''
max1 = ((chl_jan_med.max()))
min1=((chl_jan_med.min()))

max2 = ((chl_fev_med.max()))
min2=((chl_fev_med.min()))

max3 = ((chl_mar_med.max()))
min3=((chl_mar_med.min()))

max4 = ((chl_abr_med.max()))
min4=((chl_abr_med.min()))

max5 = ((chl_mai_med.max()))
min5=((chl_mai_med.min()))

max6 = ((chl_jun_med.max()))
min6=((chl_jun_med.min()))

max7 = ((chl_jul_med.max()))
min7=((chl_jul_med.min()))

max8 = ((chl_ago_med.max()))
min8=((chl_ago_med.min()))

max9 = ((chl_set_med.max()))
min9=((chl_set_med.min()))

max10 = ((chl_out_med.max()))
min10=((chl_out_med.min()))

max11 = ((chl_nov_med.max()))
min11=((chl_nov_med.min()))

max12 = ((chl_dez_med.max()))
min12=((chl_dez_med.min()))

max_total=round(max(max1,max2,max3,max4,max5,max6,max7,max8,max9,max10,max11,max12),3)
min_total=round(min(min1,min2,min3,min4,min5,min6,min7,min8,min9,min10,min11,min12),3)
fname = r'C:/Users/jdebr/Documents/clorofila_bahia/ne_10m_bathymetry_K_200.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='black',linestyle='--')


fig = plt.figure()

levels=np.linspace(np.log10(min_total),np.log10(max_total),100)
ax1 = fig.add_subplot(2,6,1, projection=ccrs.PlateCarree())
ax1.contourf(lon, lat,np.log10(chl_jan_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax1.title.set_text('Janeiro')
ax1.coastlines()
ax1.add_feature(shape_feature, facecolor='none')
g = ax1.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False
g.xlabels_bottom = False 



ax2 = fig.add_subplot(2,6,2, projection=ccrs.PlateCarree())
ax2.contourf(lon, lat,np.log10(chl_fev_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax2.title.set_text('Fevereiro')
ax2.coastlines()
ax2.add_feature(shape_feature, facecolor='none')
g = ax2.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)


ax3 = fig.add_subplot(2,6,3, projection=ccrs.PlateCarree())
ax3.contourf(lon, lat,np.log10(chl_mar_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax3.title.set_text('Março')
ax3.coastlines()
ax3.add_feature(shape_feature, facecolor='none')
g = ax3.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)


ax4 = fig.add_subplot(2,6,4, projection=ccrs.PlateCarree())
ax4.contourf(lon, lat,np.log10(chl_abr_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax4.title.set_text('Abril')
ax4.coastlines()
ax4.add_feature(shape_feature, facecolor='none')
g = ax4.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)

ax5 = fig.add_subplot(2,6,5, projection=ccrs.PlateCarree())
ax5.contourf(lon, lat,np.log10(chl_mai_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax5.title.set_text('Maio')
ax5.coastlines()
ax5.add_feature(shape_feature, facecolor='none')
g = ax5.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)

ax6 = fig.add_subplot(2,6,6, projection=ccrs.PlateCarree())
ax6.contourf(lon, lat,np.log10(chl_jun_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax6.title.set_text('Junho')
ax6.coastlines()
ax6.add_feature(shape_feature, facecolor='none')
g = ax6.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)

ax7 = fig.add_subplot(2,6,7, projection=ccrs.PlateCarree())
ax7.contourf(lon, lat,np.log10(chl_jul_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax7.title.set_text('Julho')
ax7.coastlines()
ax7.add_feature(shape_feature, facecolor='none')
g = ax7.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False



ax8 = fig.add_subplot(2,6,8, projection=ccrs.PlateCarree())
ax8.contourf(lon, lat,np.log10(chl_ago_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax8.title.set_text('Agosto')
ax8.coastlines()
ax8.add_feature(shape_feature, facecolor='none')
g = ax8.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False
g.ylabels_left=False

ax9 = fig.add_subplot(2,6,9, projection=ccrs.PlateCarree())
ax9.contourf(lon, lat,np.log10(chl_set_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax9.title.set_text('Setembro')
ax9.coastlines()
ax9.add_feature(shape_feature, facecolor='none')
g = ax9.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False

ax10 = fig.add_subplot(2,6,10, projection=ccrs.PlateCarree())
ax10.contourf(lon, lat,np.log10(chl_out_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax10.title.set_text('Outubro')
ax10.coastlines()
ax10.add_feature(shape_feature, facecolor='none')
g = ax10.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False

ax11 = fig.add_subplot(2,6,11, projection=ccrs.PlateCarree())
ax11.contourf(lon, lat,np.log10(chl_nov_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax11.title.set_text('Novembro')
ax11.coastlines()
ax11.add_feature(shape_feature, facecolor='none')
g = ax11.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False

ax12 = fig.add_subplot(2,6,12, projection=ccrs.PlateCarree())
ax12.contourf(lon, lat,np.log10(chl_dez_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='jet',vmin=np.log10(min_total),vmax=np.log10(max_total))
ax12.title.set_text('Dezembro')
ax12.add_feature(shape_feature, facecolor='none')
ax12.coastlines()
g = ax12.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False

fig.tight_layout()
fig.subplots_adjust(right=0.7)

chl_med = (chl_jan_med+chl_fev_med+chl_mar_med+chl_abr_med+chl_mai_med+chl_jun_med+chl_jul_med+chl_ago_med+chl_set_med+chl_out_med+chl_nov_med+chl_dez_med)/12
im = ax.imshow(chl_jul_med,cmap='jet', vmin=np.log10(min_total),vmax=np.log10(max_total))
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb=plt.colorbar(im, cax=cbar_ax)
list = [-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75]
for i in list:
    i = 10**i
    print(i)
cb.ax.set_yticklabels(['0.06','0.1','0.18','0.32',' 0.56','1','1.78','3.16','5.62'])
cb.ax.set_xlabel('clorofila-a(mg/m³)',fontsize=12)
plt.suptitle('Concentração média de clorofila-a na superfície, para cada mês do ano ( mg/m³)',fontsize=18)
plt.show()

##############################################################################
'''analisando a temparatura'''
##############################################################################
'''acessando os dados de temperatura da superfície do mar'''
fn = 'C:/Users/jdebr/Documents/clorofila_bahia/sst_ba.nc'
ds1 = nc.Dataset(fn)
'''definindo as variaveis'''
sst=ds['sst'][:]
lat=ds['lat'][:]
lon=ds['lon'][:]
'''removendo a bts e outros'''
sst[:,29:36,22:31]=np.nan
sst[:,147,13]=np.nan
sst[:,126:128,14]=np.nan 
sst[:,114:116,17]=np.nan
sst[:,28,24:26]=np.nan   
sst[:,149,14]=np.nan  
sst[:,154,8]=np.nan
for i in range(0,164):
    for j in range(0,102):
        if (valid[i,j]-valid[i,j])!=0:
            sst[:,i,j]=np.nan

'''separando os meses'''


sst_jan= sst[0:228:12]
sst_fev=sst[1:228:12]
sst_mar=sst[2:228:12]
sst_abr=sst[3:228:12]
sst_mai=sst[4:228:12]
sst_jun=sst[5:228:12]
sst_jul=sst[6:228:12]
sst_ago=sst[7:228:12]
sst_set=sst[8:228:12]
sst_out=sst[9:228:12]
sst_nov=sst[10:228:12]
sst_dez=sst[11:228:12]



'''media total dos meses'''
sst_jan_med = np.mean(sst_jan,axis=0)
sst_fev_med = np.mean(sst_fev,axis=0)
sst_mar_med = np.mean(sst_mar,axis=0)
sst_abr_med = np.mean(sst_abr,axis=0)
sst_mai_med = np.mean(sst_mai,axis=0)
sst_jun_med = np.mean(sst_jun,axis=0)
sst_jul_med = np.mean(sst_jul,axis=0)
sst_ago_med = np.mean(sst_ago,axis=0) 
sst_set_med = np.mean(sst_set,axis=0)
sst_out_med = np.mean(sst_out,axis=0)
sst_nov_med = np.mean(sst_nov,axis=0)
sst_dez_med = np.mean(sst_dez,axis=0)


'''identificando os maiores e menores valores e onde estão'''
round(np.nanmax(sst),2)
max_sst = np.where(sst==np.nanmax(sst))
round(np.nanmin(sst),2)
min_sst=np.where(sst==np.nanmin(sst))

'''plotando as médias juntas'''
fig = plt.figure()

levels=np.linspace(24,29,300)
ax1 = fig.add_subplot(2,6,1, projection=ccrs.PlateCarree())
ax1.contourf(lon, lat,(sst_jan_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax1.title.set_text('Janeiro')
ax1.coastlines()
g = ax1.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False
g.xlabels_bottom = False 
ax1.add_feature(shape_feature, facecolor='none')


ax2 = fig.add_subplot(2,6,2, projection=ccrs.PlateCarree())
ax2.contourf(lon, lat,(sst_fev_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax2.title.set_text('Fevereiro')
ax2.coastlines()
g = ax2.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax2.add_feature(shape_feature, facecolor='none')

ax3 = fig.add_subplot(2,6,3, projection=ccrs.PlateCarree())
ax3.contourf(lon, lat,(sst_mar_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax3.title.set_text('Março')
ax3.coastlines()
g = ax3.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax3.add_feature(shape_feature, facecolor='none')

ax4 = fig.add_subplot(2,6,4, projection=ccrs.PlateCarree())
ax4.contourf(lon, lat,(sst_abr_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax4.title.set_text('Abril')
ax4.coastlines()
g = ax4.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax4.add_feature(shape_feature, facecolor='none')

ax5 = fig.add_subplot(2,6,5, projection=ccrs.PlateCarree())
ax5.contourf(lon, lat,(sst_mai_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax5.title.set_text('Maio')
ax5.coastlines()
g = ax5.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax5.add_feature(shape_feature, facecolor='none')

ax6 = fig.add_subplot(2,6,6, projection=ccrs.PlateCarree())
ax6.contourf(lon, lat,(sst_jun_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax6.title.set_text('Junho')
ax6.coastlines()
g = ax6.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax6.add_feature(shape_feature, facecolor='none')

ax7 = fig.add_subplot(2,6,7, projection=ccrs.PlateCarree())
ax7.contourf(lon, lat,(sst_jul_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax7.title.set_text('Julho')
ax7.coastlines()
g = ax7.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False
ax7.add_feature(shape_feature, facecolor='none')


ax8 = fig.add_subplot(2,6,8, projection=ccrs.PlateCarree())
ax8.contourf(lon, lat,(sst_ago_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax8.title.set_text('Agosto')
ax8.coastlines()
g = ax8.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False
g.ylabels_left=False
ax8.add_feature(shape_feature, facecolor='none')

ax9 = fig.add_subplot(2,6,9, projection=ccrs.PlateCarree())
ax9.contourf(lon, lat,(sst_set_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax9.title.set_text('Setembro')
ax9.coastlines()
g = ax9.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False
ax9.add_feature(shape_feature, facecolor='none')

ax10 = fig.add_subplot(2,6,10, projection=ccrs.PlateCarree())
ax10.contourf(lon, lat,(sst_out_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax10.title.set_text('Outubro')
ax10.coastlines()
g = ax10.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False
ax10.add_feature(shape_feature, facecolor='none')

ax11 = fig.add_subplot(2,6,11, projection=ccrs.PlateCarree())
ax11.contourf(lon, lat,(sst_nov_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax11.title.set_text('Novembro')
ax11.coastlines()
g = ax11.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False
ax11.add_feature(shape_feature, facecolor='none')

ax12 = fig.add_subplot(2,6,12, projection=ccrs.PlateCarree())
ax12.contourf(lon, lat,(sst_dez_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=24,vmax=29)
ax12.title.set_text('Dezembro')
ax12.coastlines()
g = ax12.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False
ax12.add_feature(shape_feature, facecolor='none')

fig.tight_layout()
fig.subplots_adjust(right=0.7)

sst_med = (sst_jan_med+sst_fev_med+sst_mar_med+sst_abr_med+sst_mai_med+sst_jun_med+sst_jul_med+sst_ago_med+sst_set_med+sst_out_med+sst_nov_med+sst_dez_med)/12
im = ax.imshow(sst_med,cmap='RdBu_r', vmin=24,vmax=29)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb=plt.colorbar(im, cax=cbar_ax)
cb.ax.set_xlabel('Temperatura(°C)',fontsize=12)
plt.suptitle('Temperatura da superfície do mar, para cada mês do ano ( C°)',fontsize=18)
plt.show()

###temperatura media
temp_media = round(np.nanmean(sst),2)
desv_temp=round(np.nanstd(sst),2)
##############################################################################
'Vendo dados de Radiação fotossinteticamente ativa'
##############################################################################
fn3 = 'C:/Users/jdebr/Documents/clorofila_bahia/par_ba.nc'
ds = nc.Dataset(fn3)

'''definindo as variaveis'''
for var in ds.variables.values():
    print(var)
par=ds['par'][:]
par[:,29:36,22:31]=np.nan
par[:,147,13]=np.nan
par[:,126:128,14]=np.nan 
par[:,114:116,17]=np.nan
par[:,28,24:26]=np.nan   
par[:,149,14]=np.nan  
par[:,154,8]=np.nan
for i in range(0,164):
    for j in range(0,102):
        if (valid[i,j]-valid[i,j])!=0:
            par[:,i,j]=np.nan
#par media
par_med=round(np.nanmean(par),2)
desv_par=round(np.nanstd(par),2)
par_max=np.where(par==np.nanmax(par))
par_min=np.where(par==np.nanmin(par))
'''separando os meses'''


par_jan= par[0:228:12]
par_fev=par[1:228:12]
par_mar=par[2:228:12]
par_abr=par[3:228:12]
par_mai=par[4:228:12]
par_jun=par[5:228:12]
par_jul=par[6:228:12]
par_ago=par[7:228:12]
par_set=par[8:228:12]
par_out=par[9:228:12]
par_nov=par[10:228:12]
par_dez=par[11:228:12]



'''media total dos meses'''
par_jan_med = np.mean(par_jan,axis=0)
par_fev_med = np.mean(par_fev,axis=0)
par_mar_med = np.mean(par_mar,axis=0)
par_abr_med = np.mean(par_abr,axis=0)
par_mai_med = np.mean(par_mai,axis=0)
par_jun_med = np.mean(par_jun,axis=0)
par_jul_med = np.mean(par_jul,axis=0)
par_ago_med = np.mean(par_ago,axis=0) 
par_set_med = np.mean(par_set,axis=0)
par_out_med = np.mean(par_out,axis=0)
par_nov_med = np.mean(par_nov,axis=0)
par_dez_med = np.mean(par_dez,axis=0)

max1 = ((par_jan_med.max()))
min1=((par_jan_med.min()))

max2 = ((par_fev_med.max()))
min2=((par_fev_med.min()))

max3 = ((par_mar_med.max()))
min3=((par_mar_med.min()))

max4 = ((par_abr_med.max()))
min4=((par_abr_med.min()))

max5 = ((par_mai_med.max()))
min5=((par_mai_med.min()))

max6 = ((par_jun_med.max()))
min6=((par_jun_med.min()))

max7 = ((par_jul_med.max()))
min7=((par_jul_med.min()))

max8 = ((par_ago_med.max()))
min8=((par_ago_med.min()))

max9 = ((par_set_med.max()))
min9=((par_set_med.min()))

max10 = ((par_out_med.max()))
min10=((par_out_med.min()))

max11 = ((par_nov_med.max()))
min11=((par_nov_med.min()))

max12 = ((par_dez_med.max()))
min12=((par_dez_med.min()))

max_total=round(max(max1,max2,max3,max4,max5,max6,max7,max8,max9,max10,max11,max12),3)
min_total=round(min(min1,min2,min3,min4,min5,min6,min7,min8,min9,min10,min11,min12),3)
fig = plt.figure()

levels=np.linspace(min_total,max_total,300)
ax1 = fig.add_subplot(2,6,1, projection=ccrs.PlateCarree())
ax1.contourf(lon, lat,(par_jan_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax1.title.set_text('Janeiro')
ax1.coastlines()
g = ax1.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False
g.xlabels_bottom = False 
ax1.add_feature(shape_feature, facecolor='none')


ax2 = fig.add_subplot(2,6,2, projection=ccrs.PlateCarree())
ax2.contourf(lon, lat,(par_fev_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax2.title.set_text('Fevereiro')
ax2.coastlines()
g = ax2.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax2.add_feature(shape_feature, facecolor='none')

ax3 = fig.add_subplot(2,6,3, projection=ccrs.PlateCarree())
ax3.contourf(lon, lat,(par_mar_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax3.title.set_text('Março')
ax3.coastlines()
g = ax3.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax3.add_feature(shape_feature, facecolor='none')

ax4 = fig.add_subplot(2,6,4, projection=ccrs.PlateCarree())
ax4.contourf(lon, lat,(par_abr_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax4.title.set_text('Abril')
ax4.coastlines()
g = ax4.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax4.add_feature(shape_feature, facecolor='none')

ax5 = fig.add_subplot(2,6,5, projection=ccrs.PlateCarree())
ax5.contourf(lon, lat,(par_mai_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax5.title.set_text('Maio')
ax5.coastlines()
g = ax5.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax5.add_feature(shape_feature, facecolor='none')

ax6 = fig.add_subplot(2,6,6, projection=ccrs.PlateCarree())
ax6.contourf(lon, lat,(par_jun_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax6.title.set_text('Junho')
ax6.coastlines()
g = ax6.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=False)
ax6.add_feature(shape_feature, facecolor='none')

ax7 = fig.add_subplot(2,6,7, projection=ccrs.PlateCarree())
ax7.contourf(lon, lat,(par_jul_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax7.title.set_text('Julho')
ax7.coastlines()
g = ax7.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False
ax7.add_feature(shape_feature, facecolor='none')


ax8 = fig.add_subplot(2,6,8, projection=ccrs.PlateCarree())
ax8.contourf(lon, lat,(par_ago_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax8.title.set_text('Agosto')
ax8.coastlines()
g = ax8.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.xlabels_top = False
g.ylabels_left=False
ax8.add_feature(shape_feature, facecolor='none')

ax9 = fig.add_subplot(2,6,9, projection=ccrs.PlateCarree())
ax9.contourf(lon, lat,(par_set_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax9.title.set_text('Setembro')
ax9.coastlines()
g = ax9.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False
ax9.add_feature(shape_feature, facecolor='none')

ax10 = fig.add_subplot(2,6,10, projection=ccrs.PlateCarree())
ax10.contourf(lon, lat,(par_out_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax10.title.set_text('Outubro')
ax10.coastlines()
g = ax10.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False
ax10.add_feature(shape_feature, facecolor='none')

ax11 = fig.add_subplot(2,6,11, projection=ccrs.PlateCarree())
ax11.contourf(lon, lat,(par_nov_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax11.title.set_text('Novembro')
ax11.coastlines()
g = ax11.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False
ax11.add_feature(shape_feature, facecolor='none')

ax12 = fig.add_subplot(2,6,12, projection=ccrs.PlateCarree())
ax12.contourf(lon, lat,(par_dez_med), 60,levels=levels,transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=min_total,vmax=max_total)
ax12.title.set_text('Dezembro')
ax12.coastlines()
g = ax12.gridlines(crs=ccrs.PlateCarree(), linestyle='-.', color='gray', draw_labels=True)
g.ylabels_right = False
g.ylabels_left = False
g.xlabels_top = False
g.ylabels_left=False
ax12.add_feature(shape_feature, facecolor='none')

fig.tight_layout()
fig.subplots_adjust(right=0.7)

par_med = (par_jan_med+par_fev_med+par_mar_med+par_abr_med+par_mai_med+par_jun_med+par_jul_med+par_ago_med+par_set_med+par_out_med+par_nov_med+par_dez_med)/12
im = ax.imshow(par_med,cmap='RdBu_r', vmin=min_total,vmax=max_total)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb=plt.colorbar(im, cax=cbar_ax)
cb.ax.set_xlabel('RFA (Einstein/m²/dia)',fontsize=10)
plt.suptitle('Radiação fotossinteticamente ativa(RFA) na superfície do mar média, para cada mês do ano ( Einstein/m²/dia)',fontsize=16)
plt.show()

##############################################################################
'''correlação entre clorofila, temperatura e par'''
##############################################################################
#Correlacao de Pearson entre sst(x) e chl(y)
#criando uma matriz vazia com o mesmo shape que as de sst e chl
corr_matrx= np.empty_like(sst[0],dtype=float)

#colocando os valores de correlacao de sst e chl na matriz
   
corr_matrx[corr_matrx==0]=np.nan


for i in range(0,164):
    for j in range(0,102):
        corr_matrx[i,j]=(pearsonr((sst[:,i,j]),(chl[:,i,j])))[0]
#Correlacao de Pearson entre par(x) e chl(y)
corr_matrx_par= np.zeros_like(sst[0],dtype=float)   
corr_matrx_par[corr_matrx_par==0]=np.nan
for i in range(0,164):
    for j in range(0,102):
        corr_matrx_par[i,j]=(pearsonr((par[:,i,j]),(chl[:,i,j])))[0]