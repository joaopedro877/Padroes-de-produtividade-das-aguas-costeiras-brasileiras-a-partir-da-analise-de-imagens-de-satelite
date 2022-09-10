# -*- coding: utf-8 -*-

'''concatenacao dos dados'''
'''juntando os dados de sst em um unico arquivo em netcdf, adicionando uma dimensao de tempo'''
import xarray



''' os arquivos a serem concatenados devem estar na mesma pasta. Deve estar no formato abaixo para funcionar'''
''' nao pode ter acento nem til no endereco.Se tiver o python nao consegue achar a pasta ou arquivo'''
ds = xarray.open_mfdataset('C:/Users/jdebr/Documents/Oceanografia/Disciplinas/Semestre_2021.2/Introducao_a_geotecnologias/dados_level3/sea_level_anomaly_chl/*.nc',combine = 'nested', concat_dim="time")

'''dando um nome e salvadano'''
ds.to_netcdf('sstjan.nc')