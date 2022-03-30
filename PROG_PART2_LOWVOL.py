#%%impotando as bibliotecas
import numpy as np
import pandas as pd
from datetime import date
import quantstats as qs
import math as mt
import matplotlib.pyplot as plt
from scipy import stats
#%% Limpando as bases

def limp_base(nome_base,legenda1='',legenda2='',legenda3='',legenda4='',legenda5=''):
    
    base=pd.read_csv(nome_base, delimiter=';')
    base.columns=base.columns.str.replace(legenda1,'')
    base.columns=base.columns.str.replace(legenda2,'')
    base.columns=base.columns.str.replace(legenda3,'')
    base.columns=base.columns.str.replace(legenda4,'')
    base.columns=base.columns.str.replace(legenda5,'')
    base.columns=base.columns.str.replace('*','')
    base.columns=base.columns.str.strip()
    base=base.replace('-',np.nan)
    base.Data=pd.to_datetime(base.Data,dayfirst=True)
    base=pd.melt(base, id_vars='Data')
    base.value=base.value.str.replace('.','')
    base.value=base.value.str.replace(',','.')
    base.value=pd.to_numeric(base.value)
    base.value=base.value.replace(0,np.nan)
    base=base.pivot_table(index="Data",columns='variable',values='value')
    base=base.reset_index()
    
    return base

ineg=limp_base('ineg_novo.csv')
close=limp_base('cotação.csv')

#%% Modelagem

def Factor(data_inicial,rebalanceamento,base,lookback,pos_ativo=(0),qt_ativos=(-1),universo=[0],ordem=True):
        
        data_analise=data_inicial-pd.DateOffset(months=lookback)
        fator=base[(base.Data>data_analise)&(base.Data<=data_inicial)]
        fator=fator.dropna(axis=0,how='all')
        if universo==[0]:
            fator=fator.iloc[-1]
        if not universo==[0]:
            fator=fator.loc[:,universo]
            fator=fator.pct_change()
            fator=fator.std()*252**0.5
        fator=fator[1:]
        fator.name='value'
        fator=fator.sort_values(ascending=ordem)
        fator=pd.DataFrame(fator[pos_ativo:qt_ativos])
        fatoruni=list(fator.index)
        
        return fator, fatoruni
    
    
def backtesting(data_inicial,rebalanceamento,base,universo,longshort=1):
        

        peso=1/len(universo)
        backtest=base[(base.Data>data_inicial)&(base.Data<rebalanceamento)]
        backtest=backtest.set_index('Data')
        backtest=backtest.loc[:,universo]
        backtest=backtest.pct_change()
        backtest=backtest.replace(np.nan,0)
        backtest=backtest*(longshort)
        backtest=backtest.add(1)
        backtest[0:1]=backtest[0:1]*peso
        backtest=backtest.cumprod()
        backtest['retorno']=backtest.sum(axis=1)        
        backtest.retorno=backtest.retorno.pct_change()
        backtest_ret=pd.DataFrame(backtest.retorno[1:])
        
        # Perf Atribution
        pcr=backtest.loc[:,backtest.columns!='retorno']
        sp=backtest.retorno
        sp=sp.add(1).cumprod()
        sp=sp.replace(np.nan,1)
        peso_corrido=pcr.div(sp,axis=0)
        ret_at=peso_corrido*pcr.pct_change()
        ret_at=ret_at.add(1).cumprod().add(-1)
        ret_at=ret_at*100
        
        return backtest_ret,ret_at
    
#%%backtest

sens=pd.DataFrame({'Sharpe':[]})
atribution=pd.DataFrame()
lb_lowvol=12
rebal=1
data_inicial=pd.Timestamp(date(2006,1,1))
data_final=pd.Timestamp(date(2021,1,1))
retornos=pd.DataFrame()
retornos_D1=pd.DataFrame()
retornos_D2=pd.DataFrame()
retornos_D3=pd.DataFrame()
retornos_bench=pd.DataFrame()
        
while True:
        rebalanceamento=data_inicial+pd.DateOffset(months=rebal)
        if rebalanceamento<data_final+pd.DateOffset(months=rebal):
            
            ibx_values,ibx_list=Factor(data_inicial,rebalanceamento,base=ineg,lookback=1,qt_ativos=100,ordem=False)
            D1,D1_list=Factor(data_inicial,rebalanceamento,base=close,lookback=lb_lowvol,universo=ibx_list,qt_ativos=int(0.1*len(list(ibx_list))),ordem=True)
            D2,D2_list=Factor(data_inicial,rebalanceamento,base=close,lookback=lb_lowvol,universo=ibx_list,qt_ativos=int(0.2*len(list(ibx_list))),ordem=True)
            D3,D3_list=Factor(data_inicial,rebalanceamento,base=close,lookback=lb_lowvol,universo=ibx_list,qt_ativos=int(0.3*len(list(ibx_list))),ordem=True)
            
            backtest_D1,perf_D1=backtesting(data_inicial,rebalanceamento,base=close,universo=D1_list,longshort=1)
            retornos_D1=retornos_D1.append(backtest_D1)
            atribution=atribution.append(perf_D1)                   
            backtest_D2,perf_D2=backtesting(data_inicial,rebalanceamento,base=close,universo=D2_list,longshort=1)
            retornos_D2=retornos_D2.append(backtest_D2)
            atribution=atribution.append(perf_D2)
            backtest_D3,perf_D3=backtesting(data_inicial,rebalanceamento,base=close,universo=D3_list,longshort=1)
            retornos_D3=retornos_D3.append(backtest_D3)
            atribution=atribution.append(perf_D3)
       
            
            backtest_bench,perf_bench=backtesting(data_inicial,rebalanceamento,base=close,universo=ibx_list,longshort=1)
            retornos_bench=retornos_bench.append(backtest_bench)
            
            data_inicial=data_inicial+pd.DateOffset(months=rebal) 
            
        else:break 

#salvando os indicadores dos decis do backtest    
         
anos = (retornos_D1.index[-1]-retornos_D1.index[0]).days/365

total1=retornos_D1.retorno.add(1).prod() - 1
total2=retornos_D2.retorno.add(1).prod() - 1
total3=retornos_D3.retorno.add(1).prod() - 1
total=retornos_bench.retorno.add(1).prod() - 1

sharpe1=((retornos_D1.retorno.mean())/(retornos_D1.retorno.std()))*252**0.5
sharpe2=((retornos_D2.retorno.mean())/(retornos_D2.retorno.std()))*252**0.5
sharpe3=((retornos_D3.retorno.mean())/(retornos_D3.retorno.std()))*252**0.5
sharpe=((retornos_bench.retorno.mean())/(retornos_bench.retorno.std()))*252**0.5

cagr1=abs(total1+1)**(1/anos)-1
cagr2=abs(total2+1)**(1/anos)-1
cagr3=abs(total3+1)**(1/anos)-1
cagr=abs(total+1)**(1/anos)-1

vol1=retornos_D1.retorno.std()*mt.sqrt(252)
vol2=retornos_D2.retorno.std()*mt.sqrt(252)
vol3=retornos_D3.retorno.std()*mt.sqrt(252)
vol=retornos_bench.retorno.std()*mt.sqrt(252)


retornos_D1=retornos_D1.dropna(axis=0,how='any')
retornos_D2=retornos_D2.dropna(axis=0,how='any')
retornos_D3=retornos_D3.dropna(axis=0,how='any')
retornos_bench=retornos_bench.dropna(axis=0,how='any')


linregress=stats.linregress(retornos_bench.retorno,retornos_D1.retorno)
alfa1=linregress.intercept*252
beta1=linregress.slope
T1=(linregress.intercept)/linregress.intercept_stderr
linregress=stats.linregress(retornos_bench.retorno,retornos_D2.retorno)
alfa2=linregress.intercept*252
beta2=linregress.slope
T2=(linregress.intercept)/linregress.intercept_stderr
linregress=stats.linregress(retornos_bench.retorno,retornos_D3.retorno)
alfa3=linregress.intercept*252
beta3=linregress.slope
T3=(linregress.intercept)/linregress.intercept_stderr


sens_reb=pd.DataFrame({'Cagr':cagr1,'Sharpe':sharpe1, 'Alfa':alfa1,'T':T1},index=[1])
sens=sens.append(sens_reb)
sens_reb=pd.DataFrame({'Cagr':cagr2,'Sharpe':sharpe2, 'Alfa':alfa2,'T':T2},index=[2])
sens=sens.append(sens_reb)
sens_reb=pd.DataFrame({'Cagr':cagr3,'Sharpe':sharpe3, 'Alfa':alfa3,'T':T3},index=[3])
sens=sens.append(sens_reb)
sens_reb=pd.DataFrame({'Cagr':cagr,'Sharpe':sharpe},index=[4])
sens=sens.append(sens_reb)

#%% backtest completo pela biblioteca quantstats
qs.reports.full(retornos_D1['retorno'],retornos_D2['retorno'])
#%%gráfico dos percentuais de 10, 20 e 30 ações de menores volatilidade
quote1=retornos_D1.add(1).cumprod().add(-1)
quote2=retornos_D2.add(1).cumprod().add(-1)
quote3=retornos_D3.add(1).cumprod().add(-1)
quote=retornos_bench.add(1).cumprod().add(-1)

plt.figure(figsize=(15,5))
plt.plot(quote['retorno'],label='Benchmark')
plt.plot(quote1['retorno'],label='Top 10%')
plt.plot(quote2['retorno'],label='Top20%')
plt.plot(quote3['retorno'],label='Top 30%')
plt.legend(loc='upper left')
plt.title('retorno acumulado - Universo de 100 ações')
