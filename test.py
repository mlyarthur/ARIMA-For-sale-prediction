from __future__ import print_function
import sys
import pandas as pd
import numpy as np
import itertools
import json
from scipy import  stats
import matplotlib as mlp
mlp.use("TKAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

#config parameters#
search_mode="recycle"#no_recycle
visible=False
p_min = 4
d_min = 1
q_min = 1
p_max = 7
d_max = 2
q_max = 2
####################
def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)

def read_data(filename):
    product = {}
    ids=[]
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            eachline = line.strip().split()
            id = eachline[0]
            ids.append(id)
            sale_per_day = [int(q) for q in eachline[1:]]
            product[str(id)] = sale_per_day
    return product,ids

def grid_search(data,search_mode):

    if search_mode=="recycle_vis":
    # Initialize a DataFrame to store the results,，以BIC准则
        results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                                   columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])

        for p, d, q in itertools.product(range(p_min, p_max + 1),
                                         range(d_min, d_max + 1),
                                         range(q_min, q_max + 1)):
            if p == 0 and d == 0 and q == 0:
                results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
                continue

            try:
                model = sm.tsa.ARIMA(data, order=(p, d, q),
                                     # enforce_stationarity=False,
                                     # enforce_invertibility=False,
                                     )
                results = model.fit()
                results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
            except:
                continue
        results_bic = results_bic[results_bic.columns].astype(float)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax = sns.heatmap(results_bic,
                         mask=results_bic.isnull(),
                         ax=ax,
                         annot=True,
                         fmt='.2f',
                         )
        ax.set_title('BIC')
        plt.show()
    elif search_mode=="recycle":
        results_param={}

        for p, d, q in itertools.product(range(p_min, p_max + 1),
                                         range(d_min, d_max + 1),
                                         range(q_min, q_max + 1)):
            Index = (p, d, q)
            if p==0 and q==0 and d==0:
                results_param[Index]=np.nan
                continue
            try:
                results = sm.tsa.ARIMA(data, order=(p, d, q)).fit()
            except:
                results_param[Index] = np.nan
                continue
            results_param[Index] = 0.6 * float(results.aic) + 0.4 * float(results.bic)

        ics=[]
        keys=[]
        for key in results_param.keys():
            if pd.isnull(results_param[key]):
                continue
            keys.append(key)
            ics.append(results_param[key])

        (p,d,q)=keys[ics.index(min(ics))]
        params=(p,d,q)
        return params[0],params[1],params[2]
    elif search_mode=="error":
        results_param={}

        for p, d, q in itertools.product(range(0, 4+ 1),
                                         range(1, 3),
                                         range(3, 7 + 1)):
            Index = (p, d, q)
            if p==0 and q==0 and d==0:
                results_param[Index]=np.nan
                continue
            try:
                results = sm.tsa.ARIMA(data, order=(p, d, q)).fit()
            except:
                results_param[Index] = np.nan
                continue
            results_param[Index] = 0.6 * float(results.aic) + 0.4 * float(results.bic)

        ics=[]
        keys=[]
        for key in results_param.keys():
            if pd.isnull(results_param[key]):
                continue
            keys.append(key)
            ics.append(results_param[key])

        (p,d,q)=keys[ics.index(min(ics))]
        params=(p,d,q)
        return params[0],params[1],params[2]

def armia(id,product,predicted):
    flag=True#whethe the model fail to find the params
    #1.data prepocessing
    sale_per_product=product[str(id)]
    data=np.array(sale_per_product,dtype=np.float)
    src=data
    data=pd.Series(data)
    data.index=pd.Index(np.arange(118))
    data.plot(figsize=(12,8))
    plt.title("product_"+str(id))
    if visible:
        plt.show()
    #2.时间序列的差分d
    fig=plt.figure(figsize=(12,8))
    ax1=fig.add_subplot(111)
    diff=data.diff(1)
    diff.plot(ax=ax1)
    plt.title("diff"+str(id))
    if visible:
        plt.show()
    #3.find the proper p and q
    #3.1 model selection
    p_,d_,q_=grid_search(data,search_mode)
    print((p_,d_,q_))
    info = []
    info.append(str(id))
    with open("log.txt", 'w+') as f:
        f.writelines(info)
        #f.writelines(sale_per_product)
    try:
        arma_mod=sm.tsa.ARMA(data,order=(p_,d_,q_)).fit()
    except:
        p_, d_, q_ = grid_search(data, "error")
        arma_mod = sm.tsa.ARMA(data, order=(p_, d_, q_)).fit()
        flag=False

    #3.2 check the res
    resid=arma_mod.resid
    #print(sm.stats.durbin_watson(resid.values))

    #3.3 if normal distribution
    #print(stats.normaltest(resid))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)
    # plt.show()

    # 3.5残差序列检验
    r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    rdata = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(rdata, columns=['lag', "AC", "Q", "Prob(>Q)"])
    #print(table.set_index('lag'))

    predict_dta = arma_mod.predict(117,144, dynamic=True)
    predicted[str(id)]=np.array(predict_dta)
    print(predict_dta)
    print((p_, d_, q_))

    plt.subplot(111)
    all=np.concatenate((src,np.array(predict_dta).astype(int)))
    plt.plot(np.arange(all.size),all)
    plt.title("whole sale of the product")
    if visible:
        plt.show()
    return flag
    '''''''''
    # 3.prediction
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = data.ix[0:].plot(ax=ax)

    fig = arma_mod.plot_predict(117, 144, dynamic=True, ax=ax, plot_insample=False)
    plt.legend([ax,fig],["previous sale","predicted sale"],loc='upper right')
    plt.title('whole sale of the product')
    #plt.show()
    '''''''''
def output2file(predicted,ids,save_path):
    first=True
    for id in ids:
        sale=predicted[str(id)]
        line=np.insert(sale,0,id,axis=0)[None,:]
        #line=np.concatenate((np.array(id),sale))
        if first:
            output=line
            first=False
            continue
        np.append(output,line,axis=0)
    overall=np.sum(output,axis=0)
    output=np.insert(output,0,overall,axis=0)
    output[0][0]=0
    np.savetxt(save_path,output,fmt='%d',delimiter=' ')

def error(error_id_file,predicted_json):
    pass

def main():
    filename="/home/maliyuan/myprojects/ARIMA/product_distribution_training_set.txt"
    predicted={}
    product,ids=read_data(filename)
    '''''''''
    id=ids[0]
    armia(id, product,predicted)
    '''''''''

    error_id=[]
    for id in ids:
        try:
            f = open('result.txt', 'a+')
            num=id
            if not armia(num,product,predicted):
                error_id.append(id)
            print("day:"+str(id))
            list=[str(int(i))+' ' for i in predicted[str(num)]]
            list.append('\n')
            f.writelines(list)
            f.close()
        except:
            info=['error  happening']
            info.append("id:"+str(id))
            error_id.append(id)
            with open("error_info.txt",'a+') as f:
                f.writelines(info)
            continue

    #np.savetxt('error_id.txt',np.array(error_id),fmt='%d',delimiter=' ')

    output2file(predicted,ids,'./output.txt')
    save_dict('./sale.json', predicted)
    print("end")



if __name__ == '__main__':
    main()
    #grid_search()


