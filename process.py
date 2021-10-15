
import seaborn as sns
import pandas as pd 
import glob
import matplotlib.pyplot as plt
import os
import numpy as np

def readData(filepath, _header = None):
    csv_files = glob.glob(os.path.join(filepath, "*.csv"))
    lis = []
    for f in csv_files:
        df = pd.read_csv(f, header = _header)
        lis.append(df)

    frames = pd.concat(lis, axis = 0)
    # frames.columns = _columns
    print(frames.head())

    return frames

def calDis(x, bar):
    cnt = 0
    for i in x:
        if i > bar:
            cnt += 1
    return cnt/len(x)

def plot_reg(x, y, xname, yname, savename):

    # sns.regplot(x=x, y=y,data=tips, x_bins=20, x_estimator=np.mean, x_ci='ci', fit_reg=False, order=1)
    sns.regplot(x=x,y=y,x_estimator=np.mean, color='g', x_bins=20, fit_reg=False)
    # plt.ylabel('user average events')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(savename)
    plt.show()
    pass
    
def e1():
    dftypes=readData('./data/repoavguser_emoji_types', _columns=['rid', 'emojitype'])
    dfevent = readData('./data/repoavguser_events_fix', _columns=['rid', 'events'])
    res = dftypes.merge(dfevent, how='outer', on='rid')
    dfaids = readData('./data/repo_aids_cnt',_header=0, _columns=['rid', 'aid'])
    dfplot = dfplot.merge(repoaids, on='rid', how ='outer')
    res = res.merge(dfaids, how='outer', on='rid')
    print(res.head())
    emojitypes = res['emojitype']
    events = res['events']
    x = []
    y = []
    for idx in range(0, len(res['events'])):
        if events[idx] < 1000:
            x.append(events[idx])
            y.append(emojitypes[idx])
    plot_reg(x, y, x_bins=10,fit_reg=False)
    

def e2():
    print('started')
    filepath = "./data/repoavguser_emoji_cnts/"
    dfemoji = readData(filepath, _columns = ['rid', 'emojis'])
    filepath = "./data/repo_aids_cnt"
    dfaids = readData(filepath, _header=0, _columns=['rid', 'aids'])
    dfevent = readData('./data/repoavguser_events_fix', _columns=['rid', 'events'])
    res = dfemoji.merge(dfevent, how='outer', on='rid')
    print(res.head())

    emojis = res['emojis']
    events = res['events']
    # with sns.axes_style('dark'):
    #     sns.jointplot(emojis, events, kind='hex')
    # plt.show()
    print('the proportion of events larger than 50000', calDis(emojis, 47))
    x = []
    y = []
    print('the average user events of repo: ', np.median(events))
    print('the average user emojis of one repo', np.median(emojis))
    for idx in range(0, len(res['events'])):
        if events[idx] < 1000 and emojis[idx] < 100:
            x.append(events[idx])
            y.append(emojis[idx])


def e_aid():

    df1 = idmap.groupby('aid').agg(['count'])
    df1.columns = ['aidcnt', 'commentcnt', 'prcnt']  
    df1.reset_index(inplace=True)
    print(df1.describe())
    
    
    dfemoji = df.groupby('aid').agg(['mean'])

    
    print(dfemoji.head())
    dfemoji.columns = ["nouse1", "emojimean", "nouse2", "nouse3"]
    # print(dfemoji.describe()["emojimean"])
    dfemoji.reset_index(inplace=True)
    dfemoji = dfemoji[["aid", "emojimean"]]
    print(dfemoji.head())

    dfemoji = dfemoji.merge(idmap, how='outer', on='aid')
    print(dfemoji.head())

def e3():

    dfcomment = readData("./data/comment_cnt/", _header = 0)
    idmap = readData("./data/idmap/", _header = 0)
    df = dfcomment.merge(idmap, how = "outer", on = "commentid") 
   

    prdf = readData("./data/pr_cnt", _header = 0)
    issuedf = readData("./data/issue_cnt", _header = 0)



    # joindf = prdf.merge(dfcomment, how='outer', on='rid')
    # joindf = joindf.merge(issuedf, how='outer', on='rid')


    # # df1 = idmap[["rid", "commentid"]]
    # # print(df1.head())

    # df1 = idmap.groupby('rid').agg(['count'])
    # df1.columns = ['aidcnt','commentcnt', 'prcnt']  
    # df1.reset_index(inplace=True)
    # print(df1.describe())
    
    dfemoji = df.groupby('rid').agg(['mean'])

    
    print(dfemoji.head())
    dfemoji.columns = ["nouse1", "emojimean", "nouse2", "nouse3"]
    print(dfemoji.describe()["emojimean"])
    dfemoji.reset_index(inplace=True)
    dfemoji = dfemoji[["rid", "emojimean"]]
    print(dfemoji.head())

    # resdf = dfemoji.merge(df1, how='outer',on='rid')
    print(resdf.head())
    emojis = resdf["emojimean"]
    print(emojis)
    events = resdf["commentcnt"]
    resdf['avgevents'] = resdf['aidcnt']/resdf['commentcnt']
    print(resdf.head())
    events = resdf["avgevents"]    
    
    x = []
    y = []
    print('the average user events of repo: ', np.median(events))
    print('the average user emojis of one repo', np.mean(emojis))
    for idx in range(0, len(events)):
        if events[idx] < 100 and emojis[idx] < 100 and events[idx] > 0:
            x.append(events[idx])
            y.append(emojis[idx])
    
    plot_reg(x, y)


import numpy as np

def e4():
    repoaids = readData("./data/dfuserswork", _header = 0)
    res = readData("./data/restypenew", _header = 0)
    res = res.fillna(0)
    print(res.head())

    res = res.groupby('rid').agg({'commentemojitype':'sum', 'premojitype':'sum', 'issueemojitype':'sum'})

    print(res.head())
        
    dfevent = readData('./data/dfeventcnt', _header = 0)

    dfplot = res.merge(dfevent, on='rid', how='inner')
    dfplot = dfplot.merge(repoaids, on='rid', how ='inner')

    
    allsum = readData("./data/repoallposts", _header = 0)
    partsum = readData("./data/repofilterposts", _header = 0)
    allsum = allsum[allsum['allposts']>0]
    partsum = partsum[partsum['filterposts']>0]
    dfplot = dfplot.merge(allsum, on='rid', how='inner')
    dfplot = dfplot.merge(partsum, on='rid', how='inner')
    dfplot['containposts']=dfplot['filterposts']/dfplot['allposts']

    dfplot['totalemoji'] = dfplot['commentemojitype']+dfplot['premojitype']+dfplot['issueemojitype']
    dfplot = dfplot[dfplot['totalemoji'].notnull()]
    dfplot['avgemoji'] = (dfplot['totalemoji'])/dfplot['repouserscnt']
    print(dfplot.head())

    x = []
    y = []

    dfplot['avgevent'] = dfplot['count']/dfplot['repouserscnt']
    emojis = dfplot['avgemoji']
    events = dfplot['avgevent']
    # emojis = dfplot['totalemoji']
    dfp = dfplot[['avgemoji','rid']]
    dfp.to_csv("avgemoji1.csv")
    containposts=dfplot['containposts']
    print('the median user events of repo: ', np.median(events))
    # print('the median user emojis of one repo', np.median(emojis))
    print(dfplot.head())
    for idx in range(0, len(containposts)):
        y.append(events[idx])
        x.append(containposts[idx])

    x = events
    y = emojis
    plot_reg(x, y, 'user average events in one repo', 'user average emoji usage in one repo','week4_1new.png')

def e5():

    res = readData('./data/res', _header=0).fillna(0)
    print(res.head())
    dfold = readData('./data/res', _header=0)
    # .fillna(0)
    res['aidsum'] = res['commentemojicnt']+res['premojicnt']+res['issueemojicnt']
    print(np.median(res['aidsum']))

    
    cnt = 0
    for i in res['aidsum']:
        if i < 1:
            cnt += 1
        
    print('the cnt ', cnt)
    # res = res.groupby('aid').agg({'commentemojicnt':'mean', 'premojicnt':'mean', 'issueemojicnt':'mean'})
    # print(res.head())
    res = res.groupby('aid').agg({'aidsum':'mean'})
    dfold = dfold[['aid', 'rid']].drop_duplicates()
    df = dfold.merge(res, on='aid',how='inner')
    print(df.head())

    dfevent = readData('./data/dfeventcnt', _header = 0)
    print(df.head())
    df = df.groupby('rid').agg({'aidsum': 'mean'})
    df.reset_index(inplace=True)
    print(df.head())
    df = df[['rid', 'aidsum']]
    df = df.merge(dfevent, on='rid', how='inner')
    print(df.head())




    repoaids = readData("./data/dfusers", _header = 0)
    df = df.merge(repoaids, on='rid', how='inner')

    df.to_csv("avgemoji2.csv")
    y = df['aidsum']
    
    # y = df['repouserscnt']

    # emojitypeavg = df['aidsum']
    # y = df['count']/df['repouserscnt']

    # x = events
    x = df['repouserscnt']

    plot_reg(x, y)

    pass



from pathlib import Path
import pandas as pd
from pyarrow.parquet import ParquetDataset


if __name__ == "__main__":

    e4()
    # files = glob.glob("./data/2018_year_pid_v2.parquet/*.parquet")
    # data = [pd.read_parquet(f,engine='fastparquet') for f in files]
    # merged_data = pd.concat(data,ignore_index=True)
    # e4()
    # data_dir = Path('./data/2018_year_pid_v2.parquet/')
    # full_df = pd.concat(
    #     ParquetDataset(parquet_file).read().to_pandas()
    #     for parquet_file in data_dir.glob('*.parquet')
    # ) 
