import numpy as np
from pandas.core.frame import DataFrame
import seaborn as sns
import pandas as pd 
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import sklearn
from process import readData, plot_reg
from sklearn.decomposition import PCA


NUM_CLUSTER = 10

def calPCA(lis):

    pca = PCA(n_components = 10)

    X = np.array(lis, dtype=np.float64)
    pca.fit(X)
    print(pca.explained_variance_)
    pca.n_components = 3
    X_reduce = pca.fit_transform(X)
    print('the shape of X is: ', X_reduce.shape)

    return X_reduce



def runPCA():


    pass


def prep():

    conversation =readData('./data/conversation_comment_list', _header = 0)
    sorted_lis = conversation['sorted_list']

    liscnt = [len(i) for i in sorted_lis]
    print(np.median(liscnt))
    print()

    cnt = 0
    filterlis = []
    for lis in sorted_lis:
        if 'true' in lis:
            cnt += 1
            filterlis.append(lis)

    print(filterlis[0])
    # filterlis = [ i for i in sorted_lis if 'true' in i]

    positions = []
    print(len(filterlis))
    liscnt = []

    for lis in filterlis:

        thelis = lis[1:-1].split(",")
        for idx, val in enumerate(thelis):
            if val == 'true' or val == ' true':
                positions.append(idx)
                break
    print(np.max(positions))
    print(np.median(positions))
    print(np.mean(positions))

    print(np.max(liscnt))
    print(np.median(liscnt))
    print(np.mean(liscnt))
    

    plotpos = np.asarray(positions)
    # fig, ax = plt.subplot()
    sns.ecdfplot(data=positions)    
    plt.xlabel('postions of emoji appearance in one conversation list')
    plt.savefig('week4_5.png')
    plt.show()

    # print()
    # plt.his

def prep2():
    conversation = readData("./data/conver", _header = 0)



def e1():
    repoaids = readData("./data/dfuserswork", _header = 0)
    res = readData("./data/restypenew", _header = 0)
    res = res.fillna(0)

    res = res.groupby('rid').agg({'commentemojitype':'sum', 'premojitype':'sum', 'issueemojitype':'sum'})

        
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
    emojis = dfplot['totalemoji']
    containposts=dfplot['containposts']
    print('the median user events of repo: ', np.median(events))
    # print('the median user emojis of one repo', np.median(emojis))
    print(dfplot.head())
    for idx in range(0, len(containposts)):
        y.append(events[idx])
        x.append(containposts[idx])

    x = events
    y = emojis
    plot_reg(x, y)

import numpy as np
def getfirstLis(lis):
    
    res = []
    if lis == 0:
        return res
    flag = False
    thelis = lis[1:-1].split(",")
    for i in thelis:
        if i == "true":
            flag = True
    if flag is True:
        res.append("true")
    else:
        res.append("false")
    return res


def getLis(lis):
    thelis = []
    if lis == 0:
        return thelis
    flag = False
    thelis = lis[1:-1].split(",")
    
    return thelis

def getTwo(row):
    lis = []
    for i in row['issuelis']:
        lis.append(i)
    for i in row['commentlis']:
        lis.append(i)
    return lis
    

def filterFunc(row):
    if len(row['commentlis']) < 1:
        return False
    return True





    
def filterEmoji(row):
    Flag = False
    for i in row['commentlis']:
        if i == "true":
            Flag = True
            break
    for i in row['issuelis']:
        if i == "true":
            Flag = True
    
    return Flag


def clusterRepo():
    df = readData("./data/conver", _header = 0)
    df = df.fillna(0)
    df['issuelis'] = df['comment_list'].apply(getfirstLis)
    df['commentlis'] = df['comment_lis'].apply(getLis)

    max = -1
    for i in df['commentlis']:
        if len(i) > max:
            max = len(i)
    print('the max val is max: ', max)

    print(df.head())


    filterlabel = df.apply(filterFunc, axis = 1)
    df = df[filterlabel] 
    filteremoji = df.apply(filterEmoji, axis = 1)
    df = df[filteremoji]


    lengths = [len(i) for i in df['commentlis']]
    
    print(np.average(lengths))
    print(np.max(lengths))
    print(np.median(lengths))



    df['sortlis'] = df.apply(getTwo, axis=1)
    # def readFunc(df):
 
    #     return df[1:-1].split(",")

    # df = pd.read_csv("sortlis.csv")
    # print(df.head())
    # df['sortlis'] = df['sortlis'].apply(readFunc)
    # df.to_csv("sortlis.csv")
    # def sortFirstposvec(vec):
    #     lis = []
    #     flag = False
    #     for i in vec:
    #         if i == "true":
    #             flag = True




    print(df.head())
    
    dealis = df['sortlis']
    positions = []
    length = []
    for lis in dealis:
        length.append(len(lis))
        for idx, val in enumerate(lis):
            if val == 'true' or val == ' true':
                positions.append(idx)
                break

    print(positions)
    print(np.max(length))
    print(np.mean(length))
    print(np.median(length))  
    print(np.max(positions))
    print(np.mean(positions))
    print(np.median(positions))

    riddf = readData("data/issueridmap", _header=0)
    riddf = riddf[pd.notnull(riddf.commentissueid)]

    print(riddf.head())
    print(df.head())
    df.reset_index(inplace=True)
    print(riddf.keys())
    print(df.keys())
    
    
    riddf.reset_index(inplace=True)
    df = df.merge(riddf, on='commentissueid', how='inner')
    lengths = [len(i) for i in df['sortlis']]
    print('the max current is: ', np.max(lengths))
    maxval = int(np.max(lengths))
    print(df.head())
    dfold = df

    def groupbyFunc(df):
        vec = np.zeros(maxval)
        cnt = 0
        length = 0
        for idx, val in enumerate(df):
            length+= len(val)
            cnt += 1
            print(val)
            for _id, v in enumerate(val):
                if v == 'true' or v ==' true':
                    vec[_id] = 1
        return vec/cnt
    def groupbylength(df):
        vec = np.zeros(maxval)
        cnt = 0
        length = 0
        for idx, val in enumerate(df):
            length+= len(val)
            cnt += 1
            print(val)
            for _id, v in enumerate(val):
                if v == 'true' or v ==' true':
                    vec[_id] = 1
        return length/cnt
 
    df = df.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')
    df2 = dfold.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')
    df2['avglength'] = df2['sortlis'].apply(groupbylength)
    df2 = df2[['rid', 'avglength']]
    
    df['sortlis'] = df['sortlis'].apply(groupbyFunc)
    df = df.merge(df2, on='rid',how='inner')
    print(df.head())

    # df = df[df['avglength'] > 4]
    print('the remaining rows are: ', df.count())
    print(df.head())
    df.to_pickle('conversation.pkl')



    
    

    

    
from ast import literal_eval
def analyzeCluster():

    df = pd.read_pickle('conversation.pkl')
    # df.set_index(['rid', 'vec'])


    # df['totalsum'] = df['sortlis'].apply(sumVec)

    df = df[df['avglength']>]

    print('the bigger than 4 cnt: ', df.count())
    


    dfevent = readData('./data/repoavguser_events_fix')
    dfevent.columns = ['rid', 'events']
    event1 = dfevent['events']

    aids = readData('./data/dfusers', _header = 0)
    print('the initial mean', np.mean(event1))
    df = df.merge(dfevent, on='rid', how='inner').merge(aids,on='rid',how='inner')
    df['avgevents'] = df['events']/df['repouserscnt']
    print(df.head())

    X = df['sortlis']

    for i in X:
        arr = np.asarray(i)
        if len(i)!=924:
            print('error')
            exit(1)

    X = np.vstack(X)
    print(X.shape)
    events = df['events']
    X_new = calPCA(X)
    
    
    from sklearn import cluster

    
    kmeans = cluster.KMeans(n_clusters = NUM_CLUSTER)
    kmeans.fit(X_new)
    labels = kmeans.labels_

    keys = [i for i in range(0, NUM_CLUSTER)]
    # dictevents = dict.fromkeys(keys)
    dictevents = dict()
    events = df['avgevents']
    print('the mean of events', np.median(events))
    for idx, lab in enumerate(labels):
        if lab not in dictevents:
            dictevents[lab] = list()
        dictevents[lab].append(events[idx])

    res = [np.mean(i) for i in dictevents.values()]
    print('finished')
    plt.plot(res)
    plt.xlabel('cluster')
    plt.ylabel('repo average events (then perform average again)')

    plt.show()



if __name__ == "__main__":

    analyzeCluster()