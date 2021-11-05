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
from sklearn.cluster import KMeans




NUM_CLUSTER = 10
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic 
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
# Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
# Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        origDisp = km.inertia_
# Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
# Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)


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


def plotDataHisto(metrics, labels):
    cntdict = dict()
    figure, axis = plt.subplots(2, 5)
    for idx, val in enumerate(labels):
        if val not in cntdict:
            cntdict[val] = list()
        cntdict[val].append(metrics[idx])

    a = axis.ravel()
    for idx, ax in enumerate(a):
        ax.hist(cntdict[idx])
        title_ = "cluster"+str(idx)
        ax.set_title(title_)
        ax.set_xlabel("average length calculation")
        ax.set_ylabel("count")

    plt.tight_layout()
    plt.show()
        

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
 
    def calRepoAvg(df):
        vec = np.zeros(maxval)
        cnt = 0
        length = 0
        idxlist = []
        for idx, val in enumerate(df):
            length+= len(val)
            cnt += 1
            print(val)
            for _id, v in enumerate(val):
                if v == 'true' or v ==' true':
                    idxlist.append(_id)
                    break
        avgpos = np.average(idxlist)     
        return avgpos
            
    df = df.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')
    df2 = dfold.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')
    df3 = dfold.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')
    df2['avglength'] = df2['sortlis'].apply(groupbylength)
    df2 = df2[['rid', 'avglength']]
    df3['avgpos'] = df3['sortlis'].apply(calRepoAvg)
    df3 = df3[['rid', 'avgpos']]
    df['sortlis'] = df['sortlis'].apply(groupbyFunc)
    df = df.merge(df2, on='rid',how='inner')
    df = df.merge(df3, on='rid',how='inner')
    print(df.head())

    # df = df[df['avglength'] > 4]
    print('the remaining rows are: ', df.count())
    print(df.head())
    df.to_pickle('conversation_new.pkl')
    
from ast import literal_eval

def findOptimal(X):
    sumofsquareddis = []
    K = range(20, 10, 200)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        sumofsquareddis.append(km.inertia_)

    plt.plot(K, sumofsquareddis,'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')  
    plt.title('Elbow Method For Optimal k')
    plt.show()


def analyzeCluster():

    df = pd.read_pickle('conversation.pkl')
    # df.set_index(['rid', 'vec'])


    # df['totalsum'] = df['sortlis'].apply(sumVec)

    # df = df[df['avglength']>5]

    print('the bigger than 4 cnt: ', df.count())
    


    dfevent = readData('./data/repoavguser_events_fix')
    dfevent.columns = ['rid', 'events']
    event1 = dfevent['events']

    aids = readData('./data/dfusers', _header = 0)
    df = df.merge(dfevent, on='rid', how='inner').merge(aids,on='rid',how='inner')
    df['avgevents'] = df['events']/df['repouserscnt']
    print('the initial mean', np.mean(df['avgevents']))

    pos = df['avglength']
    print(df.head())

    X = df['sortlis']
    maxidxlis = []
    for i in X:
        arr = np.asarray(i)
        cnt = 0
        for j in arr:
            if j > 0:
                maxidxlis.append(cnt)
            cnt += 1


        if len(i)!=924:
            print('error')
            exit(1)

    print('the max value of the idx is ', max(maxidxlis))
    
    # plt.hist(maxidxlis)
    # plt.show()
    
    # X = np.vstack(X)
    # print(X.shape)
    # X_new = calPCA(X)
    X = np.vstack(X)
    # optimalK(X)
    # findOptimal(X)
    X_new = X
    
    from sklearn import cluster

    
    kmeans = cluster.KMeans(n_clusters = NUM_CLUSTER)
    kmeans.fit(X_new)
    labels = kmeans.labels_
    print(labels)

    keys = [i for i in range(0, NUM_CLUSTER)]
    # dictevents = dict.fromkeys(keys)
    dictevents = dict()
    # events = df['avgevents']

    # print('the mean of events', np.median(events))
    plotDataHisto(pos, labels=labels)
    # for idx, lab in enumerate(labels):
    #     if lab not in dictevents:
    #         dictevents[lab] = list()
    #     dictevents[lab].append(events[idx])

    # res = [np.mean(i) for i in dictevents.values()]
    

#Then get the frequency count of the non-negative labels
    counts = np.bincount(labels[labels>=0])
    actualcnts = counts*X_new.shape[0]

    print(counts)
    print(actualcnts)
    print('finished')
    # plt.plot(res)
    plt.xlabel('cluster')
    plt.ylabel('repo average events (then perform average again)')

    plt.show()



if __name__ == "__main__":
    # clusterRepo()
    analyzeCluster()