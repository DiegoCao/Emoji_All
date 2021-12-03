import numpy as np
from numpy.random import RandomState
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
import matplotlib.cm as cm
import wandb
from sklearn.preprocessing import normalize
from analyze_cluster import run_analysis


NUM_CLUSTER = 6
LENDIX = 10
SEED = 0
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


def plotDataHisto(metrics, labels):
    cntdict = dict()
    figure, axis = plt.subplots(2, 3)
    for idx, val in enumerate(labels):
        if val not in cntdict:
            cntdict[val] = list()
        cntdict[val].append(metrics[idx])

    a = axis.ravel()
    namelis = []
    for idx, ax in enumerate(a):
        ax.hist(cntdict[idx])
        title_ = "cluster"+ str(idx)
        namelis.append(title_)
        ax.set_title(title_)
        ax.set_xlabel("work user number")
        ax.set_ylabel("count")

    lis = []
    for key, val in cntdict.items():
        tmp = np.asarray(list(val))
        lis.append(tmp)
    
    lis = np.asarray(lis)
    lis = lis/np.sum(lis, axis = 1)
    x_labels = ['cluster'+str(i) for i in range(0, NUM_CLUSTER)]
    
    # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(matrix_values = lis, x_labels=x_labels, show_text=False)})   
    # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(scatt, x_labels=x_labels, show_text=False)})   

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


    
    # print(np.average(lengths))
    # print(np.max(lengths))
    # print(np.median(lengths))



    df['sortlis'] = df.apply(getTwo, axis=1)



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
    df.to_pickle('conversation_issue.pck')
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

    def countVecFunc(df):
        vec = np.zeros(maxval)
        for idx, val in enumerate(df):
            print(val)
            for _id, v in enumerate(val):
                if v == 'true' or v ==' true':
                    vec[_id] = 1
        return vec

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

    def getLen(df):
        return len(df)

    
 
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
    
    
            
    def maxPos(df):
        vec = np.zeros(maxval)
        cnt = 0
        length = 0
        max_id = 0
        for idx, val in enumerate(df):
            length+= len(val)
            cnt += 1
            print(val)
            for _id, v in enumerate(val):
                if v == 'true' or v ==' true':
                    if _id > max_id:
                        max_id = _id
        
        return max_id

    df['lislen'] = df['sortlis'].apply(getLen)
    df = df[df['lislen']==9]
    df = df.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')
    df2 = dfold.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')
    df3 = dfold.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')
    df4 = dfold.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')
    df5 = dfold.groupby('rid')['sortlis'].apply(list).reset_index(name='sortlis')

    df2['avglength'] = df2['sortlis'].apply(groupbylength)
    df2 = df2[['rid', 'avglength']]
    df3['avgpos'] = df3['sortlis'].apply(calRepoAvg)
    df3 = df3[['rid', 'avgpos']]
    df['sortlis'] = df['sortlis'].apply(groupbyFunc)
    df4['maxoneid'] = df4['sortlis'].apply(maxPos)
    df4= df4[['rid', 'maxoneid']] 
    df5['totalvec'] = df5['sortlis'].apply(countVecFunc)
    df5 = df5[['rid', 'totalvec']]


    
    df = df.merge(df2, on='rid',how='inner')
    df = df.merge(df3, on='rid',how='inner').merge(df4, on='rid', how = 'inner').merge(df5, how='inner')

    print(df.head())

    # df = df[df['avglength'] > 4]
    print('the remaining rows are: ', df.count())
    print(df.head())
    df.to_pickle('conversation_new_len9.pkl')
    
from ast import literal_eval

def SilHolette(x):
    from sklearn.metrics import silhouette_score

    sil = []
    kmax = 10
    K = range(2,10, 1)
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in K:
        kmeans = KMeans(n_clusters = k).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    plt.plot(K, sil,'rx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')  
    plt.title('Silhouette Method For Optimal k')
    plt.show()

def plotHisto(dic):
    print(dic.values())
    arr = np.vstack(list(dic.values()))
    print(arr.shape)

    sns.heatmap(arr)
    arr = normalize(arr, axis = 1, norm='l1')
    # arr = arr[/np.sum(arr, axis=1)
    
    y_labels = ["cluster" + str(i) for i in range(0, NUM_CLUSTER)]
    x_labels = ["pos" + str(i) for i in range(0, LENDIX)]
    
    wandb.log({'heatmap_with_text': wandb.plots.HeatMap(x_labels, y_labels, arr, show_text=True)})   
    plt.show()

 

def findOptimal(X):
    sumofsquareddis = []
    K = range(1, 20, 1)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        sumofsquareddis.append(km.inertia_)

    plt.plot(K, sumofsquareddis,'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')  
    plt.title('Elbow Method For Optimal k')
    plt.show()

def plotLabelmap(x, y, labels):

    import colorsys
    import numpy as np
    import matplotlib.pyplot as plt
    N = NUM_CLUSTER

    colors = cm.rainbow(np.linspace(0, 1, N))
    print(colors)
    print('the length of ')
    xdict = {}
    ydict = {}
    normy = {}

    for idx, lab in enumerate(labels):
        if lab not in xdict:
            xdict[lab] = list()
            ydict[lab]= list()
        xdict[lab].append(np.log(x[idx]))
        ydict[lab].append(y[idx])
    #     normval = [float(i)/sum(y[idx]) for i in ydict]
    #     normy[lab].append()
    for i in range(N):
        data = [[x,y] for (x, y) in zip(xdict[i], ydict[i])]
        table = wandb.Table(data=data, columns=["log events", "pos"])
        wandb.log({"scatterdiagram": wandb.plot.scatter(table, "x", "y", title="Log events vs Pos plot") })

    # for i in range(N):
    #     wandb.log("scatter",)
    # plt.scatter(xdict[i],ydict[i],color=colors[i])
    # plt.ylabel('average position')
    # plt.xlabel('average events')
    # plt.show()

import pickle


def analyzeCluster():

    df = pd.read_pickle('conversation_new_len10.pkl')
    # df = df.loc[df['maxoneid'] < 10]
    # df.set_index(['rid', 'vec'])
    # df['totalsum'] = df['sortlis'].apply(sumVec)
    # df = df[df['avglength']<5]
    print('the bigger than 5 cnt: ', df.count())
    
    dfevent = readData('./data/repoallposts',_header = 0)
    dfevent.columns = ['rid', 'events']

    aids = readData('./data/dfuserswork', _header = 0)
    df = df.merge(dfevent, on='rid', how='inner').merge(aids,on='rid',how='inner')
    df['avgevents'] = df['events']/df['repouserscnt']
    aids = df['repouserscnt']

    print('the mean value of df events', np.mean(df['avgevents']))
    avgpos = df['avgpos']
    lengths = df['avglength']
    print('the initial mean', np.mean(df['avgevents']))
    print(df.head())

    X = df['sortlis']
    def reduceDim(df, len_idx=LENDIX):
        return df[:len_idx]   

    print('start redus dims:')


    histlen = np.array(lengths)
    # density, bins, _ = plt.hist(histlen)
    bins_ = [i for i in range(0, 11)]
    plt.hist(histlen, bins=bins_)
    # count, _ = np.histogram(histlen, bins)
    # for x,y,num in zip(bins, density, count):
    #     if num != 0:
    #         plt.text(x, y+0.05, num, fontsize=10, rotation=-90) # x,y,str
    plt.xlim([0, 10])
    plt.xlabel('count')
    plt.show()


    df['sortlis'] = df['sortlis'].apply(reduceDim)
    print(df['sortlis'])
    print(df['totalvec'])

    maxidxlis = []
    for i in X:
        arr = np.asarray(i)
        cnt = 0
        for j in arr:
            if j > 0:
                maxidxlis.append(cnt)
            cnt += 1

    print('the max value of the idx is ', max(maxidxlis))
    # X = np.vstack(X)
    # print(X.shape)
    # X_new = calPCA(X)
    X = np.vstack(X)
    print('Vector shape: ', X.shape)
    print('start finding optimal K:\n')
    # optimalK(X)
    # findOptimal(X)
    # SilHolette(X)
    X_new = X
    from sklearn import cluster
    kmeans = cluster.KMeans(n_clusters = NUM_CLUSTER, random_state=SEED)
    kmeans.fit(X_new)
    labels = kmeans.labels_
    kname = "kmeans"+str(NUM_CLUSTER)+".pck"
    pickle.dump(labels, open(kname, 'wb'))

    rids = df['rid']
    pickle.dump(rids, open('kmeansrids.pck', 'wb'))
    print(labels)

    keys = [i for i in range(0, NUM_CLUSTER)]
    # dictevents = dict.fromkeys(keys)
    dictevents = dict()
    # events = df['avgevents']

    # print('the mean of events', np.median(events))
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
    events = df['avgevents']
    # plotDataHisto(aids, labels=labels)

    print('the mean of events', np.median(events))
    def getXY(Xs, Ys, labels_):
        xdict = {}
        ydict = {}
        for idx, lab in enumerate(labels_):
            if lab not in xdict:
                xdict[lab] = list()
            xdict[lab].append(Xs[idx])
        for idx, lab in enumerate(labels_):
            if lab not in ydict:
                ydict[lab] = list()
            ydict[lab].append(Ys[idx])
        x = [np.mean(i) for i in xdict.values()]
        y = [np.mean(i) for i in ydict.values()]

        return x, y
    # for idx, lab in enumerate(labels):
    #     if lab not in dictevents:
    #         dictevents[lab] = list()
    #     dictevents[lab].append(events[idx]) 

    # positions = dict()
    # for idx, lab in enumerate(labels):
    #     if lab not in positions:
    #         positions[lab] = list()
    #     positions[lab].append(avgpos[idx])

    # x, y = getXY(events,avgpos,labels)

    plotLabelmap(events, aids, labels)
    totalvec = df['totalvec']
    print('the labels dimension ', len(labels))
    heatdic = {}
    for idx, lab in enumerate(labels):
        vec = totalvec[idx][:LENDIX]
        if lab not in heatdic:
            heatdic[lab] = vec
        else:
            heatdic[lab] += vec

    plotHisto(heatdic)

    res = [np.mean(i) for i in dictevents.values()]
#Then get the frequency count of the non-negative labels
    # counts = np.bincount(labels[labels>=0])
    # actualcnts = counts*X_new.shape[0]

    # print(counts)
    # print(actualcnts)
    # print('finished')
    plt.xlabel('cluster event average')
    plt.ylabel('cluster first postion')

    # 
    # plt.xlabel('cluster')
    # plt.ylabel('repo average events (then perform average again)')
    plt.show()

def examineDistribution():
    df = pd.read_pickle('conversation_new.pkl')
    val = df.count()
    df = df.loc[df['maxoneid'] > 50]
    print('the res is: ',(val- df.count())/val)



if __name__ == "__main__":
    # clusterRepo()
    # examineDistribution()
    filename = "pos_verification_10_withoutissue"
  

    this_config = dict(num_cluster = NUM_CLUSTER, conversation_length = LENDIX, random_seed=SEED)

    wandb.init(project = "emoji", name = filename, config = this_config)
    # analyzeCluster()
    run_analysis(NUM_CLUSTER, LENDIX)
    