import numpy as np
import seaborn as sns
import pandas as pd 
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import sklearn
from process import readData, plot_reg

def calPos(lis):

    for vec in lis:
        if len(vec) > 10:
            
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

def PCA():
    pass



    
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
    df = df.merge(riddf, on='commentissueid', how='outer')

    def groupbyFunc(df):
        vec = np.zeros(923)
        cnt = 0
        for idx, val in enumerate(df['sortlis']):
            cnt += 1
            for _id, v in enumerate(val):
                if v == 'true' or v ==' true':
                    vec[_id] = 1

        return vec/cnt
        pass
        
    df = df.groupby('rid').apply(groupbyFunc)
    print(df.head())
    df.to_csv("processed_cluster.csv")    
    
    df = df['rid']


    
    

    

    

def analyzeCluster():


    
    pass


if __name__ == "__main__":

    clusterRepo()