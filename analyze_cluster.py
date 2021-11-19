"""
    This file used to quantify the other parts of the project
"""
import enum
import pickle
from process import readData, plot_reg
import pandas as pd
import wandb
import numpy as np
from sklearn.preprocessing import normalize

def constructCntarray(rid_pos, LENDIX):
    lis_arr = []
    for _, val in rid_pos.items():
        each_dict = dict().fromkeys([i for i in range(0, LENDIX)], 0)
        for _id in val:
            each_dict[_id] += 1
        dlis = list(each_dict.values())
        
        lis_arr.append(np.asarray(dlis))

    lis_arr = np.asarray(lis_arr)
    return lis_arr

def run_analysis(num_cluster, LENDIX):
    """
        Verify whether this works for some only or for all?
    """
    NUM_CLUSTER=num_cluster
    kname ="kmeans"+str(NUM_CLUSTER)+".pck"
    labels = pickle.load(open(kname, 'rb'))
    rids = pickle.load(open('kmeansrids.pck', 'rb'))

    labeldict= dict()
    dictcnt = dict().fromkeys([i for i in range(0, NUM_CLUSTER)],0)
    for label, rid in zip(labels, rids):
        labeldict[rid] = label
        dictcnt[label] += 1
    
    for k, v in dictcnt.items():
        print('k v pair: ', k, v)


    df = pd.read_pickle("conversation_issue.pck")
    print(df.head())
    def calPos(a):
        idxlis = []
        for idx, item in enumerate(a):
            if item == "true" or item == " true":
                idxlis.append(idx)

        return idxlis

    df['conv_pos'] = df['sortlis'].apply(calPos)
    # rid_pos = dict(.fromkeys([i for i in range(0, NUM_CLUSTER)],list())) !!1!!!!!!!WHY ISSUE???
    rid_pos = dict()
    length_distri = dict()
    
    print(df.head())
    for pos, rid in zip(df['conv_pos'], df['rid']):
        if rid in labeldict:
            lid = labeldict[rid] # The label id for repo id
            # the rid position append idx
            for i in pos:
                if lid not in rid_pos:
                    rid_pos[lid] = list()
                rid_pos[lid].append(i)
            if lid not in length_distri:
                length_distri[lid] = list()
            length_distri[lid].append(len(pos))
    



    for k, v in rid_pos.items():
        print('k len(v) pair: ', k, len(v))


    histval = [np.average(val) for _, val in rid_pos.items()]
    data = [[s] for s in histval]

    lis_arr = constructCntarray(rid_pos, LENDIX)
    print('the shape is ', lis_arr.shape)
    lis_arr = normalize(lis_arr, axis = 1,norm='l1')


    y_labels = ["cluster" + str(i) for i in range(0, NUM_CLUSTER)]
    x_labels = ["pos" + str(i) for i in range(0, LENDIX)]


    
    wandb.log({'heatmap_for_all_emoji_appear_pos': wandb.plots.HeatMap(x_labels, y_labels, lis_arr, show_text=True)}) 

    
    y_labels = ["cluster" + str(i) for i in range(0, NUM_CLUSTER)]
    x_labels = ["len" + str(i) for i in range(0, LENDIX)]
    x_labels[-1] = "len>=10"  

    lis_arr = constructCntarray(length_distri, LENDIX + 1)
    lis_arr = normalize(lis_arr, axis = 1,norm='l1')
    wandb.log({'heatmap_for_conv_length': wandb.plots.HeatMap(x_labels, y_labels, lis_arr, show_text=True)}) 
    table = wandb.Table(data=data, columns=['allemojiposition'])
    wandb.log({'all_position_histogram':wandb.plot.histogram(table, histval)})
