from sklearn.decomposition import PCA
import pandas as pd



if __name__ == "__main__":
    df = pd.read_csv('Pokemon.csv')
    # prepare data
    print(df.head())
    types = df['Type 1'].isin(['Grass', 'Fire', 'Water'])
    drop_cols = ['Type 1', 'Type 2', 'Generation', 'Legendary', '#']
    df = df[types].drop(columns = drop_cols)
    df.head()

    from sklearn.cluster import KMeans
    import numpy as np
    # k means
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['Attack', 'Defense']])
    # get centroids
    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids] 
    cen_y = [i[1] for i in centroids]
    ## add to df
    df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
    # define and map colors
    colors = ['#DF2020', '#81DF20', '#2095DF']
    df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})