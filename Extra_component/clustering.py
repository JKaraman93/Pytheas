from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta, datetime, time
import numpy as np


def datetime_to_int(attr, attr_type):
    """
    Converts date/time type attribute to numeric,
    replacing its values with the difference from the first value,
    in terms of days for date type and in terms of seconds for time/timestamp type,
    in order to be suitable for clustering.
    """
    attr_sec = []
    anchor = attr[0]  # non-null value
    print(anchor)
    if attr_type == 'time':
        for a in attr:
            dif = datetime.combine(date.min, a) - datetime.combine(date.min, anchor)  # it requires also a date
            sec = dif.total_seconds()
            attr_sec.append(sec)
    elif attr_type == 'date':
        for a in attr:
            dif = a - anchor
            days = dif.total_seconds() / 86400  # 86400s = 1day
            attr_sec.append(days)
    else:  # timestamp
        for a in attr:
            dif = a - anchor
            sec = dif.total_seconds()
            attr_sec.append(sec)
    return attr_sec


def opt_numOfClusters(start, end, data, ind_cat_cols):
    """
    Calculates the cost function for a range of clusters number
    and plot the Elbow plot.
    """
    no_of_clusters = list(range(start, end + 1))
    cost_values = []
    for k in no_of_clusters:
        try:
            test_model = KPrototypes(n_clusters=k, init='Huang', random_state=42)
            test_model.fit_predict(data, categorical=ind_cat_cols)
            cost_values.append(test_model.cost_)
            print('Cluster initiation: {}'.format(k), cost_values[-1])
        except:
            break
    sns.set_theme(style="whitegrid", palette="bright", font_scale=1.2)
    x_range = range(2, k)
    plt.figure(figsize=(15, 7))
    ax = sns.lineplot(x=x_range, y=cost_values, marker="o", dashes=False)
    ax.set_title('Elbow curve', fontsize=18)
    ax.set_xlabel('No of clusters', fontsize=14)
    ax.set_ylabel('Cost', fontsize=14)
    ax.set(xlim=(start - 0.1, x_range[-1] + 0.1))
    plt.show()
    return cost_function_change(cost_values) + start - 1


def cost_function_change(cost):  # start:cluster=1,   end:cluster=len(cost)
    """
    According to cost function values per number of clusters,
    calculates the optimal one.
    """
    delta1 = [0, ]
    delta2 = [0, 0, ]
    strength = [0, ]
    for ind, c in enumerate(cost):
        if ind == 0:
            continue
        delta1.append(np.round(cost[ind - 1] - c, 3))
        if ind > 1:
            delta2.append(delta1[ind - 1] - delta1[ind])
            strength.append(delta2[ind] - delta1[ind])
    return strength.index(max(strength)) + 1


def prep_for_clustering(df, attr_names, inconsistent_cols):
    """
    Drop dataframe columns in case of :
    1. Inconsistent data type
    2. NA values existence
    3. High correlations with another one (numeric)
    4. High Cardinality  (categorical)
    """
    # Drop inconsistent cols #
    df = df.loc[:, ~df.columns.isin(inconsistent_cols)]

    # Drop rows or cols with NA values
    na_cols = list(df.columns[df.isnull().any()])
    # df.dropna(axis=1, inplace=True)
    df = df.loc[:, ~df.columns.isin(na_cols)]

    # Drop highly correlated numeric features #
    corr = df.corr()
    drop = []
    for j in range(len(corr.columns)):
        for i in range(j + 1, len(corr.index)):
            print(corr.iloc[i, j])
            if corr.iloc[i, j] > 0.9:
                if j in drop:
                    drop.append(i)
                else:
                    drop.append(j)
                print(drop)
    high_cor_cols = list(corr.columns[(drop)])
    df = df.loc[:, ~df.columns.isin(high_cor_cols)]

    # Drop high cardinality categorical cols #
    row = len(df.index)
    un_values = df[[col for col in df.columns if attr_names[col] == 'text']].nunique()
    feat_high_card = []
    card_thres = 0.30
    for s in un_values.index:
        if un_values[s] / row > card_thres:
            feat_high_card.append(s)
    df = df.loc[:, ~df.columns.isin(feat_high_card)]
    return df


def table_clustering(df_original, attr_names, inconsistent_cols):
    """
    Perform clustering using KPrototypes library for mixed data.
    Kmeans for numeric variables and Kmodes for categorical.
    Return cluster ids and summary table for each cluster.
    """
    df_cluster = df_original.copy(deep=True)

    # clean dataframe before clustering
    df_cluster = prep_for_clustering(df_cluster, attr_names, inconsistent_cols)

    cat_cols = []
    num_cols = []

    for col in df_cluster:
        if attr_names[col] == 'text':
            cat_cols.append(col)
        else:
            num_cols.append(col)
            if attr_names[col] in ['date', 'time', 'timestamp']:
                df_cluster[col] = datetime_to_int(list(df_cluster[col]), attr_names[col])

    df2 = df_cluster.copy(deep=True)
    df_num = df2[num_cols].to_numpy()

    # Scale numeric variables #
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_num)
    df2[num_cols] = data_scaled
    smart_array = df2.values

    ind_cat_cols = [df2.columns.get_loc(c) for c in cat_cols if c in df2]

    # Plotting elbow curve for k=2 to k=10
    n_clusters = opt_numOfClusters(2, 10, smart_array, ind_cat_cols)
    print('Number of clusters: ', n_clusters)

    model_ = KPrototypes(n_jobs=-1, n_clusters=n_clusters, init='Huang', random_state=42)
    model_.fit_predict(smart_array, categorical=ind_cat_cols)
    cluster_labels = model_.labels_

    # Clustering summary #
    # Calculate the mean for each numeric variable and count for categorical one #
    res = {col: 'mean' if col in num_cols else lambda x: x.value_counts().index[0] for col in df_cluster.columns}
    df_cluster['Cluster Label'] = cluster_labels
    df_cluster['Clusterid'] = [c + 1 for c in cluster_labels]
    df_cluster.rename(columns={'Cluster Label': 'Total'}, inplace=True)
    res['Total'] = 'count'
    df_cluster_results = df_cluster.groupby('Clusterid').agg(res).reset_index()
    for col in df_original:
        if attr_names[col] in ['time', 'date', 'timestamp']:
            print(col)
            for ic, c in enumerate(df_cluster_results[col]):
                if attr_names[col] == 'date':
                    df_cluster_results.loc[ic, col] = timedelta(days=c) + df_original.loc[0, col]
                else:
                    print(df_original.loc[0, col])
                    print(timedelta(seconds=c))
                    print(c)
                    df_cluster_results.loc[ic, col] = (
                            timedelta(seconds=np.round(c)) + datetime.combine(date.min, df_original.loc[0, col])).time()
    return ([df_cluster_results, cluster_labels])
