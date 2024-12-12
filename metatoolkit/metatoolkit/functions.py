#!/usr/bin/env python
from itertools import combinations, count, permutations
from matplotlib_venn import venn3
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import distance
from scipy.stats import chi2_contingency, fisher_exact, zscore, levene, mannwhitneyu, shapiro, spearmanr
from skbio import stats
from skbio.diversity.alpha import pielou_e, shannon, chao1, faith_pd
from skbio.stats.composition import ancom, clr
from skbio.stats.composition import multiplicative_replacement as mul
from skbio.stats.distance import permanova
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_absolute_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
from sklearn_som.som import SOM
from statsmodels.stats.multitest import fdrcorrection
from upsetplot import UpSet, from_contents
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import pycircos
import seaborn as sns
import shap
import shutil
import skbio
import subprocess
import umap

# Prediction
def mlpclassifier(df, random_state=1):
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=random_state)
    X, y = df, df.index
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    performance = str(roc_auc_score(y_test, y_prob)) + '\n\n' + pd.DataFrame(confusion_matrix(y_test, y_pred)).to_string()
    print(performance)
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=y[0])
    aucrocdata = pd.DataFrame(pd.concat([pd.Series(fpr), pd.Series(tpr)],axis=1)).set_axis(['fpr','tpr'], axis=1)
    return model, performance, aucrocdata

def classifier100(df, shapval=False, shapinteract=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    aucrocs = []
    meanabsshaps = pd.DataFrame()
    mean_shap_interacts = pd.DataFrame(index=df.columns)
    random_state = 1
    for random_state in range(100):
        model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
        X, y = df, df.index
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)
        model.fit(X_train, y_train)
        # AUCROC
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        aucrocs.append(roc_auc_score(y_test, y_prob))
        if shapval:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)
            meanabsshaps[random_state] = pd.Series(
                np.abs(shap_values.values[:, :, 0]).mean(axis=0),
                index=X.columns
            )
        if shapinteract:
            inter_shaps_values = explainer.shap_interaction_values(X)
            sum_shap_interacts = pd.DataFrame(
                    data=np.abs(inter_shaps_values[0]).sum(0),
                    columns=df.columns,
                    index=df.columns)
    return aucrocs, meanabsshaps, mean_shap_interacts

def classifier(df, random_state=1):
    model = RandomForestClassifier(n_jobs=-1, random_state=random_state, oob_score=True)
    X, y = df, df.index
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    performance = roc_auc_score(y_test, y_prob)
    print(performance)
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=y[0])
    aucrocdata = pd.DataFrame(pd.concat([pd.Series(fpr), pd.Series(tpr)],axis=1)).set_axis(['fpr','tpr'], axis=1)
    return model, performance, aucrocdata

def regressor(df, random_state=1, **kwargs):
    model = RandomForestRegressor(n_jobs=-1, random_state=random_state, oob_score=True)
    X, y = df, df.index
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    model.fit(X_train, y_train, )
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    performance = mae
    print(performance)
    return model, performance

def networkpredict(df):
    i = df.columns[0]
    shaps = pd.DataFrame(index=df.columns)
    for i in df.columns:
        tdf = df.set_index(i)
        X,y = tdf,tdf.index
        #X_train, X_test, y_train, y_test =  train_test_split(X,y)
        model = RandomForestRegressor(random_state=1, n_jobs=-1)
        #model.fit(X_train, y_train)
        model.fit(X, y)
        shaps[i] = SHAP_reg(df, model)
        #shp = SHAP_reg(df, model)
        #shaps[i] = norm(shp.abs().to_frame().T).T[0].mul(shp)
    edges = to_edges(shaps, thresh=0.01)
    return edges

def binnetworkpredict(df):
    i = df.columns[3]
    shaps = pd.DataFrame(index=df.columns)
    aucroc = pd.Series(index=df.columns)
    for ii,i in enumerate(df.columns):
        print(i)
        tdf = df.set_index(i)
        tdf = upsample(tdf)
        X, y = tdf, tdf.index
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
        model = RandomForestClassifier(random_state=1, n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        aucroc[i] = roc_auc_score(y_test, y_prob)
        shaps[i] = SHAP_bin(tdf, model)
        print(shaps)
        print(aucroc)
    return aucroc, shaps

def predict(analysis, df, **kwargs):
    available={
        'regressor':regressor,
        'classifier':classifier,
        'classifier100':classifier100,
        }
    output = available.get(analysis)(df, **kwargs)
    return output

# Plotting
def setupplot(grid=False, logy=False, agg=False, figsize=(3,3), fontsize=6):
    #matplotlib.use('WebAgg')
    if agg: matplotlib.use('Agg')
    linewidth = 0.25
    matplotlib.rcParams['grid.color'] = 'lightgray'
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.size"] = fontsize
    matplotlib.rcParams["lines.linewidth"] = linewidth
    matplotlib.rcParams["figure.figsize"] = figsize
    matplotlib.rcParams["axes.linewidth"] = linewidth
    matplotlib.rcParams['axes.facecolor'] = 'none'
    matplotlib.rcParams['xtick.major.width'] = linewidth
    matplotlib.rcParams['ytick.major.width'] = linewidth
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['axes.axisbelow'] = True
    if grid: plt.grid(alpha=0.3)
    if logy: plt.yscale('log')


def clustermap(df, sig=None, figsize=(4,4), cluster=True, corr=False):
    pd.set_option("use_inf_as_na", True)
    df = df.fillna(0)
    kwargs = {}
    if not cluster:
        kwargs['row_cluster'] = False
        kwargs['col_cluster'] = False
    g = sns.clustermap(
        data=df,
        cmap="vlag",
        center=0,
        figsize=figsize,
        dendrogram_ratio=(0.25, 0.25),
        yticklabels=True,
        xticklabels=True,
        **kwargs,
    )
    if not sig is None:
        if cluster:
            for i, ix in enumerate(g.dendrogram_row.reordered_ind):
                for j, jx in enumerate(g.dendrogram_col.reordered_ind):
                    text = g.ax_heatmap.text(
                        j + 0.5,
                        i + 0.5,
                        "*" if sig.iloc[ix, jx] else "",
                        ha="center",
                        va="center",
                        color="black",
                    )
                    text.set_fontsize(8)
        else:
            for i, ix in enumerate(df.index):
                for j, jx in enumerate(df.columns):
                    text = g.ax_heatmap.text(
                        j + 0.5,
                        i + 0.5,
                        "*" if sig.iloc[i, j] else "",
                        ha="center",
                        va="center",
                        color="black",
                    )
                    text.set_fontsize(8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=40, ha="right")
    return g

def joseplot(df, x, y, color=None):
    power['class'] = power.index.str.replace('\ .*', '', regex=True) 
    sns.scatterplot(power, x=power.iloc[:,0], y=power.iloc[:,1], hue='class')
    currentAxis = plt.gca()
    currentAxis.add_patch(plt.Rectangle((-3,-3), 6, 6, ec="red", fc="none"))
    plt.grid()

def heatmap(df, sig=None, ax=None, center=0, **kwargs):
    pd.set_option("use_inf_as_na", True)
    if ax is None: fig, ax= plt.subplots()
    g = sns.heatmap(
        data=df,
        square=True,
        cmap="vlag",
        center=center,
        yticklabels=True,
        xticklabels=True,
        annot=sig.replace({True:'*',False:''}) if sig is not None else None,
        fmt='',
    )
    plt.setp(g.get_xticklabels(), rotation=40, ha="right")
    return g

def pointheatmap(df, ax=None, size_scale=300, **kwargs):
    df.columns.name='x'
    df.index.name='y'
    vals = df.unstack()
    vals.name='size'
    fvals = vals.to_frame().reset_index()
    x, y, size= fvals.x, fvals.y, fvals['size']
    if ax is None: fig, ax= plt.subplots()
    x_labels = x.unique()
    y_labels = y.unique()
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size * size_scale,
    )
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    plt.grid()
    return ax

def spindle(df, ax=None, palette=None, **kwargs):
    if palette is None: palette = pd.Series(sns.color_palette("hls", df.index.nunique()).as_hex(), index=df.index.unique())
    if ax is None: fig, ax= plt.subplots()
    x=df.columns[0]
    y=df.columns[1]
    centers = df.groupby(df.index).mean()
    centers.columns=['nPC1','nPC2']
    j = df.join(centers)
    j['colours'] = palette
    i = j.reset_index().index[0]
    for i in j.reset_index().index:
        ax.plot(
            j[[x,'nPC1']].iloc[i],
            j[[y,'nPC2']].iloc[i],
            linewidth = 0.5,
            color = j['colours'].iloc[i],
            zorder=1,
            alpha=0.3
        )
        ax.scatter(j[x].iloc[i], j[y].iloc[i], color = j['colours'].iloc[i], s=1)
    for i in centers.index:
        ax.text(centers.loc[i,'nPC1']+0.002,centers.loc[i,'nPC2']+0.002, s=i, zorder=3)
    ax.scatter(centers.nPC1, centers.nPC2, c='black', zorder=2, s=10, marker='+')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.spines[['right', 'top']].set_visible(False)
    return ax

def polar(df, **kwargs):
    palette = pd.Series(sns.color_palette("hls", df.index.nunique()).as_hex(), index=df.index.unique())
    data = df.T.copy().to_numpy()
    angles = np.linspace(0, 2*np.pi, len(df.columns), endpoint=False)
    data = np.concatenate((data, [data[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    categories = df.columns.to_list()
    loopcategories = df.columns.to_list()
    loopcategories.append(df.columns[0])
    alldf = pd.DataFrame(data=data, index = loopcategories, columns=df.index).T
    allangles = pd.Series(data=angles, index=loopcategories)
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for color in alldf.index.unique():
        plotdf = alldf.loc[alldf.index==color]
        ax.plot(allangles, plotdf.T, linewidth=1, color = palette[color])
    ax.set_xticks(allangles[:-1])
    ax.set_xticklabels(categories)
    ax.grid(True)
    return ax

def circos(datasets, thresh=0.5, **kwargs):
    '''
    datasets = {'msp':f.load('msp'),
                'eeg':f.load('eeg'),
                'pathways':f.load('pathways')} 
    thresh=0.5
    '''
    out, labs = [], []
    for data in datasets:
        tdf = datasets[data]
        out.append(tdf)
        tmp = tdf.columns.to_frame()
        tmp.index = tmp.shape[0] * [data]
        tmp = tmp.reset_index().reset_index().set_axis(['ID','datatype','var'], axis=1).set_index('var')
        labs.append(tmp)
    labels = pd.concat(labs)
    alldata = pd.concat(out, axis=1, join='inner')
    edges = to_edges(alldata.corr(), thresh=thresh)
    data = edges.join(labels[['ID','datatype']], how='inner').rename(columns={'datatype':'datatype1', 'ID':'index1'})
    data = data.reset_index().set_index('target').join(labels[['ID','datatype']], how='inner').rename_axis('target').rename(columns={'datatype':'datatype2', 'ID':'index2'}).reset_index()
    data = data.loc[data.datatype1 != data.datatype2]
    Gcircle = pycircos.Gcircle
    Garc = pycircos.Garc
    circle = Gcircle()
    for i, row in labels.groupby('datatype', sort=False):
        arc = Garc(arc_id=i,
                   size=row['ID'].max(),
                   interspace=30,
                   label_visible=True)
        circle.add_garc(arc)
    circle.set_garcs()
    for i, row in data.iterrows():
        circle.chord_plot(start_list=(row.datatype1, row.index1-1, row.index1, 500), end_list=(row.datatype2, row.index2-1, row.index2, 500),
        facecolor=plt.cm.get_cmap('coolwarm')(row.weight)) 
    return circle

def abund(df, order=None, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    tdf = df.copy()
    if tdf.columns.shape[0] > 20:
        tdf['others'] = tdf[tdf.mean().sort_values(ascending=False).iloc[21:].index].sum(axis=1)
    tdf = norm(tdf.loc[:, tdf.mean().sort_values(ascending=False).iloc[:20].index])
    if order:
        tdf = tdf.loc[:, tdf.mean().sort_values(ascending=True).index]
    tdf.plot(kind='bar', stacked=True, width=0.9, cmap='tab20', **kwargs)
    plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def bar(df, maximum=20, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    if df.columns.shape[0] > maximum:
        df['other'] = df[df.mean().sort_values().iloc[maximum+1:].index].sum(axis=1)
    df = df[df.median().sort_values(ascending=False).head(maximum).index]
    mdf = df.melt()
    kwargs['ax'] = sns.boxplot(data=mdf, x=mdf.columns[0], y='value', showfliers=False, boxprops=dict(alpha=.25))
    sns.stripplot(data=mdf, x=mdf.columns[0], y='value', size=2, color=".3", ax=kwargs['ax'])
    kwargs['ax'].set_xlabel(mdf.columns[0])
    kwargs['ax'].set_ylabel('Relative abundance')
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def hist(df, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    kwargs['column'] = 'sig' if not kwargs.get('column') else kwargs.get('column')
    kwargs['ax'] = sns.histplot(data=df[kwargs['column']])
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def regplot(df, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    x = kwargs.get('x')
    y = kwargs.get('y')
    sns.regplot(data=df, x=x, y=y, scatter_kws={"color": "black", 's':1}, line_kws={"color": "red"})
    return kwargs['ax']

def box(df, **kwargs):
    df = df.reset_index()
    if kwargs.get('x') is None: kwargs['x'] = df.columns[0]
    if kwargs.get('y') is None: kwargs['y'] = df.columns[1]
    if kwargs.get('ax') is None: kwargs['ax'] = plt.subplots()[1]
    sns.boxplot(data=df, showfliers=False, showcaps=False, linewidth=0.25, **kwargs)
    if kwargs.get('hue'): kwargs['dodge'] = True
    sns.stripplot(data=df, s=1, color="0.2", **kwargs)
    #plt.setp(kwargs['ax'].get_xticklabels(), rotation=40, ha="right")
    kwargs['ax'].spines[['right', 'top']].set_visible(False)
    return kwargs['ax']

def violin(df, **kwargs):
    df = df.reset_index()
    if kwargs.get('x') is None: kwargs['x'] = df.columns[0]
    if kwargs.get('y') is None: kwargs['y'] = df.columns[1]
    if kwargs.get('ax') is None: kwargs['ax'] = plt.subplots()[1]
    sns.violinplot(data=df, inner="point", density_norm="count", cut=0, **kwargs)
    if kwargs.get('hue'): kwargs['dodge'] = True
    #sns.stripplot(data=df, s=1, color="0.2", **kwargs)
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=40, ha="right")
    return kwargs['ax']

def multibox(df, sharey=True, logy=False):
    fig, ax = plt.subplots(nrows=1, ncols=len(df.columns), sharey=sharey, constrained_layout=True)
    i, j = 0, df.columns[0]
    for i, j in enumerate(df.columns):
        box(df=df[j].to_frame(), x=df[j].index, y=j, ax=ax[i])
        ax[i].set_ylabel("")
        ax[i].set_title(j)
    if logy: plt.yscale('log')
    return ax

def volcano(df, hue=None, change='Log2FC', sig='MWW_pval', fc=1, pval=0.05, annot=True, ax=None, size=None):
    if ax is None:
        fig, ax = plt.subplots()

    # Extracting the fold change and p-values
    lfc = df[change]
    pvals = df[sig]
    lpvals = -np.log10(pvals)

    # Set default color for all points
    colors = 'black'
    if hue:
        # Use a red color map
        unique_vals = df[hue].unique()
        color_palette = sns.color_palette("Reds", len(unique_vals))
        color_dict = dict(zip(unique_vals, color_palette))
        colors = df[hue].map(color_dict)

    # Set default size for all points or use the size column
    sizes = 20  # default size
    if size:
        sizes = df[size]

    # Scatter plot
    ax.scatter(lfc, lpvals, c=colors, s=sizes, alpha=0.5)

    # Adding threshold lines
    ax.axvline(0, color='gray', linestyle='--')
    ax.axhline(-np.log10(pval), color='red', linestyle='-')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(fc, color='red', linestyle='-')
    ax.axvline(-fc, color='red', linestyle='-')

    # Setting labels and limits
    ax.set_ylabel('-log10 p-value')
    ax.set_xlabel('log2 fold change')
    ax.set_ylim(ymin=-0.1)
    x_max = np.abs(lfc).max() * 1.1
    ax.set_xlim(xmin=-x_max, xmax=x_max)

    # Annotate significant points
    sig_species = (lfc.abs() > fc) & (pvals < pval)
    filter_df = df[sig_species]

    if annot:
        for idx, row in filter_df.iterrows():
            ax.text(row[change], -np.log10(row[sig]), s=idx, fontsize=6)

    return ax

def aucroc(df, ax=None, colour=None, **kwargs):
    AUC = auc(df.fpr, df.tpr)
    if ax is None: fig, ax= plt.subplots()
    if colour is None:
        ax.plot(df.fpr, df.tpr, label=r"AUCROC = %0.2f" % AUC )
    else:
        ax.plot(df.fpr, df.tpr, color=colour, label=r"AUCROC = %0.2f" % AUC )
    ax.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    return ax

def curve(df, mapping=None, ax=None, **kwargs):
    if ax is None: fig, ax= plt.subplots()
    df = df.apply(np.log1p).T
    df.loc['other'] = df[~df.index.isin(mapping.index)].sum()
    df = df.drop(df[~df.index.isin(mapping.index)].index).T
    grid = np.linspace(df.index.min(), df.index.max(), num=500)
    df1poly = df.apply(lambda x: np.polyfit(x.index, x.values, 3), axis=0).T
    df1polyfit = df1poly.apply(lambda x: np.poly1d(x)(grid), axis=1)
    griddf = pd.DataFrame( np.row_stack(df1polyfit), index=df1polyfit.index, columns=grid)
    griddf.clip(lower=0, inplace=True)
    if griddf.shape[0] > 20:
        griddf.loc["other"] = griddf.loc[
            griddf.T.sum().sort_values(ascending=False).iloc[21:].index
        ].sum()
    griddf = griddf.loc[griddf.T.sum().sort_values().tail(20).index].T
    griddf.sort_index(axis=1).plot.area(stacked=True, color=mapping.to_dict(), ax=ax)
    plt.tick_params(bottom = False)
    return ax

def dendrogram(df, **kwargs):
    df = df.unstack()
    Z = linkage(df, method='average')
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.title('Dendrogram')
    return ax

def upset(dfdict=None, **kwargs):
    intersections = from_contents(dfdict) 
    upset = UpSet(intersections, **kwargs)
    upset.plot()
    return upset

def networkplot(*args, group=None, **kwargs):
    G = nx.read_gpickle(f'../results/kwargs.get(subject).gpickle')
    if group:
        nx.set_node_attributes(G, group, "group")
        groups = set(nx.get_node_attributes(G, 'group').values())
        mapping = dict(zip(sorted(groups),count()))
        nodes = G.nodes()
        colors = [mapping[G.nodes[n]['group']] for n in nodes]
    pos = nx.spring_layout(G)
    #pos= nx.spring_layout(G, with_labels=True, node_size=50)
    ax = nx.draw_networkx_edges(G, pos, alpha=0.2)
    #nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)
    ax = nx.draw_networkx_nodes(G, pos, node_color=group, node_size=20, cmap=plt.cm.jet)
    #nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
    plt.colorbar(ax)
    return ax

def venn(*args, **kwargs):
    DF1 = pd.read_csv(f'../results/kwargs.get(df1).tsv', sep='\t', index_col=0)
    DF2 = pd.read_csv(f'../results/kwargs.get(df2).tsv', sep='\t', index_col=0)
    DF3 = pd.read_csv(f'../results/kwargs.get(df3).tsv', sep='\t', index_col=0)
    def combs(x): return [c for i in range(1, len(x)+1) for c in combinations(x,i)]
    comb = combs([DF1, DF2, DF3])
    result = []
    for i in comb:
        if len(i) > 1:
            result.append(len(set.intersection(*(set(j.columns) for j in i))))
        else:
            result.append(len(i[0].columns))
    ax = venn3(subsets = result)
    return ax

def expvsobs(df, **kwargs):
    plt.scatter(y_test, y_pred, alpha=0.5, color='darkblue', marker='o')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray', linewidth=2)
    for i in range(len(y_test)):
        plt.plot([y_test[i], y_test[i]], [y_test[i], y_pred[i]], color='red', alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Regression Model')
    return None

def merge(datasets=None, join='inner', append=None):
    if append:
        outdf = pd.concat(datasets, axis=0, join=join)
    else:
        outdf = pd.concat(datasets, axis=1, join=join)
    return outdf

def group(df, type='sum'):
    if type=='sum':
        outdf = df.groupby(level=0).sum()
    elif type=='mean':
        outdf = df.groupby(level=0).mean()
    return outdf

def filter(df, **kwargs):
    df.index = df.index.astype(str)
    if kwargs.get('filter_df') is not None:
        if kwargs.get('filter_df_axis') == 1:
            df = df.loc[:, kwargs.get('filter_df').index]
        else:
            df = df.loc[kwargs.get('filter_df').index]
    if kwargs.get('colfilt'):
        df = df.loc[:, df.columns.str.contains(kwargs.get('colfilt'), regex=True)]
    if kwargs.get('rowfilt'):
        df = df.loc[df.index.str.contains(kwargs.get('rowfilt'), regex=True)]
    if kwargs.get('prevail'):
        df = df.loc[:, df.agg(np.count_nonzero, axis=0).gt(df.shape[0]*kwargs.get('prevail'))]
    if kwargs.get('abund'):
        df = df.loc[:, df.mean().gt(kwargs.get('abund'))]
    if kwargs.get('min_unique'):
        df = df.loc[:, df.nunique().gt(kwargs.get('min_unique'))]
    if kwargs.get('nonzero'):
        df = df.loc[df.sum(axis=1) != 0, df.sum(axis=0) !=0]
    if kwargs.get('numeric_only'):
        if ~(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())).any():
            df = pd.DataFrame()
    if kwargs.get('query'):
        df = df.query(kwargs.get('query'))
    if kwargs.get('dtype'):
        df = df.select_dtypes(kwargs.get('dtype'))
    if df.empty:
        return None
    else:
        return df

def SHAP_interact(df, model, **kwargs):
    X = df.copy()
    explainer = shap.TreeExplainer(model)
    inter_shaps_values = explainer.shap_interaction_values(X)
    vals = inter_shaps_values[0]
    for i in range(1, vals.shape[0]): vals[0] += vals[i]
    final = pd.DataFrame(vals[0], index=X.columns, columns=X.columns)
    final = final.stack().sort_values().to_frame('SHAP_interaction')
    final.index = final.index.set_names(['source', 'target'], level=[0,1])
    return final

def SHAP_bin(df, model, **kwargs):
    # work on this to get std shap
    X = df.copy()
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series(
        np.abs(shaps_values.values[:, :, 0]).mean(axis=0),
        index=X.columns
    )
    corrs = [spearmanr(shaps_values.values[:, x, 1], X.iloc[:, x])[0] for x in range(len(X.columns))]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    final = final.sort_values()
    return final

def SHAP_reg(X, model):
    import numpy as np
    from scipy.stats import spearmanr
    import pandas as pd
    import shap
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series(
            np.abs(shaps_values.values).mean(axis=0),
            index=X.columns
            )
    corrs = [spearmanr(shaps_values.values[:,x], X.iloc[:,x])[0] for x in range(len(X.columns))]
    shaps = pd.DataFrame(shaps_values.values, columns=X.columns, index=X.index)
    #shaps = shaps.loc[:,shaps.sum() != 0]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    return final

def explain(analysis, subject, **kwargs):
    available={
        'SHAP_bin':SHAP_bin,
        'SHAP_interact':SHAP_interact
        }
    output = available.get(analysis)(df, model, **kwargs)
    return output

def upsample(df):
    upsample = resample(df.loc[df.index.value_counts().index[1]],
                 replace=True,
                 n_samples=df.index.value_counts().iloc[0],
                 random_state=42)
    return pd.concat([df.drop(df.index.value_counts().index[1]), upsample])

def corrpair(df1, df2, FDR=True, min_unique=0):
    df1 = df1.loc[:, df1.nunique() > min_unique]
    df2 = df2.loc[:, df2.nunique() > min_unique]
    df = df1.join(df2, how='inner')
    cor, pval = spearmanr(df)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    cordf = cordf.loc[df1.columns, df2.columns]
    pvaldf = pvaldf.loc[df1.columns, df2.columns]
    pvaldf.fillna(1, inplace=True)
    if FDR:
        pvaldf = pd.DataFrame(
            fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
            index = pvaldf.index,
            columns = pvaldf.columns)
    return cordf, pvaldf

def corr(df):
    combs = list(permutations(df.columns.unique(), 2))
    outdf = pd.DataFrame(index=pd.MultiIndex.from_tuples(combs), columns=['cor','pval'])
    for comb in combs:
        tdf = pd.concat([df[comb[0]], df[comb[1]]], axis=1).dropna()
        cor, pval = spearmanr(tdf[comb[0]], tdf[comb[1]])
        outdf.loc[comb, 'cor'] = cor
        outdf.loc[comb, 'pval'] = pval
    outdf['qval'] = fdrcorrection(outdf.pval)[1]
    outdf.index.set_names(['source', 'target'], inplace=True)
    return outdf

def chisq(df, **kwargs):
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    combs = list(permutations(df.columns.unique(), 2))
    comb = combs[0]
    pvals=[]
    obsa=[]
    for comb in combs:
        test = df.loc[:, [comb[0], comb[1]]]
        obs = pd.crosstab(test.iloc[:,0], test.iloc[:,1])
        if not obs.empty:
            chi2, p, dof, expected = chi2_contingency(obs)
            exp = pd.DataFrame(expected,index=obs.index, columns=obs.columns)
        # Test assumptions - if passes con
        # EX of cells should be <= 5 in 80%
        # No cell should have EX < 1
        obsa.append(obs)
        if (~exp.lt(1).any().any()) & ((exp.gt(5).sum().sum()) / (exp.shape[0] * exp.shape[1]) > 0.8):
            pvals.append(p)
        else:
            pvals.append(np.nan)
    pvaldf = pd.Series(pvals, index=pd.MultiIndex.from_tuples(combs), dtype=float)
    obsdict = dict(zip(combs, obsa))
    return obsdict, pvaldf

def shapiro(df, **kwargs):
    #fix
    output = pd.DataFrame()
    for col in df.columns: 
        for cat in df.index.unique():
            output.loc[col,cat] = shapiro(df.loc[cat,col])[1]
    return output

def levene(df_true, df_false):
    #fix
    output = pd.Series()
    for col in df_true.columns: 
        output[col] = levene(*[df.loc[cat,col] for cat in df.index.unique()])[1]
    return output

def ANCOM(df):
    outdf = ancom(df, df.index.to_series())[0]['Reject null hypothesis'].to_frame('ANCOM')
    return outdf

def mww(df_true, df_false):
    outdf = pd.DataFrame(
            mannwhitneyu(df_true, df_false).pvalue,
            index=df_true.columns,
            columns=['MWW_pval'])
    outdf['MWW_qval'] = fdrcorrection(outdf.MWW_pval)[1]
    return outdf

def fc(df_true, df_false):
    outdf = pd.DataFrame(
        df_true.mean().div(df_false.mean()),
        columns = ['FC'],
        index = df_true.columns)
    outdf['Log2FC'] = outdf.FC.apply(np.log2)
    outdf['Log10FC'] = outdf.FC.apply(np.log10)
    return outdf

def diffmean(df_true, df_false):
    outdf = pd.DataFrame(
        df_true.mean().sub(df_false.mean()),
        columns = ['diffmean'],
        index = df_true.columns)
    return outdf

def summary(df_true: pd.DataFrame, df_false: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics combining mean and standard deviation for true and false values.
    Parameters
    ----------
    df_true : pd.DataFrame
        DataFrame containing the true values.
    df_false : pd.DataFrame
        DataFrame containing the false values.
    Returns
    -------
    pd.DataFrame
        DataFrame containing mean ± standard deviation for each column in both df_true and df_false.
    """
    def get_summary(df: pd.DataFrame) -> pd.Series:
        mean_values = df.mean().round(2).astype(str)
        std_values = df.std().round(2).astype(str)
        return mean_values + ' ± ' + std_values
    true_summary = get_summary(df_true)
    false_summary = get_summary(df_false)
    summary_df = pd.DataFrame({
        'source_true_summary': true_summary,
        'source_false_summary': false_summary
    })
    return summary_df

def change(df, df2, columns=None, analysis=['mww','fc','diffmean','summary']):
    """
    Perform various analyses on specified columns of a DataFrame and summarize the results.
    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame containing the data to be analyzed.
    df2 : pd.DataFrame
        DataFrame containing one-hot encoded columns for splitting data in df.
    columns : list, optional
        List of column names to be analyzed. If None, all columns in df2 are used.
    analysis : list, optional
        List of analyses to perform. Default is ['mww', 'fc', 'diffmean', 'summary'].
    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the specified analyses for each column.
    Example
    -------
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 6, 7, 8, 9],
    ...     'C': [10, 11, 12, 13, 14]
    ... })
    >>> df2 = pd.DataFrame({
    ...     'X': [1, 0, 1, 0, 1],
    ...     'Y': [0, 1, 0, 1, 0],
    ...     'Z': [1, 1, 0, 0, 1]
    ... })
    >>> result = change(df, df2)
    """
    if columns is None: columns = df2.columns
    available={'mww':mww, 'fc':fc, 'diffmean':diffmean, 'summary':summary}
    i = 'summary'
    label = df2.columns[2]
    out = {}
    for label in df2[columns].columns:
        output = []
        split = splitter(df, df2, label)
        if len(split) != 2: continue
        for i in analysis:
            df_true, df_false = split[True], split[False]
            output.append(available.get(i)(df_true, df_false))
        out[label] = pd.concat(output, axis=1)
    outdf = pd.concat(out.values(), axis=0, keys=out.keys())
    outdf.index.set_names(['source','target'], inplace=True)
    return outdf

# Calculate
def Richness(df, axis=1):
    return df.agg(np.count_nonzero, axis=axis)

def Evenness(df, axis=1):
    return df.agg(pielou_e, axis=axis)

def Shannon(df, axis=1):
    return df.agg(shannon, axis=axis)

def Chao(df, axis=1):
    return df.agg(chao1, axis=axis)

def FaithPD(df, tree, axis=1):
    return df.agg(faith_pd, tree, axis=axis)

def diversity(df):
    diversity = pd.concat(
            [Evenness(df.copy()).to_frame('Evenness'),
             Richness(df.copy()).to_frame('Richness'),
             Chao(df.copy()).to_frame('Chao'),
             Shannon(df.copy()).to_frame('Shannon')],
            axis=1).sort_index().sort_index(ascending=False)
    return diversity

def dist(df, metric='braycurtis'):
    Ar_dist = squareform(pdist(df, metric=metric))
    output = pd.DataFrame(Ar_dist, index=df.index, columns=df.index)
    return output

def fbratio(df):
    phylum = df.copy()
    phylum.columns = phylum.columns.str.extract('p__(.*)\|c')[0]
    taxoMsp = phylum.T.groupby(phylum.columns).sum()
    taxoMsp = taxoMsp.loc[taxoMsp.sum(axis=1) != 0, taxoMsp.sum(axis=0) != 0]
    taxoMsp = taxoMsp.T.div(taxoMsp.sum(axis=0), axis=0)
    FB = taxoMsp.Firmicutes.div(taxoMsp.Bacteroidetes)
    FB.replace([np.inf, -np.inf], np.nan, inplace=True)
    FB.dropna(inplace=True)
    FB = FB.reset_index().set_axis(['Host Phenotype', 'FB_Ratio'], axis=1).set_index('Host Phenotype')
    return FB

def pbratio(genus):
    tmpdf = mult(genus)
    pb = tmpdf.Prevotella.div(genus.Bacteroides)
    pb.replace([np.inf, -np.inf], np.nan, inplace=True)
    pb.dropna(inplace=True)
    pb = pb.reset_index().set_axis(['Host Phenotype', 'PB_Ratio'], axis=1).set_index('Host Phenotype')
    return pb

def pca(df):
    scaledDf = StandardScaler().fit_transform(df)
    pca = PCA()
    results = pca.fit_transform(scaledDf).T
    df['PC1'], df['PC2'] = results[0,:], results[1,:]
    return df[['PC1', 'PC2']]

def pcoa(df):
    ndf = df.copy()
    Ar_dist = distance.squareform(distance.pdist(ndf, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)
    print(PCoA.proportion_explained)
    results = PCoA.samples.copy()
    ndf['PC1'], ndf['PC2'] = results.iloc[:,0].values, results.iloc[:,1].values
    return ndf[['PC1', 'PC2']]

def nmds(df):
    BC_dist = pd.DataFrame(
        distance.squareform(distance.pdist(df, metric="braycurtis")),
        columns=df.index,
        index=df.index) 
    mds = MDS(n_components = 2, metric = False, max_iter = 500, eps = 1e-12, dissimilarity = 'precomputed')
    results = mds.fit_transform(BC_dist)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def tsne(df):
    results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def top20(df):
    if df.shape[1] > 20:
        df['other'] = df[df.sum().sort_values(ascending=False).iloc[19:].index].sum(axis=1)
    df = df[df.sum().sort_values().tail(20).index]
    return df

def som(df):
    som = SOM(m=3, n=1, dim=2)
    som.fit(df)
    return som

def umap(df):
    scaledDf = StandardScaler().fit_transform(df)
    reducer = umap.UMAP()
    results = reducer.fit_transform(scaledDf)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def prevail(df):
    return df.agg(np.count_nonzero, axis=0).div(df.shape[0]).to_frame('baseprev')

def onehot(df):
    return pd.get_dummies(df, prefix_sep='.')

def calculate(analysis, df):
    available={
        'prevail':prevail,
        'diversity':diversity,
        'fbratio':fbratio,
        'pbratio':pbratio,
        'pca':pca,
        'pcoa':pcoa,
        'nmds':nmds,
        'tsne':tsne,
        'top20':top20,
        'som':som,
        'umap':umap,
        'onehot':onehot,
        }
    output = available.get(analysis)(df)
    return output

# Scale
def norm(df):
    return df.T.div(df.sum(axis=1), axis=1).T

def zscore(df, axis=0):
    return df.apply(zscore, axis=axis)

def standard(df):
    scaledDf = pd.DataFrame(
            StandardScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def minmax(df):
    scaledDf = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def log(df):
    return df.apply(np.log1p)

def CLR(df):
    return df.T.apply(clr).T

def mult(df):
    return pd.DataFrame(mul(df.T).T, index=df.index, columns=df.columns)

def scale(analysis, df):
    available={
        'norm':norm,
        'standard':standard,
        'minmax':minmax,
        'log':log,
        'CLR':CLR,
        'mult':mult,
        }
    output = available.get(analysis)(df)
    return output

# Variance
def PERMANOVA(df, pval=True, full=False):
    np.random.seed(0)
    Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    result = permanova(DM_dist, df.index)
    if full:
        return result
    if pval:
        return result['p-value']
    else:
        return result['test statistic']

def explainedvariance(df1, df2, pval=True, **kwargs):
    # how does df1 explain variance in df2 where df2 is meta (only categories)
    # should rework this one to include in calculate but hard
    # 4.32 is significant
    # -np.log(0.05)
    target = df2.columns[0]
    output = pd.Series()
    for target in df2.columns:
        tdf = df1.join(df2[target].fillna('missing'),how='inner').set_index(target)
        if all(tdf.index.value_counts().lt(10)):
            continue
        if tdf.index.nunique() <= 1:
            continue
        output[target] = PERMANOVA(tdf, pval=pval)
    if pval:
        power = -output.apply(np.log2)
    else:
        power = output
    power = power.to_frame(df1)
    return power

# Stratify
def stratify(df, meta, level):
    metadf = df.join(meta[level].dropna(), how='inner').reset_index(drop=True).set_index(level).sort_index()
    if metadf.empty:
        return None
    return metadf

# Splitter
def splitter(df, df2, column):
    output = {}
    if df2 is None:
        df2 = df.copy()
    for level in df2[column].unique():
        merge = df.loc[:, df.columns[df.columns != column]].join(df2.loc[df2[column] == level, column], how='inner').drop(column, axis=1)
        output[level] = merge
    return output

# Misc
def taxofunc(msp, taxo, short=False):
    m, t = msp.copy(), taxo.copy()
    t['superkingdom'] = 'k_' + t['superkingdom']
    t['phylum'] = t[['superkingdom', 'phylum']].apply(lambda x: '|p_'.join(x.dropna().astype(str).values), axis=1)
    t['class'] = t[['phylum', 'class']].apply(lambda x: '|c_'.join(x.dropna().astype(str).values), axis=1)
    t['order'] = t[['class', 'order']].apply(lambda x: '|o_'.join(x.dropna().astype(str).values), axis=1)
    t['family'] = t[['order', 'family']].apply(lambda x: '|f_'.join(x.dropna().astype(str).values), axis=1)
    t['genus'] = t[['family', 'genus']].apply(lambda x: '|g_'.join(x.dropna().astype(str).values), axis=1)
    t['species'] = t[['genus', 'species']].apply(lambda x: '|s_'.join(x.dropna().astype(str).values), axis=1)
    mt = m.join(t, how='inner')
    df = pd.concat([mt.set_index(t.columns[i])._get_numeric_data().groupby(level=0).sum() for i in range(len(t.columns))])
    df.index = df.index.str.replace(" ", "_", regex=True).str.replace('/.*','', regex=True)
    if short:
        df.index = df.T.add_prefix("|").T.index.str.extract(".*\|([a-z]_.*$)", expand=True)[0]
    df = df.loc[df.sum(axis=1) != 0, df.sum(axis=0) != 0]
    return df

def fdr(pvaldf):
    out = pd.DataFrame(
        fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
        index = pvaldf.index,
        columns = pvaldf.columns)
    return out

def lmer(df): 
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    data = df.copy()
    md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
    mdf = md.fit()
    print(mdf.summary())

def MERF(df):
    from merf.utils import MERFDataGenerator
    from merf.merf import MERF
    from merf.viz import plot_merf_training_stats
    mrf = MERF(max_iterations=5)
    mrf.fit(X_train, Z_train, clusters_train, y_train)

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)

def save(df, subject, index=True):
    output_path = f'../results/{subject}.tsv' 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def savefig(subject, tl=False, show=False):
    if os.path.isfile(subject): subject = Path(subject).stem
    if tl: plt.tight_layout()
    plt.savefig(f'../results/{subject}.svg')
    plt.savefig(f'../results/{subject}.pdf')
    plt.savefig(f'../results/{subject}.jpg')
    if show: plt.show()
    subprocess.call(f'zathura ../results/{subject}.pdf &', shell=True)
    plt.clf()

def selecttaxa(df, level):
    mapping = df.index.str.split('\|', expand=True).to_frame().set_index(df.index)
    mapping = mapping.set_axis([mapping[i].str[0].value_counts().index[0] for i in mapping.columns], axis=1)
    mapping = mapping.replace('.__','', regex=True)
    mapping.loc[mapping.s.isna()]['g'].dropna()

def to_edges(df, thresh=0.5, directional=True):
    df = df.rename_axis('source', axis=0).rename_axis("target", axis=1)
    edges = df.stack().to_frame()[0]
    nedges = edges.reset_index()
    edges = nedges[nedges.target != nedges.source].set_index(['source','target']).drop_duplicates()[0]
    if directional:
        fin = edges.loc[(edges < 0.99) & (edges.abs() > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index('source').sort_values('weight')
    else:
        fin = edges.loc[(edges < 0.99) & (edges > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index('source').sort_values('weight')
    return fin

if __name__ == '__main__':
    print('All listed python functions:')
    print(vars().keys())
