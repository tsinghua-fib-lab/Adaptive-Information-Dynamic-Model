import pandas as pd
import numpy as np 
import scipy as sp
import time
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import pickle
from collections import Counter, defaultdict

import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap

path = './raw_data/'
output_path = './processed_data/'

def merge_data():
    all_video_p1_df = pd.read_csv(path+'video_raw_data_p1.csv',index_col=0)
    all_video_p2_df = pd.read_csv(path+'video_raw_data_p2.csv',index_col=0)
    all_video_df = pd.concat([all_video_p1_df, all_video_p2_df])
    all_video_df.to_csv(path+'video_raw_data.csv')



def fig1a():
    # read data
    all_video_df = pd.read_csv(path+'video_raw_data.csv',index_col=0)
    all_news_df = pd.read_csv(path+'news_raw_data.csv',index_col=0)

    all_video_df = all_video_df.loc[all_video_df['valid_activated'] == True] # handle cold-start related issues 
    all_video_df = all_video_df[['delta_entropy']+['entropy_squeezed_sequence_{}'.format(i) for i in range(11)]] # extract data

    # fit the distribution
    n, bins, _ = plt.hist(all_video_df['delta_entropy'], bins=100, density=True, cumulative=False)
    bins = np.array((bins[1:]+bins[:-1])/2)
    def interp(x, data):
        f = sp.interpolate.interp1d(x, data)
        return f
    f = interp(bins, n)  

    # obtain the bin edges
    delta_entropy = all_video_df['delta_entropy'].to_numpy()
    sep_bins = np.quantile(delta_entropy[delta_entropy<0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sep_bins[0] = np.min(bins)
    sep_bins[-1] = 0
    sep_bins = list(sep_bins)
    sep_bins.append(np.max(bins))
    num = np.sum(delta_entropy<0)/delta_entropy.shape[0]
    min_delta_entropy = np.min(delta_entropy)

    # dump processed data 
    param_dict = {'f':f, 'num':num, 'bins':bins,'min_delta_entropy':min_delta_entropy, 'sep_bins':sep_bins}
    with open(output_path+'/fig1a_preprocessed.pkl', 'wb') as f: 
        pickle.dump(param_dict,f)

def fig1b():
    def merge_entropy_squeezed_sequence(x):
        sqz = []
        for i in range(11):
            sqz.append(x['entropy_squeezed_sequence_{}'.format(i)])
        return sqz
            

    all_video_df = pd.read_csv(path+'video_raw_data.csv',index_col=0)
    all_news_df = pd.read_csv(path+'news_raw_data.csv',index_col=0)

    all_video_df = all_video_df.loc[all_video_df['valid_activated'] == True] # process cold-start related issues 
    all_video_df['entropy_squeezed_sequence'] = all_video_df.apply(lambda x: merge_entropy_squeezed_sequence(x), axis=1)
    all_video_df['indicator'] = all_video_df['entropy_squeezed_sequence'].transform(lambda x: x[-1]-x[0])
    all_video_df = all_video_df.loc[all_video_df['indicator']<0]
    all_video_df['delta_q'] = pd.qcut(all_video_df['indicator'], q=[0, 0.2, 0.4, 0.6, 0.8, 1], labels = np.arange(5))
    all_dfs = []
    for i in range(5):
        df = pd.DataFrame(np.vstack(all_video_df.loc[all_video_df['delta_q']==i]['entropy_squeezed_sequence'].values.tolist()))
        df = df.melt()
        df['thres'] = [f'{int(i*20)}%~{int((i+1)*20)}%']*len(df)
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    
    # dump processed data
    param_dict = {'all_dfs':all_dfs}
    with open(output_path+'/fig1b_preprocessed.pkl', 'wb') as f: 
        pickle.dump(param_dict,f)

def fig1d():
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import summary_table
    from scipy import stats
    from scipy.interpolate import interp1d

    # read data
    all_video_df = pd.read_csv(path+'video_raw_data.csv',index_col=0)
    all_news_df = pd.read_csv(path+'news_raw_data.csv',index_col=0)

    # video fitting
    all_video_df = all_video_df.loc[all_video_df['pos_num']>200]
    Y_video = np.array(all_video_df['normalized_entropy'])
    X_video = np.array(all_video_df['sim_strength'])
    X_video = sm.add_constant(X_video)
    model_video = sm.OLS(Y_video, X_video)
    model_video = model_video.fit()
    n_data_video = X_video.shape[0]

    X_pred = X_video
    Y_pred = model_video.predict(X_pred)
    Y_err = Y_video - Y_pred
    mean_X = np.mean(X_video)
    dof = n_data_video - model_video.df_model - 1 # degree of freedom
    alpha = 0.025
    t = stats.t.ppf(1-alpha, df=dof) # t-value
    s_err = np.sum(Y_err**2)
    std_err = np.sqrt(s_err/(n_data_video-2))
    std_X = np.std(X_video)
    conf = t*std_err/np.sqrt(n_data_video)*np.sqrt(1+((X_pred[:,1]-mean_X)/std_X)**2) 
    upper = Y_pred + abs(conf)
    lower = Y_pred - abs(conf)

    idx = np.argsort(X_pred[:,1])
    X_video, Y_video, lower_video, upper_video = X_pred[idx], Y_pred[idx], lower[idx], upper[idx]

    X_video = X_video[:,1]
    upper_video_f = interp1d(X_video, upper_video)
    lower_video_f = interp1d(X_video, lower_video)
    mean_video_f = interp1d(X_video, Y_video)


    # news_fitting
    Y_news = np.array(all_news_df['normalized_entropy'])
    X_news = np.array(all_news_df['sim_strength'])
    X_news = sm.add_constant(X_news)
    model_news = sm.OLS(Y_news, X_news)
    model_news = model_news.fit()
    n_data_news = X_news.shape[0]


    X_pred = X_news
    Y_pred = model_news.predict(X_pred)
    Y_err = Y_news - Y_pred
    mean_X = np.mean(X_news)
    dof = n_data_news - model_news.df_model - 1 # degree of freedom
    alpha = 0.025
    t = stats.t.ppf(1-alpha, df=dof) # t-value
    s_err = np.sum(Y_err**2)
    std_err = np.sqrt(s_err/(n_data_news-2))
    std_X = np.std(X_news)
    conf = t*std_err/np.sqrt(n_data_news)*np.sqrt(1+((X_pred[:,1]-mean_X)/std_X)**2) 
    upper = Y_pred + abs(conf)
    lower = Y_pred - abs(conf)

    idx = np.argsort(X_pred[:,1])
    X_news, Y_news, lower_news, upper_news = X_pred[idx], Y_pred[idx], lower[idx], upper[idx]

    X_news = X_news[:,1]
    upper_news_f = interp1d(X_news, upper_news)
    lower_news_f = interp1d(X_news, lower_news)
    mean_news_f = interp1d(X_news, Y_news)

    # dump processed data
    param_dict = {'X_video':X_video, 'upper_video_f':upper_video_f, 'lower_video_f': lower_video_f, 'mean_video_f':mean_video_f,\
                'X_news':X_news, 'upper_news_f':upper_news_f, 'lower_news_f': lower_news_f, 'mean_news_f':mean_news_f}

    with open(output_path+'/fig1d_preprocessed.pkl', 'wb') as f: 
        pickle.dump(param_dict, f)

def fig1e():
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import summary_table
    from scipy import stats
    from scipy.interpolate import interp1d

    # read data
    all_video_df = pd.read_csv(path+'video_raw_data.csv',index_col=0)
    all_news_df = pd.read_csv(path+'news_raw_data.csv',index_col=0)

    # video fitting
    all_video_df = all_video_df.loc[all_video_df['pos_num']>200]
    hist, X, Y= np.histogram2d(all_video_df['pos_ratio'], all_video_df['normalized_entropy'], bins=[100, 100])
    ind = np.where(hist==np.max(hist))
    X_fix, Y_fix= X[ind[0][0]], Y[ind[1][0]]

    Y = np.array(all_video_df['normalized_entropy']-Y_fix)
    X = np.array(all_video_df['pos_ratio']-X_fix)
    model = sm.OLS(Y, X)
    model = model.fit()
    n_data = X.shape[0]


    X_pred = X
    Y_pred = model.predict(X_pred)
    Y_err = Y - Y_pred
    mean_X = np.mean(X)
    dof = n_data - model.df_model - 1 # degree of freedom
    alpha = 0.025
    t = stats.t.ppf(1-alpha, df=dof) # t-value
    s_err = np.sum(Y_err**2)
    std_err = np.sqrt(s_err/(n_data-2))
    std_X = np.std(X)
    conf = t*std_err/np.sqrt(n_data)*np.sqrt(1+((X_pred-mean_X)/std_X)**2) 
    upper = Y_pred + abs(conf)
    lower = Y_pred - abs(conf)

    idx = np.argsort(X_pred)
    X_video, Y_video, lower_video, upper_video = X_pred[idx]+X_fix, Y_pred[idx]+Y_fix, lower[idx]+Y_fix, upper[idx]+Y_fix

    # news fitting
    hist, X, Y= np.histogram2d(all_news_df['pos_ratio'], all_news_df['normalized_entropy'], bins=[100, 100])
    ind = np.where(hist==np.max(hist))
    X_fix, Y_fix= X[ind[0][0]], Y[ind[1][0]]


    Y = np.array(all_news_df['normalized_entropy']-Y_fix)
    X = np.array(all_news_df['pos_ratio']-X_fix)
    model = sm.OLS(Y, X)
    model = model.fit()
    n_data = X.shape[0]


    X_pred = X
    Y_pred = model.predict(X_pred)
    Y_err = Y - Y_pred
    mean_X = np.mean(X)
    dof = n_data - model.df_model - 1 # degree of freedom
    alpha = 0.025
    t = stats.t.ppf(1-alpha, df=dof) # t-value
    s_err = np.sum(Y_err**2)
    std_err = np.sqrt(s_err/(n_data-2))
    std_X = np.std(X)
    conf = t*std_err/np.sqrt(n_data)*np.sqrt(1+((X_pred-mean_X)/std_X)**2) 
    upper = Y_pred + abs(conf)
    lower = Y_pred - abs(conf)

    idx = np.argsort(X_pred)
    X_news, Y_news, lower_news, upper_news = X_pred[idx]+X_fix, Y_pred[idx]+Y_fix, lower[idx]+Y_fix, upper[idx]+Y_fix

    upper_news_f = interp1d(X_news, upper_news)
    lower_news_f = interp1d(X_news, lower_news)
    mean_news_f = interp1d(X_news, Y_news)

    upper_video_f = interp1d(X_video, upper_video)
    lower_video_f = interp1d(X_video, lower_video)
    mean_video_f = interp1d(X_video, Y_video)

    # dump processed data
    param_dict = {'X_video':X_video, 'upper_video_f':upper_video_f, 'lower_video_f': lower_video_f, 'mean_video_f':mean_video_f,\
                'X_news':X_news, 'upper_news_f':upper_news_f, 'lower_news_f': lower_news_f, 'mean_news_f':mean_news_f}

    with open(output_path+'fig1e_preprocessed.pkl', 'wb') as f: 
        pickle.dump(param_dict, f)


def fig1f():
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import summary_table
    from scipy import stats
    from scipy.interpolate import interp1d
    # read data
    all_video_df = pd.read_csv(path+'video_raw_data.csv',index_col=0)
    all_news_df = pd.read_csv(path+'news_raw_data.csv',index_col=0)

    # video fitting
    all_video_df = all_video_df.loc[all_video_df['pos_num']>200]
    hist, X, Y= np.histogram2d(all_video_df['neg_ratio'], all_video_df['normalized_entropy'], bins=[100, 100])
    ind = np.where(hist==np.max(hist))
    X_fix, Y_fix= X[ind[0][0]], Y[ind[1][0]]

    Y = np.array(all_video_df['normalized_entropy']-Y_fix)
    X = np.array(all_video_df['neg_ratio']-X_fix)
    model = sm.OLS(Y, X)
    model = model.fit()
    n_data = X.shape[0]


    X_pred = X
    Y_pred = model.predict(X_pred)
    Y_err = Y - Y_pred
    mean_X = np.mean(X)
    dof = n_data - model.df_model - 1 # degree of freedom
    alpha = 0.025
    t = stats.t.ppf(1-alpha, df=dof) # t-value
    s_err = np.sum(Y_err**2)
    std_err = np.sqrt(s_err/(n_data-2))
    std_X = np.std(X)
    conf = t*std_err/np.sqrt(n_data)*np.sqrt(1+((X_pred-mean_X)/std_X)**2) 
    upper = Y_pred + abs(conf)
    lower = Y_pred - abs(conf)

    idx = np.argsort(X_pred)
    X_video, Y_video, lower_video, upper_video = X_pred[idx]+X_fix, Y_pred[idx]+Y_fix, lower[idx]+Y_fix, upper[idx]+Y_fix

    # news fitting
    hist, X, Y= np.histogram2d(all_news_df['neg_ratio'], all_news_df['normalized_entropy'], bins=[100, 100])
    ind = np.where(hist==np.max(hist))
    X_fix, Y_fix= X[ind[0][0]], Y[ind[1][0]]

    Y = np.array(all_news_df['normalized_entropy']-Y_fix)
    X = np.array(all_news_df['neg_ratio']-X_fix)
    model = sm.OLS(Y, X)
    model = model.fit()
    n_data = X.shape[0]


    X_pred = X
    Y_pred = model.predict(X_pred)
    Y_err = Y - Y_pred
    mean_X = np.mean(X)
    dof = n_data - model.df_model - 1 # degree of freedom
    alpha = 0.025
    t = stats.t.ppf(1-alpha, df=dof) # t-value
    s_err = np.sum(Y_err**2)
    std_err = np.sqrt(s_err/(n_data-2))
    std_X = np.std(X)
    conf = t*std_err/np.sqrt(n_data)*np.sqrt(1+((X_pred-mean_X)/std_X)**2) 
    upper = Y_pred + abs(conf)
    lower = Y_pred - abs(conf)

    idx = np.argsort(X_pred)
    X_news, Y_news, lower_news, upper_news = X_pred[idx]+X_fix, Y_pred[idx]+Y_fix, lower[idx]+Y_fix, upper[idx]+Y_fix

    upper_news_f = interp1d(X_news, upper_news)
    lower_news_f = interp1d(X_news, lower_news)
    mean_news_f = interp1d(X_news, Y_news)

    upper_video_f = interp1d(X_video, upper_video)
    lower_video_f = interp1d(X_video, lower_video)
    mean_video_f = interp1d(X_video, Y_video)
    
    # dump processed data
    param_dict = {'X_video':X_video, 'upper_video_f':upper_video_f, 'lower_video_f': lower_video_f, 'mean_video_f':mean_video_f,\
                'X_news':X_news, 'upper_news_f':upper_news_f, 'lower_news_f': lower_news_f, 'mean_news_f':mean_news_f}

    with open(output_path+'/fig1f_preprocessed.pkl', 'wb') as f: 
        pickle.dump(param_dict, f)

if __name__ == "__main__":
    merge_data()
    fig1a()
    fig1b()
    fig1d()
    fig1e()
    fig1f()
