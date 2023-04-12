# -*- coding: utf-8 -*-
"""
Created on Sun May 22 15:57:49 2022

@author: XPS
"""
import numpy as np
import scipy as sp
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os 
from collections import defaultdict
from scipy.stats import entropy
import seaborn as sns
from matplotlib import gridspec
import copy 
sns.set_style('darkgrid')


record_path = r'./exp_record.txt'
#%%
def read_para_search(idx_range):
    para_search = []
    with open('./exp_record.txt', mode='r') as f:
        lines = f.readlines()
        for i in idx_range:
            line = lines[i]
            para_search.append([eval(line.split("|")[0]), eval(line.split("|")[1])])
    return para_search

def gini_impurity(p):
    p = np.array(p, dtype='float')
    p =  p/np.sum(p, axis=1, keepdims=True)
    return  np.sum(p*(1-p), axis=1)


#%%
def calculate_int_obs_metrics(user_int_matrix, user_obs_matrix, user_rec_matrix, user_pos_matrix):
    # entropy
    entropy_int = entropy(user_int_matrix, axis=1)
    entropy_obs = entropy(user_obs_matrix, axis=1)
    entropy_rec = entropy(user_rec_matrix, axis=1)
    entropy_pos = entropy(user_pos_matrix, axis=1)
    
    
    entropy_diff_1 = entropy_obs/entropy_int
    entropy_diff_2 = entropy_rec/entropy_int
    entropy_diff_3 = entropy_pos/entropy_int
    
    # coverage
    coverage_int = np.sum(user_int_matrix>0, axis=1)
    coverage_obs = np.sum(user_obs_matrix>0, axis=1)
    coverage_rec = np.sum(user_rec_matrix>0, axis=1)
    coverage_pos = np.sum(user_pos_matrix>0, axis=1)
    
    coverage_diff_1 = coverage_obs/coverage_int
    coverage_diff_2 = coverage_rec/coverage_int
    coverage_diff_3 = coverage_pos/coverage_int

    # gini impurity
    gini_int = gini_impurity(user_int_matrix)
    gini_obs  = gini_impurity(user_obs_matrix)
    gini_rec = gini_impurity(user_rec_matrix)
    gini_pos = gini_impurity(user_pos_matrix)

    gini_diff_1 = gini_obs/gini_int
    gini_diff_2 = gini_rec/gini_int
    gini_diff_3 = gini_pos/gini_int

    # bias
    interest_int_indicator = (user_int_matrix>0).astype('int')
    obs_diff = np.around(user_obs_matrix - user_int_matrix, 5)
    user_rec_matrix = user_rec_matrix/np.sum(user_rec_matrix, axis=1, keepdims=True)
    rec_diff = np.around(user_rec_matrix - user_int_matrix, 5)
 
    obs_over_exp_ratio = np.sum((obs_diff>0)*interest_int_indicator*user_obs_matrix, axis=1)
    obs_under_ser_ratio = np.sum((obs_diff<0)*interest_int_indicator*user_obs_matrix, axis=1)
    rec_over_exp_ratio = np.sum((rec_diff>0)*interest_int_indicator*user_rec_matrix, axis=1)
    rec_under_ser_ratio = np.sum((rec_diff<0)*interest_int_indicator*user_rec_matrix, axis=1)


    obs_over_exp_degree = np.sum((obs_diff>0)*obs_diff*interest_int_indicator, axis=1)/np.sum((obs_diff>0)*user_int_matrix*interest_int_indicator, axis=1)
    obs_under_ser_degree= np.sum((obs_diff<0)*obs_diff*interest_int_indicator, axis=1)/np.sum((obs_diff<0)*user_int_matrix*interest_int_indicator, axis=1)
    rec_over_exp_degree = np.sum((rec_diff>0)*rec_diff*interest_int_indicator, axis=1)/np.sum((rec_diff>0)*user_int_matrix*interest_int_indicator, axis=1)
    rec_under_ser_degree= np.sum((rec_diff<0)*rec_diff*interest_int_indicator, axis=1)/np.sum((rec_diff<0)*user_int_matrix*interest_int_indicator, axis=1)

    if  np.sum((rec_diff<0)*user_int_matrix*interest_int_indicator, axis=1).any()==0:
        print("!",np.sum((rec_diff<0)*rec_diff*interest_int_indicator, axis=1)[np.sum((rec_diff<0)*user_int_matrix*interest_int_indicator, axis=1)==0])    
    return {'entropy_int': entropy_int, 'entropy_obs': entropy_obs, 'entropy_rec': entropy_rec, 'entropy_pos': entropy_pos,\
            'entropy_diff_1':entropy_diff_1,'entropy_diff_2':entropy_diff_2,'entropy_diff_3':entropy_diff_3,\
            'gini_int': gini_int, 'gini_obs': gini_obs, 'gini_rec': gini_rec, 'gini_pos': gini_pos,\
            'gini_diff_1':gini_diff_1,'gini_diff_2':gini_diff_2,'gini_diff_3':gini_diff_3,\
            'coverage_int': coverage_int, 'coverage_obs': coverage_obs, 'coverage_rec': coverage_rec, 'coverage_pos': coverage_pos,\
            'coverage_diff_1':coverage_diff_1,'coverage_diff_2':coverage_diff_2,'coverage_diff_3':coverage_diff_3,\
            'obs_oe_ratio': obs_over_exp_ratio, 'obs_us_ratio': obs_under_ser_ratio,\
            'rec_oe_ratio': rec_over_exp_ratio, 'rec_us_ratio': rec_under_ser_ratio,\
            'obs_oe_degree': obs_over_exp_degree, 'obs_us_degree': obs_under_ser_degree,\
            'rec_oe_degree': rec_over_exp_degree, 'rec_us_degree': rec_under_ser_degree}

def calculate_feedback(user_pos_cnt, user_neg_cnt, user_pos_matrix, user_neg_matrix):

    hit_rate = user_pos_cnt/(user_neg_cnt + user_pos_cnt)
    
    entropy_pos = entropy(user_pos_matrix, axis=1)
    entropy_neg = entropy(user_neg_matrix, axis=1)

    return {'hit_rate': hit_rate, 'entropy_pos': entropy_pos, 'entropy_neg':entropy_neg}

def calculate_go_to_static(all_time_step_result, thres=0.01):
    num_iter = len(all_time_step_result)
    ts_1 = all_time_step_result[0]['entropy_diff_2']
    ts_final = all_time_step_result[-1]['entropy_diff_2']
    for t in range(1, num_iter):
        diff_1 = all_time_step_result[t]['entropy_diff_2'] - ts_1
        diff_2 = all_time_step_result[t]['entropy_diff_2'] - ts_final
 
        if np.abs(np.nanmean(diff_2)) < 0.001:
            break

    result = {'ts': t, 'degree': diff_1, 'init': ts_1, 'mean_static': np.nanmean(all_time_step_result[t]['entropy_diff_2']), 'mean_init': np.nanmean(ts_1)} 
    return result



def generate_name(para_basic):
    pic_name = r'a:{:.2f} b:{:.2f} gp:{:.2f} gm:{:.2f} s:{:.2f}'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])
    pic_path = r'./pic/'+r'a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.2f}/'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])
    temporal_result_path = r'./tmp_re/'+r'a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.2f}/'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])

    if 'fined_sigma' in para_basic.keys():
       pic_name = r'a:{:.2f} b:{:.2f} gp:{:.2f} gm:{:.2f} s:{:.3f}'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])
       pic_path = r'./pic/'+r'a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.3f}/'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])
       temporal_result_path = r'./tmp_re/'+r'a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.3f}/'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])

    if para_basic['sim_type'] != 'dot':
       pic_name = r'a:{:.2f} b:{:.2f} gp:{:.2f} gm:{:.2f} s:{:.3f}'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])
       pic_path = r'./'+para_basic['sim_type']+'/pic/'+r'a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.2f}/'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])
       temporal_result_path = r'./'+para_basic['sim_type']+'/tmp_re/'+r'a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.2f}/'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])


    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    if not os.path.exists(temporal_result_path):
        os.mkdir(temporal_result_path)

    return pic_name, pic_path,temporal_result_path

# 画图封装成类

class pics:
    def __init__(self, para_basic, mode, step=10):
        self.para_basic = para_basic
        self.mode = mode
        self.step = step
    
    
    def calculate_metrics(self):
        path = self.para_basic['results_path']

        if not os.path.exists(path):
            path =  './'+self.para_basic['results_path'].split('/')[2]+'/'
        if not os.path.exists(path+'init_users.pkl'):
            print("Rerun:", self.para_basic['results_path'].split('/')[2])
            return 0

        with open(path+'init_users.pkl', 'rb') as f:
            init_users = pickle.load(f) 
        user_int_matrix = np.vstack([np.around(u.interest_int, 5) for u in init_users])

        # stationary
        all_time_step_result = []

        for i in range(0, self.para_basic['num_iter'], self.step):

            user_obs_matrix = np.zeros((self.para_basic['num_users'], self.para_basic['num_topics']))
            user_rec_matrix = np.zeros((self.para_basic['num_users'], self.para_basic['num_topics']))
            user_pos_matrix = np.zeros((self.para_basic['num_users'], self.para_basic['num_topics']))
            user_neg_matrix = np.zeros((self.para_basic['num_users'], self.para_basic['num_topics']))
            
            user_pos_cnt = np.zeros((self.para_basic['num_users']))
            user_neg_cnt = np.zeros((self.para_basic['num_users']))
            

            for j in range(self.step):
                with open(path+'run'+str(i+j)+'.pkl', 'rb') as f:
                    current_users = pickle.load(f) 
                user_obs_matrix += np.vstack([np.around(u['interest_obs'], 5) for u in current_users])
                user_rec_matrix += np.vstack([np.sum(u['item2rec'], axis=0) for u in current_users])
                
                
                user_feedback_idx =  np.vstack([u['feedback'] for u in current_users])
                
                user_pos_cnt += np.sum((user_feedback_idx==1), axis=1)
                user_neg_cnt += np.sum((user_feedback_idx==0), axis=1)
                
                user_pos_matrix += np.vstack([np.sum(u['item2rec'][np.where(u['feedback']==1)[1], :], axis=0) for u in current_users])
                user_neg_matrix += np.vstack([np.sum(u['item2rec'][np.where(u['feedback']==0)[1], :], axis=0) for u in current_users])
            
            user_obs_matrix = user_obs_matrix/self.step
            user_rec_matrix = user_rec_matrix/self.step
            user_pos_matrix = user_pos_matrix/self.step
            user_neg_matrix = user_neg_matrix/self.step
            
            re_1 = calculate_int_obs_metrics(user_int_matrix, user_obs_matrix, user_rec_matrix, user_pos_matrix)
            re_2 = calculate_feedback(user_pos_cnt, user_neg_cnt, user_pos_matrix, user_neg_matrix)
            re = {**re_1, **re_2}
            all_time_step_result.append(re)
            
        # metrics
        user_obs_matrix = np.zeros((self.para_basic['num_users'], self.para_basic['num_topics']))
        user_rec_matrix = np.zeros((self.para_basic['num_users'], self.para_basic['num_topics']))
        for i in range(self.para_basic['num_iter']-self.step, self.para_basic['num_iter']):
            with open(path+'run'+str(i)+'.pkl', 'rb') as f:
                current_users = pickle.load(f)
            user_obs_matrix += np.vstack([np.around(u['interest_obs'], 5) for u in current_users])
            user_rec_matrix += np.vstack([np.sum(u['item2rec'], axis=0)/np.sum(u['item2rec']) for u in current_users])
        user_obs_matrix = user_obs_matrix/self.step
        user_rec_matrix = user_rec_matrix/self.step
        re_3 = {'user_int_matrix': user_int_matrix, 'user_obs_matrix': user_obs_matrix, 'user_rec_matrix': user_rec_matrix}
        
        # metrics
        re_4 = calculate_go_to_static(all_time_step_result, 0.01)

        # storage
        pic_name, pic_path, tmp_path = generate_name(self.para_basic)

        self.all_time_step_result = all_time_step_result
        self.re_3 = re_3
        self.re_4 = re_4
        self.pic_name = pic_name
        self.pic_path = pic_path

        with open(tmp_path+'dynamic_v1.pkl','wb') as f:
            pickle.dump(all_time_step_result, f)
        with open(tmp_path+'static_v1.pkl', 'wb') as f:
            pickle.dump(re_3, f)
        with open(tmp_path+'gotostatic_v1.pkl', 'wb') as f:
            pickle.dump(re_4, f)        
    
    def read_processed_file(self):
        pic_name, pic_path, tmp_path = generate_name(self.para_basic)
        self.pic_name = pic_name
        self.pic_path = pic_path

        with open(tmp_path+'dynamic_v1.pkl','rb') as f:
            self.all_time_step_result = pickle.load(f)
        with open(tmp_path+'static_v1.pkl', 'rb') as f:
            self.re_3 = pickle.load(f)
        with open(tmp_path+'gotostatic_v1.pkl', 'rb') as f:
            self.re_4 = pickle.load(f)

    def run_pics(self):
        if self.mode == 'raw':
            error = self.calculate_metrics()
            if error == 0:
                return 0
        elif self.mode == 'processed':
            self.read_processed_file()
    
    def fig0(self):
        # 画图
        # pic_index: 0 总图
        pic_index = 0
        fig = plt.figure(figsize=(12,12))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 15
        
        ax0 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
        ax2 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
        ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=1)
        ax4 = plt.subplot2grid((3, 2), (2, 1), colspan=1)
        
        # axe0
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for i in range(len(self.all_time_step_result)):
            tmp1.append(np.nanmean(self.all_time_step_result[i]['entropy_diff_1']))
            tmp2.append(np.nanmean(self.all_time_step_result[i]['entropy_diff_2']))
            tmp3.append(np.nanmean(self.all_time_step_result[i]['entropy_diff_3']))
        ax0.plot(tmp1, label='obs/intr')
        ax0.plot(tmp2, label='rec/intr')
        ax0.plot(tmp3, label='pos/intr')
        ax0.hlines(y=1, xmin=0, xmax=len(self.all_time_step_result), color='r', linestyle='--', label='equal')
        ax0.hlines(y=self.re_4['mean_static'], xmin=0, xmax=self.re_4['ts'], color='purple', linestyle='dashdot', label='rec_static')
        ax0.legend()
        ax0.set_title(self.pic_name)
        
        #axes1
        tmp1 = []
        tmp2 = []
        #tmp3 = []
        for i in range(len(self.all_time_step_result)):
            tmp1.append(np.nanmean(self.all_time_step_result[i]['rec_oe_ratio']))
            tmp2.append(np.nanmean(self.all_time_step_result[i]['rec_us_ratio']))
            #tmp3.append(np.nanmean(self.all_time_step_result[i]['mis_ratio']))
        ax1.plot(tmp1, label='Over Exploit')
        ax1.plot(tmp2, label='Under Serve ')
        #ax1.plot(tmp3, color='black', linestyle='--',label='Mistake')
        ax1.set_ylim(0,1)
        ax1.legend()
        ax1.set_title("Overall")
        
        #axes2
        tmp1 = []
        tmp2 = []
        #tmp3 = []
        for i in range(len(self.all_time_step_result)):
            tmp1.append(np.nanmean(self.all_time_step_result[i]['rec_oe_degree']))
            tmp2.append(np.nanmean(self.all_time_step_result[i]['rec_us_degree']))
            #tmp3.append(np.nanmean(self.all_time_step_result[i]['ab_degree']))
        ax2.plot(tmp1, label='Over Exploit')
        ax2.plot(tmp2, label='Under Serve')
        #ax2.plot(tmp3, color='green', linestyle='--',label='All')
        #ax2.set_ylim(0,1.1)
        ax2.legend()
        ax2.set_title("Bias")
        
        #axes3 hit
        tmp1 = []
        for i in range(len(self.all_time_step_result)):
            tmp1.append(np.nanmean(self.all_time_step_result[i]['hit_rate']))
        ax3.plot(tmp1)
        ax3.set_ylim(0,1)
        ax3.set_title("Hit Rate")
        
        
        #axes4 entropy_pos vs entropy_neg
        tmp1 = []
        tmp2 = []
        for i in range(len(self.all_time_step_result)):
            tmp1.append(np.nanmean(self.all_time_step_result[i]['entropy_pos']))
            tmp2.append(np.nanmean(self.all_time_step_result[i]['entropy_neg']))
        ax4.plot(tmp1, color='r', label = 'Pos')
        ax4.plot(tmp2, color='blue',label = 'Neg')
        ax4.set_title("Entropy:pos vs neg")
        ax4.legend()
        
        plt.savefig(self.pic_path+str(pic_index)+".png")
        plt.show()
        plt.close(fig=fig)

    
    
    
    
    
    

