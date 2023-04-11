# -*- coding: utf-8 -*-
"""
Created on Sun May 22 15:57:49 2022

@author: XPS
"""
import numpy as np
import scipy as sp
import pandas as pd
import simulator
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
    # 熵
    entropy_int = entropy(user_int_matrix, axis=1)
    entropy_obs = entropy(user_obs_matrix, axis=1)
    entropy_rec = entropy(user_rec_matrix, axis=1)
    entropy_pos = entropy(user_pos_matrix, axis=1)
    
    
    entropy_diff_1 = entropy_obs/entropy_int
    entropy_diff_2 = entropy_rec/entropy_int
    entropy_diff_3 = entropy_pos/entropy_int
    
    #覆盖度
    
    coverage_int = np.sum(user_int_matrix>0, axis=1)
    coverage_obs = np.sum(user_obs_matrix>0, axis=1)
    coverage_rec = np.sum(user_rec_matrix>0, axis=1)
    coverage_pos = np.sum(user_pos_matrix>0, axis=1)
    
    coverage_diff_1 = coverage_obs/coverage_int
    coverage_diff_2 = coverage_rec/coverage_int
    coverage_diff_3 = coverage_pos/coverage_int

    # 基尼不纯度
    gini_int = gini_impurity(user_int_matrix)
    gini_obs  = gini_impurity(user_obs_matrix)
    gini_rec = gini_impurity(user_rec_matrix)
    gini_pos = gini_impurity(user_pos_matrix)

    gini_diff_1 = gini_obs/gini_int
    gini_diff_2 = gini_rec/gini_int
    gini_diff_3 = gini_pos/gini_int

    # 信息茧房
    interest_int_indicator = (user_int_matrix>0).astype('int')
    obs_diff = np.around(user_obs_matrix - user_int_matrix, 5)
    user_rec_matrix = user_rec_matrix/np.sum(user_rec_matrix, axis=1, keepdims=True)
    rec_diff = np.around(user_rec_matrix - user_int_matrix, 5)
    #有兴趣，但是被高估/低估
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
    # 准确率
    hit_rate = user_pos_cnt/(user_neg_cnt + user_pos_cnt)
    
    # 正负反馈多样性对比
    entropy_pos = entropy(user_pos_matrix, axis=1)
    entropy_neg = entropy(user_neg_matrix, axis=1)
    #print(entropy(user_neg_matrix, axis=1))
    return {'hit_rate': hit_rate, 'entropy_pos': entropy_pos, 'entropy_neg':entropy_neg}

def calculate_go_to_static(all_time_step_result, thres=0.01):
    num_iter = len(all_time_step_result)
    ts_1 = all_time_step_result[0]['entropy_diff_2']
    ts_final = all_time_step_result[-1]['entropy_diff_2']
    for t in range(1, num_iter):
        diff_1 = all_time_step_result[t]['entropy_diff_2'] - ts_1
        diff_2 = all_time_step_result[t]['entropy_diff_2'] - ts_final
        #if np.abs(np.nanmean(diff_2)) < thres*np.abs(np.nanmean(diff_1)):
        if np.abs(np.nanmean(diff_2)) < 0.001:
            #print(t, diff_2)
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

    if 'sim_type' != 'dot':
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

        # 动态情况情况分析
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
            
        # 稳态解情况分析--各个topic的分布情况
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
        
        # 到达稳态的动力学计量 rec_entropy
        re_4 = calculate_go_to_static(all_time_step_result, 0.01)

        # 储存计算结果
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
        self.fig0() # 总图
        # self.fig1() # entropy-推荐vs观测
        # self.fig2() # entropy-观看
        # self.fig3() # coverage-推荐vs观测
        # self.fig4() # coverage-观看
        # self.fig5() # P(u^j)--推荐+观测+固有兴趣（通常坐标轴）
        # self.fig6() # P(u^j)--推荐+观测+固有兴趣（log-log）

        # self.fig7() # 推荐、观看、观测多样性变化图 entropy,gini impurity 带有ci图
        # self.fig8() # oe & us 推荐、观测

        # self.fig9() #多个topic的P(u^j)/P(u_0) -- 推荐结果
        # self.fig10() #多个topic的分布的比值
    

    
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

    def fig1(self):
        # pic_index: 1 多样性
        pic_index = 1
        fig = plt.figure(figsize=(5, 5))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for i in range(len(self.all_time_step_result)):
            tmp1.append(np.nanmean(self.all_time_step_result[i]['entropy_obs']))
            tmp2.append(np.nanmean(self.all_time_step_result[i]['entropy_rec']))
            tmp3.append(np.nanmean(self.all_time_step_result[i]['entropy_int']))
            
        plt.plot(tmp1, label='Observed')
        plt.plot(tmp2, label='Recommended')
        plt.plot(tmp3, color='r', linestyle='--', label='Intrinsic')
        plt.ylabel('Entropy')
        plt.xlabel('Time')
        plt.legend()
        plt.title(self.pic_name)
        
        plt.savefig(self.pic_path+str(pic_index)+".png")
        plt.show()
        plt.close(fig=fig)

    def fig2(self):
        # pic_index: 2 多样性
        pic_index = 2
        fig = plt.figure(figsize=(5, 5))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        
        tmp1 = []
        tmp3 = []
        for i in range(len(self.all_time_step_result)):
            tmp1.append(np.nanmean(self.all_time_step_result[i]['entropy_pos']))
            tmp3.append(np.nanmean(self.all_time_step_result[i]['entropy_int']))
        plt.plot(tmp1, label='Watched')
        plt.plot(tmp3, color='r', linestyle='--', label='Intrinsic')
        plt.ylabel('Entropy')
        plt.xlabel('Time')
        plt.legend()
        plt.title(self.pic_name)
        
        plt.savefig(self.pic_path+str(pic_index)+".png")
        plt.show()
        plt.close(fig=fig)

    def fig3(self):
        # pic_index: 3 10步1测量 覆盖性
        pic_index = 3
        fig = plt.figure(figsize=(5, 5))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for i in range(len(self.all_time_step_result)):
            tmp1.append(np.nanmean(self.all_time_step_result[i]['coverage_obs']))
            tmp2.append(np.nanmean(self.all_time_step_result[i]['coverage_rec']))
            tmp3.append(np.nanmean(self.all_time_step_result[i]['coverage_int']))
            
        plt.plot(tmp1, label='Observed')
        plt.plot(tmp2, label='Recommended')
        plt.plot(tmp3, color='r', linestyle='--', label='Intrinsic')
        plt.ylabel('Coverage')
        plt.xlabel('Time')
        plt.legend()
        plt.title(self.pic_name)
        
        plt.savefig(self.pic_path+str(pic_index)+".png")
        plt.show()
        plt.close(fig=fig)
    
    def fig4(self):
        # pic_index: 4 覆盖性
        pic_index = 4
        fig = plt.figure(figsize=(5, 5))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        
        tmp1 = []
        tmp3 = []
        for i in range(len(self.all_time_step_result)):
            tmp1.append(np.nanmean(self.all_time_step_result[i]['coverage_pos']))
            tmp3.append(np.nanmean(self.all_time_step_result[i]['coverage_int']))
        plt.plot(tmp1, label='Watched')
        plt.plot(tmp3, color='r', linestyle='--', label='Intrinsic')
        plt.ylabel('Coverage')
        plt.xlabel('Time')
        plt.legend()
        plt.title(self.pic_name)
        
        plt.savefig(self.pic_path+str(pic_index)+".png")
        plt.show()
        plt.close(fig=fig)

    def fig5(self):    
        #pic_index: 5 稳态时各个topic的分布
        pic_index = 5
        
        
        if not os.path.exists(self.pic_path+str(pic_index)):
            os.mkdir(self.pic_path+str(pic_index))
        

        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        for i in range(self.para_basic['num_topics']):
        
            
            fig = plt.figure(figsize=(5, 5))
            
            bins_max = np.max(np.concatenate([self.re_3['user_int_matrix'][:, i], self.re_3['user_obs_matrix'][:, i], self.re_3['user_rec_matrix'][:, i]]))
            bins = np.linspace(0, bins_max, 10)
            n, bins = np.histogram(self.re_3['user_int_matrix'][:, i], bins=bins)
            
            show_xticks = (bins[:-1]+bins[1:])/2 
            width = (bins[1]-bins[0])*0.75
            plt.bar(x= show_xticks, height=n/np.sum(n), color='blue', width = width, alpha=0.25)
            plt.plot(show_xticks, n/np.sum(n), color='blue', linestyle='--', label='Intrinsic')
            
            n, bins = np.histogram(self.re_3['user_obs_matrix'][:, i], bins=bins)
            plt.bar(x=show_xticks, height=n/np.sum(n), color='orange', width = width,  alpha=0.25)
            plt.plot(show_xticks, n/np.sum(n), color='orange', linestyle=':', label='Observed')
            
            n, bins = np.histogram(self.re_3['user_rec_matrix'][:, i], bins=bins)
            plt.bar(x=show_xticks, height=n/np.sum(n), color='green', width = width, alpha=0.25)
            plt.plot(show_xticks, n/np.sum(n), color='green', linestyle='-.', label='Recommended')
            
            #plt.xticks(np.arange(0, 120, 20), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.ylabel('Distribution')
            plt.xlabel('Value')
            plt.legend()
            plt.title(self.pic_name)
            plt.ylim
        
            plt.savefig(self.pic_path+str(pic_index)+"/"+str(i)+".png")
            plt.show()
            plt.close(fig=fig)
    
    def fig6(self):
        #pic_index: 6 稳态时各个topic的分布
        pic_index = 6
        
        if not os.path.exists(self.pic_path+str(pic_index)):
            os.mkdir(self.pic_path+str(pic_index))
        

        #print("int", np.sum(re_3['user_int_matrix']<0))
        #print("obs", np.sum(re_3['user_obs_matrix']<0))
        #print("rec", np.sum(re_3['user_obs_matrix']<0))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10

        for i in range(self.para_basic['num_topics']):
        
            
            fig = plt.figure(figsize=(5, 5))
            
            bins_max = np.max(np.concatenate([self.re_3['user_int_matrix'][:, i], self.re_3['user_obs_matrix'][:, i], self.re_3['user_rec_matrix'][:, i]]))
            bins = np.linspace(0, bins_max, 100)
            n, bins = np.histogram(self.re_3['user_int_matrix'][:, i], bins=bins)
            
            show_xticks = (bins[:-1]+bins[1:])/2
            #plt.bar(x= show_xticks, height=n/np.sum(n), color='blue', width= 0.01, alpha=0.25)
            plt.plot(show_xticks, n/np.sum(n), color='blue', linestyle='--', marker='v', label='Intrinsic')
            
            n, bins = np.histogram(self.re_3['user_obs_matrix'][:, i], bins=bins)
            #plt.bar(x=show_xticks, height=n/np.sum(n), color='orange', width= 0.01 , alpha=0.25)
            plt.plot(show_xticks, n/np.sum(n), color='orange', linestyle=':', marker='o' ,label='Observed')
            
            n, bins = np.histogram(self.re_3['user_rec_matrix'][:, i], bins=bins)
            #plt.bar(x=show_xticks, height=n/np.sum(n), color='green', width= 0.01, alpha=0.25)
            plt.plot(show_xticks, n/np.sum(n), color='green', linestyle='-.',  marker='x',  label='Recommended')
            
            #plt.xticks(np.arange(0, 120, 20), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel('Distribution')
            plt.xlabel('Value')
            plt.legend()
            plt.title(self.pic_name)
            plt.ylim
        
            plt.savefig(self.pic_path+str(pic_index)+"/"+str(i)+".png")
            plt.show()
            plt.close(fig=fig)

    def fig7(self):
        #pic_index: 7 # 推荐、观看、观测多样性变化图 entropy
        pic_index = 7
        
        if not os.path.exists(self.pic_path+str(pic_index)):
            os.mkdir(self.pic_path+str(pic_index))
        
        keys2plot = ['entropy_diff_2', 'gini_diff_2',\
                     'entropy_diff_3', 'gini_diff_3',\
                     'entropy_diff_1', 'gini_diff_1']
        name2plot = ['rec_entropy', 'rec_gini',\
                     'pos_entropy', 'pos_gini',\
                     'obs_entropy', 'obs_gini']
        
        tmp = defaultdict(list)
        for i in range(self.para_basic['num_iter']//self.step):
            for k in keys2plot:
                tmp[k].append(self.all_time_step_result[i][k])
        for k in keys2plot:
            tmp[k] = pd.DataFrame(np.vstack(tmp[k]).T)
            tmp[k] = tmp[k].melt()

        for i, k in enumerate(keys2plot):
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            plt.rcParams['font.size'] = 25
            fig = plt.figure(figsize=(8, 8))

            sns.lineplot(data=tmp[k], x='variable',y ='value', ci=95, linewidth=2)
            plt.hlines(y=1, xmin=0, xmax=500, color='r', linestyle='--', linewidth=2)
            plt.ylim(0, 1.5)
            plt.yticks([0, 0.5, 1.0, 1.5])
            plt.xticks([0,100,200,300,400,500])
            plt.xlabel('Time, $t$')
            plt.ylabel('Relative Entropy, $s_t/s_0$')
            
            plt.savefig(self.pic_path+str(pic_index)+"/"+name2plot[i]+".png")
            plt.show()
            plt.close(fig=fig)
        
    def fig8(self):
        #pic_index: 8 # OE&US ratio
        pic_index = 8
    
        if not os.path.exists(self.pic_path+str(pic_index)):
            os.mkdir(self.pic_path+str(pic_index))
        
        keys2plot = [('obs_oe_ratio', 'obs_us_ratio'),\
                    ('rec_oe_ratio', 'rec_us_ratio'),\
                    ('obs_oe_degree', 'obs_us_degree'),\
                    ('rec_oe_degree', 'rec_us_degree')]
        name2plot = ['obs_ratio',\
                        'rec_ratio',\
                        'obs_degree',\
                        'rec_degree']
        
        tmp = defaultdict(list)
        for i in range(self.para_basic['num_iter']//self.step):
            for k in keys2plot:
                tmp[k[0]].append(self.all_time_step_result[i][k[0]])
                tmp[k[1]].append(self.all_time_step_result[i][k[1]])

        for k in keys2plot:
            tmp[k[0]] = pd.DataFrame(np.vstack(tmp[k[0]]).T)
            tmp[k[0]] = tmp[k[0]].melt()
            tmp[k[1]] = pd.DataFrame(np.vstack(tmp[k[1]]).T)
            tmp[k[1]] = tmp[k[1]].melt()

        for i, k in enumerate(keys2plot):
            if 'ratio' in k[0]: 
                plt.rcParams['font.sans-serif'] = 'Times New Roman'
                plt.rcParams['font.size'] = 25
                fig = plt.figure(figsize=(8, 8))

                sns.lineplot(data=tmp[k[0]], x='variable',y ='value', ci=95, color='#4F9D9D', linewidth=2, label='Over Exploit')
                sns.lineplot(data=tmp[k[1]], x='variable',y ='value', ci=95, color='#FF5809', linewidth=2, label='Under Serve')
                plt.ylim(0, 1)
                plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                plt.xticks([0, 100, 200, 300, 400, 500])
                plt.xlabel('Time, $t$')
                plt.ylabel('Ratio')
                
                plt.tight_layout()
                plt.savefig(self.pic_path+str(pic_index)+"/"+name2plot[i]+".png")
                plt.show()
                plt.close(fig=fig)
            
            elif 'degree' in k[0]:
                plt.rcParams['font.sans-serif'] = 'Times New Roman'
                plt.rcParams['font.size'] = 25
                fig = plt.figure(figsize=(8, 8))

                sns.lineplot(data=tmp[k[0]], x='variable',y ='value', ci=95, color='#4F9D9D', linewidth=2, label='Over Exploit')
                sns.lineplot(data=tmp[k[1]], x='variable',y ='value', ci=95, color='#FF5809', linewidth=2, label='Under Serve')
                #plt.ylim(-1, 1)
                #plt.yticks([-1, -0.5, .0, 0.5, 1])
                plt.xticks([0,100,200,300,400,500])
                plt.xlabel('Time, $t$')
                plt.ylabel('Ratio')

                plt.tight_layout()
                plt.savefig(self.pic_path+str(pic_index)+"/"+name2plot[i]+".png")
                plt.show()
                plt.close(fig=fig)
    
    def fig9(self):
        #多个topic的P(u^j)/P(u_0) -- 推荐
        pic_index = 9
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        fig, axes = plt.subplots(nrows= 4, ncols=5, figsize=(25, 20))

        user_rec_matrix = self.re_3['user_rec_matrix']
        user_int_matrix = self.re_3['user_int_matrix']
        user_rec_matrix = user_rec_matrix/np.sum(user_rec_matrix, axis=1, keepdims=True)
        user_int_matrix = user_int_matrix/np.sum(user_int_matrix, axis=1, keepdims=True)

        user_ratio_matrix = user_rec_matrix/user_int_matrix
        user_ratio_matrix[(user_rec_matrix==0)&(user_int_matrix==0)] = 1
        bins = np.power(10.0, np.arange(-3,3.1,0.1))
        for i in range(user_ratio_matrix.shape[1]):
            tmp = user_ratio_matrix[:, i]
            tmp = tmp[(~np.isnan(tmp))&(~np.isinf(tmp))]
            axes[i//5, i%5] = sns.histplot(data=tmp, bins=bins, stat='probability', ax= axes[i//5, i%5])
            axes[i//5, i%5].set_title(str(i))
            axes[i//5, i%5].set_xlim(0.001, 1000)
            axes[i//5, i%5].set_xticks(np.power(10.0, np.arange(-3,4,1)))
            axes[i//5, i%5].vlines(x=1, ymin=0, ymax=1, color='r',linestyle='--')
            axes[i//5, i%5].set_xscale('log')
            axes[i//5, i%5].set_yscale('log')

        
        plt.savefig(self.pic_path+str(pic_index)+".png")
        plt.show()
        plt.close(fig=fig)
    
    def fig10(self):
        #P(u^j)分布的比P(u_0) -- 推荐
        pic_index = 10
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        fig, axes = plt.subplots(nrows= 4, ncols=5, figsize=(25, 20))

        user_rec_matrix = self.re_3['user_rec_matrix']
        user_int_matrix = self.re_3['user_int_matrix']
        user_rec_matrix = user_rec_matrix/np.sum(user_rec_matrix, axis=1, keepdims=True)
        user_int_matrix = user_int_matrix/np.sum(user_int_matrix, axis=1, keepdims=True)

        
        bins = np.arange(0, 1.01, 0.01)

        for i in range(user_rec_matrix.shape[1]):
            
            tmp = user_int_matrix[:, i]
            tmp = tmp[~np.isnan(tmp)]

            int_hist, _ = np.histogram(tmp, bins=bins)
            rec_hist, _ = np.histogram(user_rec_matrix[:, i], bins = bins)

            ratio_hist = rec_hist/int_hist

            axes[i//5, i%5].bar(bins[1:]+0.005, ratio_hist, width=0.009)
            axes[i//5, i%5].set_title(str(i))
        
        plt.savefig(self.pic_path+str(pic_index)+".png")
        plt.show()
        plt.close(fig=fig)
        
    def fig11(self):
        #P(u^j)分布 -- 推荐&观测&初始化
        pic_index = 10
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        fig, axes = plt.subplots(nrows= 4, ncols=5, figsize=(25, 20))

        user_rec_matrix = self.re_3['user_rec_matrix']
        user_obs_matrix = self.re_3['user_obs_matrix']
        user_int_matrix = self.re_3['user_int_matrix']
        
        user_rec_matrix = user_rec_matrix/np.sum(user_rec_matrix, axis=1, keepdims=True)
        user_obs_matrix = user_obs_matrix/np.sum(user_obs_matrix, axis=1, keepdims=True)
        user_int_matrix = user_int_matrix/np.sum(user_int_matrix, axis=1, keepdims=True)

        
        #bins = np.arange(0, 1.01, 0.01)

        for i in range(user_rec_matrix.shape[1]):
            
            axes[i//5, i%5] = sns.histplot(data= user_rec_matrix[:, i], bins = 20, color = 'green', label = 'rec', kde = True, stat = 'probability' , ax = axes[i//5, i%5])
            #axes[i//5, i%5] = sns.histplot(data= user_int_matrix[:, i], bins = bins, color = 'blue', label = 'init', kde = True, stat = 'probability' , ax = axes[i//5, i%5])
            #axes[i//5, i%5] = sns.histplot(data= user_obs_matrix[:, i], bins = bins, color = 'orange', label = 'obs', kde = True, stat = 'probability' , ax = axes[i//5, i%5])
            
            axes[i//5, i%5].set_title(str(i))
            axes[i//5, i%5].set_yscale('log')
        
        #plt.savefig(self.pic_path+str(pic_index)+".png")
        plt.show()
        #plt.close(fig=fig)

    def fig12(self):
        #P(u^j)分布 -- 推荐&观测&初始化
        pic_index = 10
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        fig, axes = plt.subplots(nrows= 4, ncols=5, figsize=(25, 20))

        user_rec_matrix = self.re_3['user_rec_matrix']
        user_obs_matrix = self.re_3['user_obs_matrix']
        user_int_matrix = self.re_3['user_int_matrix']
        
        user_rec_matrix = user_rec_matrix/np.sum(user_rec_matrix, axis=1, keepdims=True)
        user_obs_matrix = user_obs_matrix/np.sum(user_obs_matrix, axis=1, keepdims=True)
        user_int_matrix = user_int_matrix/np.sum(user_int_matrix, axis=1, keepdims=True)

        
        #bins = np.arange(0, 1.01, 0.01)

        for i in range(user_rec_matrix.shape[1]):
            
            axes[i//5, i%5] = sns.histplot(data= user_rec_matrix[:, i], bins = 20, color = 'green', label = 'rec', kde = True, stat = 'probability' , ax = axes[i//5, i%5])
            #axes[i//5, i%5] = sns.histplot(data= user_int_matrix[:, i], bins = bins, color = 'blue', label = 'init', kde = True, stat = 'probability' , ax = axes[i//5, i%5])
            #axes[i//5, i%5] = sns.histplot(data= user_obs_matrix[:, i], bins = bins, color = 'orange', label = 'obs', kde = True, stat = 'probability' , ax = axes[i//5, i%5])
            
            axes[i//5, i%5].set_title(str(i))
            axes[i//5, i%5].set_xscale('log')
            axes[i//5, i%5].set_yscale('log')
        
        #plt.savefig(self.pic_path+str(pic_index)+".png")
        plt.show()
        #plt.close(fig=fig)

        




def get_pics(para_basic):
    path = para_basic['results_path']
    with open(path+'init_users.pkl', 'rb') as f:
         init_users = pickle.load(f) 
    user_int_matrix = np.vstack([np.around(u.interest_int, 5) for u in init_users])

    # 动态情况情况分析
    all_time_step_result = []
    step = 10
    for i in range(0, para_basic['num_iter'], step):

        user_obs_matrix = np.zeros((para_basic['num_users'], para_basic['num_topics']))
        user_rec_matrix = np.zeros((para_basic['num_users'], para_basic['num_topics']))
        user_pos_matrix = np.zeros((para_basic['num_users'], para_basic['num_topics']))
        user_neg_matrix = np.zeros((para_basic['num_users'], para_basic['num_topics']))
        
        user_pos_cnt = np.zeros((para_basic['num_users']))
        user_neg_cnt = np.zeros((para_basic['num_users']))
        

        for j in range(step):
            with open(path+'run'+str(i+j)+'.pkl', 'rb') as f:
                current_users = pickle.load(f) 
            user_obs_matrix += np.vstack([np.around(u['interest_obs'], 5) for u in current_users])
            user_rec_matrix += np.vstack([np.sum(u['item2rec'], axis=0) for u in current_users])
            
            
            user_feedback_idx =  np.vstack([u['feedback'] for u in current_users])
            
            user_pos_cnt += np.sum((user_feedback_idx==1), axis=1)
            user_neg_cnt += np.sum((user_feedback_idx==0), axis=1)
            
            user_pos_matrix += np.vstack([np.sum(u['item2rec'][np.where(u['feedback']==1)[1], :], axis=0) for u in current_users])
            user_neg_matrix += np.vstack([np.sum(u['item2rec'][np.where(u['feedback']==0)[1], :], axis=0) for u in current_users])
        
        user_obs_matrix = user_obs_matrix/step
        user_rec_matrix = user_rec_matrix/step
        user_pos_matrix = user_pos_matrix/step
        user_neg_matrix = user_neg_matrix/step
        
        re_1 = calculate_int_obs_metrics(user_int_matrix, user_obs_matrix, user_rec_matrix, user_pos_matrix)
        re_2 = calculate_feedback(user_pos_cnt, user_neg_cnt, user_pos_matrix, user_neg_matrix)
        re = {**re_1, **re_2}
        all_time_step_result.append(re)
        
    # 稳态解情况分析--各个topic的分布情况
    user_obs_matrix = np.zeros((para_basic['num_users'], para_basic['num_topics']))
    user_rec_matrix = np.zeros((para_basic['num_users'], para_basic['num_topics']))
    for i in range(para_basic['num_iter']-step, para_basic['num_iter']):
        with open(path+'run'+str(i)+'.pkl', 'rb') as f:
            current_users = pickle.load(f)
        user_obs_matrix += np.vstack([np.around(u['interest_obs'], 5) for u in current_users])
        user_rec_matrix += np.vstack([np.sum(u['item2rec'], axis=0)/np.sum(u['item2rec']) for u in current_users])
    user_obs_matrix = user_obs_matrix/step
    user_rec_matrix = user_rec_matrix/step
    re_3 = {'user_int_matrix': user_int_matrix, 'user_obs_matrix': user_obs_matrix, 'user_rec_matrix': user_rec_matrix}
        
    # 储存计算结果
    
    pic_name, pic_path, tmp_path = generate_name(para_basic)
    
    with open(tmp_path+'dynamic_v1.pkl','wb') as f:
        pickle.dump(all_time_step_result, f)
    with open(tmp_path+'static_v1.pkl', 'wb') as f:
        pickle.dump(re_3, f)
    
    # 画图
    # pic_index: 1 总图
    pic_index = 0
    fig = plt.figure(figsize=(10,10))
    
    ax0 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
    ax2 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
    ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=1)
    ax4 = plt.subplot2grid((3, 2), (2, 1), colspan=1)
    
    # axe0
    tmp1 = []
    tmp2 = []
    tmp3 = []
    for i in range(len(all_time_step_result)):
        tmp1.append(np.nanmean(all_time_step_result[i]['entropy_diff_1']))
        tmp2.append(np.nanmean(all_time_step_result[i]['entropy_diff_2']))
        tmp3.append(np.nanmean(all_time_step_result[i]['entropy_diff_3']))
    ax0.plot(tmp1, label='obs/intr')
    ax0.plot(tmp2, label='rec/intr')
    ax0.plot(tmp3, label='pos/intr')
    ax0.hlines(y=1, xmin=0, xmax=len(all_time_step_result), color='r', linestyle='--', label='equal')
    ax0.legend()
    ax0.set_title(pic_name)
    
    #axes1
    tmp1 = []
    tmp2 = []
    #tmp3 = []
    for i in range(len(all_time_step_result)):
        tmp1.append(np.nanmean(all_time_step_result[i]['rec_oe_ratio']))
        tmp2.append(np.nanmean(all_time_step_result[i]['rec_us_ratio']))
        #tmp3.append(np.nanmean(all_time_step_result[i]['mis_ratio']))
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
    for i in range(len(all_time_step_result)):
        tmp1.append(np.nanmean(all_time_step_result[i]['rec_oe_degree']))
        tmp2.append(np.nanmean(all_time_step_result[i]['rec_us_degree']))
        #tmp3.append(np.nanmean(all_time_step_result[i]['ab_degree']))
    ax2.plot(tmp1, label='Over Exploit')
    ax2.plot(tmp2, label='Under Serve')
    #ax2.plot(tmp3, color='green', linestyle='--',label='All')
    #ax2.set_ylim(0,1.1)
    ax2.legend()
    ax2.set_title("Bias")
    
    #axes3 hit
    tmp1 = []
    for i in range(len(all_time_step_result)):
        tmp1.append(np.nanmean(all_time_step_result[i]['hit_rate']))
    ax3.plot(tmp1)
    ax3.set_ylim(0,1)
    ax3.set_title("Hit Rate")
    
    
    #axes4 entropy_pos vs entropy_neg
    tmp1 = []
    tmp2 = []
    for i in range(len(all_time_step_result)):
        tmp1.append(np.nanmean(all_time_step_result[i]['entropy_pos']))
        tmp2.append(np.nanmean(all_time_step_result[i]['entropy_neg']))
    ax4.plot(tmp1, color='r', label = 'Pos')
    ax4.plot(tmp2, color='blue',label = 'Neg')
    ax4.set_title("Entropy:pos vs neg")
    ax4.legend()
    
    plt.savefig(pic_path+str(pic_index)+".png")
    plt.show()
    plt.close(fig=fig)
    
    # pic_index: 1 多样性
    pic_index = 1
    fig = plt.figure(figsize=(5, 5))
    
    tmp1 = []
    tmp2 = []
    tmp3 = []
    for i in range(len(all_time_step_result)):
        tmp1.append(np.nanmean(all_time_step_result[i]['entropy_obs']))
        tmp2.append(np.nanmean(all_time_step_result[i]['entropy_rec']))
        tmp3.append(np.nanmean(all_time_step_result[i]['entropy_int']))
        
    plt.plot(tmp1, label='Observed')
    plt.plot(tmp2, label='Recommended')
    plt.plot(tmp3, color='r', linestyle='--', label='Intrinsic')
    plt.ylabel('Entropy')
    plt.xlabel('Time')
    plt.legend()
    plt.title(pic_name)
    
    plt.savefig(pic_path+str(pic_index)+".png")
    plt.show()
    plt.close(fig=fig)
    
    # pic_index: 2 多样性
    pic_index = 2
    fig = plt.figure(figsize=(5, 5))
    
    tmp1 = []
    tmp3 = []
    for i in range(len(all_time_step_result)):
        tmp1.append(np.nanmean(all_time_step_result[i]['entropy_pos']))
        tmp3.append(np.nanmean(all_time_step_result[i]['entropy_int']))
    plt.plot(tmp1, label='Watched')
    plt.plot(tmp3, color='r', linestyle='--', label='Intrinsic')
    plt.ylabel('Entropy')
    plt.xlabel('Time')
    plt.legend()
    plt.title(pic_name)
    
    plt.savefig(pic_path+str(pic_index)+".png")
    plt.show()
    plt.close(fig=fig)
    
    # pic_index: 3 10步1测量 覆盖性
    pic_index = 3
    fig = plt.figure(figsize=(5, 5))
    
    tmp1 = []
    tmp2 = []
    tmp3 = []
    for i in range(len(all_time_step_result)):
        tmp1.append(np.nanmean(all_time_step_result[i]['coverage_obs']))
        tmp2.append(np.nanmean(all_time_step_result[i]['coverage_rec']))
        tmp3.append(np.nanmean(all_time_step_result[i]['coverage_int']))
        
    plt.plot(tmp1, label='Observed')
    plt.plot(tmp2, label='Recommended')
    plt.plot(tmp3, color='r', linestyle='--', label='Intrinsic')
    plt.ylabel('Coverage')
    plt.xlabel('Time')
    plt.legend()
    plt.title(pic_name)
    
    plt.savefig(pic_path+str(pic_index)+".png")
    plt.show()
    plt.close(fig=fig)
    
    # pic_index: 4 覆盖性
    pic_index = 4
    fig = plt.figure(figsize=(5, 5))
    
    tmp1 = []
    tmp3 = []
    for i in range(len(all_time_step_result)):
        tmp1.append(np.nanmean(all_time_step_result[i]['coverage_pos']))
        tmp3.append(np.nanmean(all_time_step_result[i]['coverage_int']))
    plt.plot(tmp1, label='Watched')
    plt.plot(tmp3, color='r', linestyle='--', label='Intrinsic')
    plt.ylabel('Coverage')
    plt.xlabel('Time')
    plt.legend()
    plt.title(pic_name)
    
    plt.savefig(pic_path+str(pic_index)+".png")
    plt.show()
    plt.close(fig=fig)
    
    #pic_index: 5 稳态时各个topic的分布
    pic_index = 5
    
    if not os.path.exists(pic_path+str(pic_index)):
        os.mkdir(pic_path+str(pic_index))
    

    #print("int", np.sum(re_3['user_int_matrix']<0))
    #print("obs", np.sum(re_3['user_obs_matrix']<0))
    #print("rec", np.sum(re_3['user_obs_matrix']<0))
    
    for i in range(para_basic['num_topics']):
    
        
        fig = plt.figure(figsize=(5, 5))
        
        bins_max = np.max(np.concatenate([re_3['user_int_matrix'][:, i], re_3['user_obs_matrix'][:, i], re_3['user_rec_matrix'][:, i]]))
        bins = np.linspace(0, bins_max, 10)
        n, bins = np.histogram(re_3['user_int_matrix'][:, i], bins=bins)
        
        show_xticks = (bins[:-1]+bins[1:])/2 
        width = (bins[1]-bins[0])*0.75
        plt.bar(x= show_xticks, height=n/np.sum(n), color='blue', width = width, alpha=0.25)
        plt.plot(show_xticks, n/np.sum(n), color='blue', linestyle='--', label='Intrinsic')
        
        n, bins = np.histogram(re_3['user_obs_matrix'][:, i], bins=bins)
        plt.bar(x=show_xticks, height=n/np.sum(n), color='orange', width = width,  alpha=0.25)
        plt.plot(show_xticks, n/np.sum(n), color='orange', linestyle=':', label='Observed')
        
        n, bins = np.histogram(re_3['user_rec_matrix'][:, i], bins=bins)
        plt.bar(x=show_xticks, height=n/np.sum(n), color='green', width = width, alpha=0.25)
        plt.plot(show_xticks, n/np.sum(n), color='green', linestyle='-.', label='Recommended')
        
        #plt.xticks(np.arange(0, 120, 20), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylabel('Distribution')
        plt.xlabel('Value')
        plt.legend()
        plt.title(pic_name)
        plt.ylim
    
        plt.savefig(pic_path+str(pic_index)+"/"+str(i)+".png")
        plt.show()
        plt.close(fig=fig)

    #pic_index: 6 稳态时各个topic的分布
    pic_index = 6
    
    if not os.path.exists(pic_path+str(pic_index)):
        os.mkdir(pic_path+str(pic_index))
    

    #print("int", np.sum(re_3['user_int_matrix']<0))
    #print("obs", np.sum(re_3['user_obs_matrix']<0))
    #print("rec", np.sum(re_3['user_obs_matrix']<0))
    
    for i in range(para_basic['num_topics']):
    
        
        fig = plt.figure(figsize=(5, 5))
        
        bins_max = np.max(np.concatenate([re_3['user_int_matrix'][:, i], re_3['user_obs_matrix'][:, i], re_3['user_rec_matrix'][:, i]]))
        bins = np.linspace(0, bins_max, 100)
        n, bins = np.histogram(re_3['user_int_matrix'][:, i], bins=bins)
        
        show_xticks = (bins[:-1]+bins[1:])/2
        #plt.bar(x= show_xticks, height=n/np.sum(n), color='blue', width= 0.01, alpha=0.25)
        plt.plot(show_xticks, n/np.sum(n), color='blue', linestyle='--', marker='v', label='Intrinsic')
        
        n, bins = np.histogram(re_3['user_obs_matrix'][:, i], bins=bins)
        #plt.bar(x=show_xticks, height=n/np.sum(n), color='orange', width= 0.01 , alpha=0.25)
        plt.plot(show_xticks, n/np.sum(n), color='orange', linestyle=':', marker='o' ,label='Observed')
        
        n, bins = np.histogram(re_3['user_rec_matrix'][:, i], bins=bins)
        #plt.bar(x=show_xticks, height=n/np.sum(n), color='green', width= 0.01, alpha=0.25)
        plt.plot(show_xticks, n/np.sum(n), color='green', linestyle='-.',  marker='x',  label='Recommended')
        
        #plt.xticks(np.arange(0, 120, 20), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Distribution')
        plt.xlabel('Value')
        plt.legend()
        plt.title(pic_name)
        plt.ylim
    
        plt.savefig(pic_path+str(pic_index)+"/"+str(i)+".png")
        plt.show()
        plt.close(fig=fig)


    
    
    
    
    
    

