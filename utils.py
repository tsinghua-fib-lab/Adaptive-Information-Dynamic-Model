# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:54:52 2022

@author: XPS
"""
import os
import numpy as np
from draw_pic import *


# 实验记录路径
record_path = r'./exp_record.txt'


# 生成带有路径的参数集合
def generate_parameters(**kwargs):
    
    para_basic = {'num_users': 100,
              'num_items': 10000,
              'num_topics': 20,
              'num_iter': 1000,
              'num_rec': 10, 
              'phi_path': './Data/phi_v1.npy',
              'item_path': './Data/item_popularity_v1.npy',
              'dt':0.01,
              'sim_type': 'dot',       
              'user_method': 'real',
              'mu_user_path': './Data/user_alpha_mean_v1.npy'
              }
    
    for k, v in kwargs.items():
        para_basic[k] = v

    results_path = r'./results/'+r'a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.2f}/'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])
    para_basic['results_path'] = results_path
    
    if 'fined_sigma' in kwargs.keys():
        results_path = r'./results/'+r'a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.3f}/'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma'])
        para_basic['results_path'] = results_path
        print(para_basic['results_path'])
    
    return para_basic

def get_result_zipped_mv(para_basic):
    dir_name = para_basic['results_path'][:-1]
    
    cmd_1 = "tar -zvcf "+str(dir_name)+".tar.gz "+str(dir_name)+"/"
    os.system(cmd_1)
    
    cmd_2 = "mv "+str(dir_name)+".tar.gz /data/piaojinghua/results/"
    os.system(cmd_2)
    
    cmd_3 = "rm -rf "+str(dir_name)
    os.system(cmd_3)

def get_result_zipped(para_basic):
    dir_name = para_basic['results_path'][:-1]
    
    cmd_1 = "tar -zvcf "+str(dir_name)+".tar.gz "+str(dir_name)+"/"
    os.system(cmd_1)
    
    cmd_3 = "rm -rf "+str(dir_name)+"/"
    os.system(cmd_3)    

def get_result_deleted(para_basic):
    dir_name = para_basic['results_path'][:-1]
    
    #cmd_1 = "tar -zvcf "+str(dir_name)+".tar.gz "+str(dir_name)+"/"
    #os.system(cmd_1)
    
    cmd_3 = "rm -rf "+str(dir_name)+"/"
    os.system(cmd_3)   

def get_unzipped(para_basic):

    zipped_name = '/data/piaojinghua/results/'+para_basic['results_path'].split('/')[2]+'.tar.gz'
    cmd_1 = "tar -xzf "+zipped_name
    print(os.system(cmd_1))
    print(zipped_name)

def get_zipped(para_basic):

    unzipped_name = '/data/piaojinghua/results/'+para_basic['results_path'].split('/')[2]

    cmd_1 = "tar -zvcf "+str(unzipped_name)+".tar.gz "+str(unzipped_name)+"/"
    os.system(cmd_1)

def get_rm(para_basic):

    unzipped_name = '/data/piaojinghua/results/'+para_basic['results_path'].split('/')[2]

    cmd_1 = "rm -rf "+str(unzipped_name)+"/"
    os.system(cmd_1)


def set_rand_seed(num):
    rand_seed = num
    np.random.seed(rand_seed)
    
    
    
def run(para_basic, pic=True, zip=True, mv=True, optimized=False, no_results=False):
    set_rand_seed(4)
    
    if not os.path.exists(para_basic['results_path']):
        os.mkdir(para_basic['results_path'])
    
    if optimized:
        import simulator_optimized
        para_basic['optimized'] = optimized
        simulator_tmp = simulator_optimized.simulator(**para_basic)
        print("Optimized!")
    else:
        import simulator
        simulator_tmp = simulator.simulator(**para_basic)
        
    with open('./exp_record.txt', mode='r') as f:
        lines = f.readlines()
        if len(lines) > 0:
            lines = lines[-1]
            lines = lines.split("|")
            idx_last = eval(lines[0])
        else:
            idx_last = -1
    
    with open('./exp_record.txt', mode='a') as f:
        f.write(str(idx_last+1)+'|'+str(para_basic)+'\n')
    
    for sm in iter(simulator_tmp):
        pass
    
    if pic:
        #get_pics(para_basic)
        pics_tmp = pics(para_basic=para_basic, mode='raw', step=10)
        pics_tmp.run_pics()
    
    if no_results:
        get_result_deleted(para_basic)
    else:
        if zip:
            if mv:
                get_result_zipped_mv(para_basic)
            else:
                get_result_zipped(para_basic)
    
    return simulator_tmp

def run_simulated(para_basic, pic=True, step=10):
    get_unzipped(para_basic)
    pics_tmp = pics(para_basic=para_basic, mode='raw', step=step)
    pics_tmp.run_pics()
    get_rm(para_basic)

def run_only_pics(para_basic, pic=True, step=10):
    get_unzipped(para_basic)
    pics_tmp = pics(para_basic=para_basic, mode='processed', step=step)
    pics_tmp.run_pics()
    get_rm(para_basic)

