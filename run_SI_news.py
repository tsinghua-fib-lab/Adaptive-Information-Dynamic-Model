# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:03:12 2022

@author: XPS
"""
from math import gamma
import numpy as np
import scipy as sp
import pickle
import matplotlib.pyplot as plt
import os 
import copy
from tqdm import tqdm
from utils import *
import setproctitle

part_num = 0
setproctitle.setproctitle("Adaptive_Information_Dynamics_{}".format('news datasets'))

set_rand_seed(4)

para_search = []

sigma_list = np.arange(0,0.11,0.01)
beta_list = np.arange(0,11,1)
gamma_minus_list = np.round(-np.arange(0.0, 1.1, 0.1), 2)
gamma_plus_list = np.round(np.arange(0.0, 1.1, 0.1), 2)

for s in sigma_list:
    for b in beta_list:
        for gp in gamma_plus_list:
            for gm in gamma_minus_list:

                para_tmp = {}
                            
                para_tmp['alpha'] = 0.00
                para_tmp['beta'] = b
                para_tmp['gamma_plus'], para_tmp['gamma_minus'] = (gp, gm)
                para_tmp['sigma'] = s
                para_tmp['sim_type'] = 'dot' 

                para_tmp['num_topics'] = 14
                para_tmp['phi_path'] = './Data/phi_news.npy'
                para_tmp['item_path'] = './Data/item_popularity_news.npy'
                para_tmp['mu_user_path'] = './Data/user_alpha_mean_news.npy' 

                para_tmp['num_iter'] = 5000
                para_tmp['num_users'] = 1000
                para_tmp = generate_parameters(**para_tmp)
                
                para_search.append(para_tmp)


for para_basic in tqdm(para_search):
    print('---- a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.2f} ----'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma']))
    # pic, zip, mv, optimize, results
    run(para_basic, True, True, False, True, True)

