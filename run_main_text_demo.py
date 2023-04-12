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


setproctitle.setproctitle("Adaptive_Information_Dynamics_{}".format('video datasets'))
set_rand_seed(4)


para_search = []

sigma_list = [0.01]
beta_list = [0,4,7]
gamma_minus_list = [-0.1]
gamma_plus_list = [1.0]

for s in sigma_list:
    for b in beta_list:
        for gp in gamma_plus_list:
            for gm in gamma_minus_list:

                para_tmp = {}
                            
                para_tmp['alpha'] = 0.00
                para_tmp['beta'] = b
                para_tmp['gamma_plus'], para_tmp['gamma_minus'] = (gp, gm)
                para_tmp['sigma'] = s
                
                para_tmp['num_iter'] = 5000
                para_tmp['num_users'] = 1000
                para_tmp = generate_parameters(**para_tmp)
                
                para_search.append(para_tmp)


sigma_list = [0.01]
beta_list = [7]
gamma_minus_list = [-0.1]
gamma_plus_list = [0.0, 0.3, 1.0]

for s in sigma_list:
    for b in beta_list:
        for gp in gamma_plus_list:
            for gm in gamma_minus_list:

                para_tmp = {}
                            
                para_tmp['alpha'] = 0.00
                para_tmp['beta'] = b
                para_tmp['gamma_plus'], para_tmp['gamma_minus'] = (gp, gm)
                para_tmp['sigma'] = s
                
                para_tmp['num_iter'] = 5000
                para_tmp['num_users'] = 1000
                para_tmp = generate_parameters(**para_tmp)
                
                para_search.append(para_tmp)

sigma_list = [0.01]
beta_list = [7]
gamma_minus_list = [-0.1, -0.3, -0.9]
gamma_plus_list = [1.0]

for s in sigma_list:
    for b in beta_list:
        for gp in gamma_plus_list:
            for gm in gamma_minus_list:

                para_tmp = {}
                            
                para_tmp['alpha'] = 0.00
                para_tmp['beta'] = b
                para_tmp['gamma_plus'], para_tmp['gamma_minus'] = (gp, gm)
                para_tmp['sigma'] = s
                
                para_tmp['num_iter'] = 5000
                para_tmp['num_users'] = 1000
                para_tmp = generate_parameters(**para_tmp)
                
                para_search.append(para_tmp)


sigma_list = [0.04, 0.07, 0.16]
beta_list = [10]
gamma_minus_list = [-0.1]
gamma_plus_list = [1.0]

for s in sigma_list:
    for b in beta_list:
        for gp in gamma_plus_list:
            for gm in gamma_minus_list:

                para_tmp = {}
                            
                para_tmp['alpha'] = 0.00
                para_tmp['beta'] = b
                para_tmp['gamma_plus'], para_tmp['gamma_minus'] = (gp, gm)
                para_tmp['sigma'] = s
                
                para_tmp['num_iter'] = 5000
                para_tmp['num_users'] = 1000
                para_tmp = generate_parameters(**para_tmp)
                
                para_search.append(para_tmp)
       
       
       

for para_basic in tqdm(para_search):
    print('---- a_{:.2f}_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.2f} ----'.format(para_basic['alpha'], para_basic['beta'], para_basic['gamma_plus'], para_basic['gamma_minus'], para_basic['sigma']))
    run(para_basic, True, True, False, True, True)

