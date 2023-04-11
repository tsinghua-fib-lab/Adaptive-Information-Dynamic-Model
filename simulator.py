# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:57:50 2022

@author: XPS
"""

from math import gamma
import numpy as np
import pickle as pkl
import scipy.stats as st
from collections import defaultdict
from copy import copy
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os
import time
import pandas as pd
import pickle
import copy


np.random.seed(4)

def sigmoid(x):
    x = np.array(x)
    return 1/(1+np.exp(-x))


def scores(user_vec, items_vec, phi, sim_type):
    if sim_type == 'dot':
        user_vec = user_vec.reshape(1, -1) #(1, |T|)
        items_vec = items_vec.T # (|T|, N_item)
        return np.dot(np.dot(user_vec, phi), items_vec)
    elif sim_type == 'cosine':
        user_vec = user_vec.reshape(1, -1) #(1, |T|)
        items_vec = items_vec # (N_item, |T|)
        return cosine_similarity(user_vec, np.dot(phi, items_vec)) #(1, N_item)
    elif sim_type == 'euclidean':
        user_vec = user_vec.reshape(1, -1) #(1, |T|)
        items_vec = items_vec # (N_item, |T|)
        return 1/(1+euclidean_distances(user_vec, np.dot(phi, items_vec))) #(1, N_item)        
    
def scores2recp(beta, scores):
    tmp = np.exp(scores*beta)
    tmp = tmp/np.sum(tmp)
    tmp = tmp.squeeze()
    
    #if np.isnan(tmp).any():
    #    print(scores)
    return tmp

def user_scores(user_vec, items_vec, sim_type):
    if sim_type == 'dot':
        user_vec = user_vec.reshape(1, -1) #(1, |T|)
        items_vec = items_vec.T # (|T|, N_item)
        score = np.dot(user_vec, items_vec)
        
        #return np.dot(user_vec, items_vec)
    elif sim_type == 'cosine':
        user_vec = user_vec.reshape(1, -1) #(1, |T|)
        items_vec = items_vec # (N_item, |T|)
        score = cosine_similarity(user_vec, items_vec)
        
        
    elif sim_type == 'euclidean':
        user_vec = user_vec.reshape(1, -1) #(1, |T|)
        items_vec = items_vec # (N_item, |T|)
        score = 1/(1+euclidean_distances(user_vec, items_vec)) 
        
    return score  #(1, N_item)


class item:
    def __init__(self, name, feature):
        self.name = name
        self.feature = feature

class user:
    def __init__(self, name, alpha, interest_int, interest_obs, sim_type):
        self.name = name
        self.interest_int, self.interest_obs = interest_int, interest_obs
        self.alpha = alpha
        
        self.sim_type = sim_type
        
    def feedback(self, item2rec, scores2rec): 
        int_scores = user_scores(user_vec = self.interest_int, items_vec = item2rec, sim_type=self.sim_type)
        
        pi =  self.alpha*scores2rec+(1-self.alpha)*int_scores
        #print(pi)
        #pi =  self.alpha*scores2rec/np.max(scores2rec)+(1-self.alpha)*int_scores/np.max(int_scores)
        y = np.random.rand(item2rec.shape[0])<pi
        return y
    

class recommender:
    def __init__(self, item_groups, phi, num_rec, beta, sim_type):
        self.items_vec = np.array([itm.feature for itm in item_groups])
        self.phi = phi
        self.num_rec = num_rec
        self.beta = beta
        self.sim_type = sim_type
        
    def recommend(self, user2rec):
        user_vec = user2rec.interest_obs
        
        # (1) scores based on similarity
        sim_tmp = scores(user_vec = user_vec, items_vec = self.items_vec, phi=self.phi, sim_type=self.sim_type)
        
        # (2) recommend based on scores
        p_tmp = scores2recp(beta=self.beta, scores=sim_tmp)
        
        #if np.isnan(p_tmp).any():
        #    print(user_vec)
        # (3) k items are selected
        item2rec = np.random.choice(a=np.arange(p_tmp.shape[0]), size=self.num_rec, p=p_tmp, replace=False)
        scores2rec = np.squeeze(sim_tmp)
        scores2rec = scores2rec[item2rec]
        item2rec = self.items_vec[item2rec,:]
        
        
        return item2rec, scores2rec
    
        

class simulator:
    def __init__(self, num_users, num_items, num_topics, num_iter, num_rec,
                 phi_path, item_path, dt, sim_type,
                 alpha, beta, gamma_plus, gamma_minus, sigma, **kwargs):
        self.num_users = num_users
        self.num_items = num_items
        self.num_topics = num_topics
        self.num_iter = num_iter
        self.num_rec = num_rec
        
        
        self.phi_path = phi_path
        self.item_path = item_path
        
        
        self.dt = dt
            
        self.sim_type = sim_type
        
        self.alpha = alpha
        self.beta = beta
        self.gamma_plus = gamma_plus
        self.gamma_minus = gamma_minus
        self.sigma = sigma
        
        
        self.other_para = kwargs
        
        
    
        self.init_parameters()
        self.init_users()
        self.init_items()
        
        self.recsys = recommender(self.item_groups, self.phi, self.num_rec, self.beta, self.sim_type)
        
        
    def init_parameters(self):
        if self.other_para['user_method'] == 'simulated':
            mu_users = np.random.dirichlet(alpha=np.ones(self.num_topics))*10
            #mu_items = np.random.dirichlet(alpha=np.ones(self.num_topics*100))*0.1
        
            self.mu_users = mu_users
            #self.mu_items = mu_items
        
        elif self.other_para['user_method'] == 'real':
            self.mu_users = np.load(self.other_para['mu_user_path'])*15

        # 选择特定的topics   
        elif self.other_para['user_method'] == 'real_selected':
            self.mu_users = np.load(self.other_para['mu_user_path'])*15
            
    def init_users(self):
        self.user_groups = []
        i = 0
        while i < self.num_users:
        #for i in range(self.num_users):
            interet_int_tmp = np.random.dirichlet(alpha= self.mu_users)
            interet_obs_tmp = np.random.dirichlet(alpha= self.mu_users)
            
            if self.other_para['user_method'] == 'real_selected':
                selected_category = self.other_para['selected_category']
                interet_int_tmp = interet_int_tmp[selected_category]
                interet_obs_tmp = interet_obs_tmp[selected_category]
                if (np.sum(interet_int_tmp)==0) or (np.sum(interet_obs_tmp)==0):
                    continue
                else:
                    interet_int_tmp = interet_int_tmp/np.sum(interet_int_tmp)
                    interet_obs_tmp = interet_obs_tmp/np.sum(interet_obs_tmp)
            
            self.user_groups.append(user(name=i, interest_int=interet_int_tmp, interest_obs=interet_obs_tmp, alpha= self.alpha,sim_type=self.sim_type))
            
            i += 1
        
        with open(self.other_para['results_path']+'init_users.pkl', 'wb') as f:
            pickle.dump(self.user_groups, f)
        
        
    def init_items(self):
        #self.item_groups = []
        #for i in range(self.num_items):
        #    feature_tmp = np.random.dirichlet(alpha= self.mu_items)
        #    self.item_groups.append(item(name=i, features = feature_tmp))
        
        # phi
        self.phi = np.load(self.phi_path)
        
        # 选择特定的topics
        if self.other_para['user_method'] == 'real_selected':
            selected_category = self.other_para['selected_category']
            #print(selected_category)
            self.phi = self.phi[selected_category, :]
            self.phi = self.phi[:, selected_category]
            #print(self.phi)
        # items
        tmp = np.load(self.item_path)
        
        # 选择特定的topics
        if self.other_para['user_method'] == 'real_selected':
            selected_category = self.other_para['selected_category']
            tmp = tmp[selected_category, :]
        
        item_idxs = np.random.choice(a=np.arange(tmp.shape[0]), size=self.num_items, p=tmp[:,1]/np.sum(tmp[:,1]), replace=True)
        self.item_groups = [] 
        for i, item_idx in enumerate(item_idxs):
            feature_tmp = np.zeros(tmp.shape[0])
            feature_tmp[item_idx] = 1.0
            self.item_groups.append(item(name=i, feature = feature_tmp))
        
            

    def tracer(self, user_group_tracer):        
        with open(self.other_para['results_path']+'run'+str(self.ts)+'.pkl', 'wb') as f:
            pickle.dump(user_group_tracer, f)
        
    def interaction(self):
        user_group_tracer = []
        for u in self.user_groups:
            # 推荐系统进行推荐
            item2rec, scores2rec = self.recsys.recommend(u)
            # 用户进行反馈
            y = u.feedback(item2rec, scores2rec)
            y_pos_idx = np.where(y==1)[1]
            y_neg_idx = np.where(y==0)[1]
            #print(y_pos_idx, y_neg_idx)
            # 更新用户向量
            topic_idx = (item2rec>0).astype('float')
            tmp = (item2rec - u.interest_obs) * topic_idx
  

            
            pos_term = (self.gamma_plus*np.sum(tmp[y_pos_idx, :], axis=0)*self.dt) if y_pos_idx.shape[0]!=0 else np.zeros_like(u.interest_obs)
            neg_term = (self.gamma_minus*np.sum(tmp[y_neg_idx, :], axis=0)*self.dt) if y_neg_idx.shape[0]!=0 else np.zeros_like(u.interest_obs)
            random_term = self.sigma*np.random.normal(size= u.interest_obs.shape[0], loc=0, scale = np.sqrt(self.dt))
            
    
            
            u.interest_obs = u.interest_obs + pos_term + neg_term + random_term
            #print("a",u.interest_obs)   
            u.interest_obs = np.clip(u.interest_obs, 0, 1)
            #print("b",u.interest_obs)
            u.interest_obs = u.interest_obs/np.sum(u.interest_obs)
            #print("c",u.interest_obs)
            #if np.sum(np.isnan(u.interest_obs))>0:
            #    print("d",u.interest_obs)

            
            user_group_tracer.append({'item2rec': item2rec, 'feedback': y, 'interest_obs': u.interest_obs})
        
        self.tracer(user_group_tracer = user_group_tracer)
    def __iter__(self):
        self.ts = 0
        self.time_start = time.time()
        return self
        
    def __next__(self):
        if self.ts < self.num_iter:
            #print("--- Run: {} --- Total Time: {} ---".format(self.ts, time.time()-self.time_start))
            self.interaction()
            self.ts = self.ts+1
            return self
        else:
            raise StopIteration