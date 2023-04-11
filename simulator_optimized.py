# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:57:50 2022

@author: XPS
"""

from math import gamma
from turtle import pos
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
from multiprocessing import Pool
from scipy.spatial.distance import jensenshannon

np.random.seed(4)

def sigmoid(x):
    x = np.array(x)
    return 1/(1+np.exp(-x))


def scores(user_vec, items_vec, phi, sim_type, method='single_user'):
    if method == 'single_user':
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
    elif method == 'batch_users':
        if sim_type == 'dot':
            user_vec = user_vec #(N_user, |T|)
            items_vec = items_vec.T #(|T|, N_item)
            return np.dot(np.dot(user_vec, phi), items_vec)
        elif sim_type == 'cosine':
            user_vec = user_vec #(N_user, |T|)
            items_vec = items_vec.T # (|T|, N_item)
            item_tmp = np.dot(phi, items_vec).T
            return cosine_similarity(user_vec, item_tmp) #(N_user, N_item)
        elif sim_type == 'euclidean':
            user_vec = user_vec #(N_user, |T|)
            items_vec = items_vec.T # (|T|, N_item)
            item_tmp = np.dot(phi, items_vec).T
            return 1/(1+euclidean_distances(user_vec, item_tmp)) #(N_user, N_item)
        elif sim_type == 'jsd': # jsd 1008
            user_vec = user_vec #(N_user, |T|)
            items_vec = items_vec.T # (|T|, N_item)
            item_tmp = np.dot(phi, items_vec).T # (N_item, |T|)
            jsd = jensenshannon(user_vec, item_tmp)
            # one vector is zero, jsd==nan --> jsd=1 very dissimilarity
            jsd[np.isnan(jsd)] = 1
            reverse_jsd = 1-jsd
            return reverse_jsd








    
def scores2recp(beta, scores, method='single_user'):
    if method == 'single_user':
        tmp = np.exp(scores*beta)
        tmp = tmp/np.sum(tmp)
        tmp = tmp.squeeze()
        return tmp
    elif method == 'batch_users':
        tmp = np.exp(scores*beta)
        tmp = tmp/np.sum(tmp, axis=1, keepdims=True)
        return tmp

def user_scores(user_vec, items_vec, sim_type, method='single_user'):
    if method == 'single_user':
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
    elif method == 'batch_users':

        if sim_type == 'dot':
            user_vec = user_vec #(N_users, |T|)
            items_vec = items_vec # (N_users, #rec, |T|) -> (|T|, N_rec)
            score = np.vstack([np.dot(user_vec[u_idx, :], np.squeeze(items_vec[u_idx, :, :].T)) for u_idx in range(user_vec.shape[0])])
            # print("4", score.shape)
            #return np.dot(user_vec, items_vec)
        elif sim_type == 'cosine':
            user_vec = user_vec #(N_users, |T|)
            items_vec = items_vec # (N_item, #rec, |T|)

            score = np.vstack([cosine_similarity(user_vec[[u_idx], :], np.squeeze(items_vec[u_idx, :, :])) for u_idx in range(user_vec.shape[0])])
            #score = cosine_similarity(user_vec, items_vec)
            
        elif sim_type == 'euclidean':
            user_vec = user_vec #(N_users, |T|)
            items_vec = items_vec # (N_item, |T|)
            
            score = np.vstack([1/(1+euclidean_distances(user_vec[[u_idx], :], np.squeeze(items_vec[u_idx, :, :]))) for u_idx in range(user_vec.shape[0])])
            #score = 1/(1+euclidean_distances(user_vec, items_vec)) 
        
        elif sim_type == 'jsd': # jsd 1008
            user_vec = user_vec #(N_users, |T|)
            items_vec = items_vec # (N_item, |T|)
            score = np.vstack([jensenshannon(user_vec[[u_idx], :], np.squeeze(items_vec[u_idx, :, :])) for u_idx in range(user_vec.shape[0])])
            score[np.isnan(score)] = 1
            score = 1 - score
        return score  #(N_users, N_item)
    


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
    
class users_groups:
    def __init__(self, alpha, interest_int, interest_obs, sim_type):
        self.interest_int, self.interest_obs = interest_int, interest_obs
        self.alpha = alpha
        self.sim_type = sim_type
        
    def feedback(self, item2rec, scores2rec): 
        #print("3", item2rec.shape)
        int_scores = user_scores(user_vec = self.interest_int, items_vec = item2rec, sim_type=self.sim_type, method='batch_users')
        
        pi =  self.alpha*scores2rec+(1-self.alpha)*int_scores
        #print(pi)
        #pi =  self.alpha*scores2rec/np.max(scores2rec)+(1-self.alpha)*int_scores/np.max(int_scores)
        y = np.random.rand(item2rec.shape[0], item2rec.shape[1])<pi
        return y


class recommender:
    def __init__(self, item_groups, phi, num_rec, beta, sim_type):
        self.items_vec = np.array([itm.feature for itm in item_groups])
        self.phi = phi
        self.num_rec = num_rec
        self.beta = beta
        self.sim_type = sim_type
        
    def recommend(self, user2rec, method='single_user'):

        if method == 'single_user':
            user_vec = user2rec.interest_obs
            
            # (1) scores based on similarity
            sim_tmp = scores(user_vec = user_vec, items_vec = self.items_vec, phi=self.phi, sim_type=self.sim_type)
            
            # (2) recommend based on scores
            p_tmp = scores2recp(beta=self.beta, scores=sim_tmp)
            
            # (3) k items are selected
            item2rec = np.random.choice(a=np.arange(p_tmp.shape[0]), size=self.num_rec, p=p_tmp, replace=False)
            scores2rec = np.squeeze(sim_tmp)
            scores2rec = scores2rec[item2rec]
            item2rec = self.items_vec[item2rec,:]
            return item2rec, scores2rec
        elif method == 'batch_users':
            user_vec = user2rec.interest_obs

            # (1) scores based on similarity
            sim_tmp = scores(user_vec = user_vec, items_vec = self.items_vec, phi=self.phi, sim_type=self.sim_type, method='batch_users')

            # (2) recommend based on scores
            p_tmp = scores2recp(beta=self.beta, scores=sim_tmp, method='batch_users')

            # (3) k items are selected
            scores2rec_batch = np.zeros((p_tmp.shape[0], self.num_rec)) #(#users, #rec)
            item2rec_batch = np.zeros((p_tmp.shape[0], self.num_rec, user_vec.shape[1])) #(#users, #rec, #topic)
            
            for user_idx in range(p_tmp.shape[0]):
                item2rec = np.random.choice(a=np.arange(p_tmp.shape[1]), size=self.num_rec, p=p_tmp[user_idx, :], replace=False)
                scores2rec = np.squeeze(sim_tmp[user_idx, item2rec])
                scores2rec_batch[user_idx, :] = scores2rec # (#rec)
                item2rec_batch[user_idx, :, :]= self.items_vec[item2rec, :] # (#rec, #topics)
                
            return item2rec_batch, scores2rec_batch
        
        
    
        

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
            
            self.user_groups.append(user(name=i, interest_int=interet_int_tmp, interest_obs=interet_obs_tmp, alpha=self.alpha, sim_type=self.sim_type))
            
            i += 1
        
        with open(self.other_para['results_path']+'init_users.pkl', 'wb') as f:
            pickle.dump(self.user_groups, f)

        
        
        # 优化为numpy
        if self.other_para['optimized']:
            #print("?")
            #print(self.other_para['optimized'])
            interet_int_tmp = np.vstack([u.interest_int for u in self.user_groups])
            interet_obs_tmp = np.vstack([u.interest_obs for u in self.user_groups])
            sim_type = self.sim_type
            alpha = self.alpha 

            self.user_groups = users_groups(alpha=alpha, interest_int=interet_int_tmp, interest_obs=interet_obs_tmp, sim_type=sim_type)

        if self.other_para['optimized'] > 1 :
            #print("!")
            self.user_group_split(self.other_para['optimized'])
        
     
        
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
        
        # # 优化为numpy
        # if self.other_para['optimized']:

        #     feature_tmp = np.vstack([u.feature for u in self.user_groups])
        #     self.item_groups = {}
        #     self.item_groups['feature'] = feature_tmp

    def interaction_mp(self, sub_idx):        
        user_group_tracer = []
        num_users = self.user_subgroups[sub_idx].interest_obs.shape[0]
        # 推荐系统进行推荐
        item2rec, scores2rec = self.recsys.recommend(self.user_subgroups[sub_idx], method='batch_users')
        #print("2", item2rec.shape)
        # 用户进行反馈
        y = self.user_subgroups[sub_idx].feedback(item2rec, scores2rec) #(#users, #rec)
        #print("5", y.shape)

        #item2rec (#users, #rec, #topic)
        topic_idx = (item2rec>0).astype('float')
        tmp = (item2rec - self.user_subgroups[sub_idx].interest_obs[:, np.newaxis, :]) * topic_idx
        
        for i in range(num_users):
            y_pos_idx = np.where(y[i, :]==1)[0]
            y_neg_idx = np.where(y[i, :]==0)[0]
            #print("6", y_pos_idx, y_neg_idx)

            pos_term = (self.gamma_plus*np.squeeze(np.sum(tmp[i, y_pos_idx, :], axis=0))*self.dt) if y_pos_idx.shape[0]!=0 else np.zeros_like(self.user_subgroups[sub_idx].interest_obs[0, :])
            neg_term = (self.gamma_minus*np.squeeze(np.sum(tmp[i, y_neg_idx, :], axis=0))*self.dt) if y_neg_idx.shape[0]!=0 else np.zeros_like(self.user_subgroups[sub_idx].interest_obs[0, :])
            random_term = self.sigma*np.random.normal(size = self.user_subgroups[sub_idx].interest_obs[0, :].shape[0], loc=0, scale = np.sqrt(self.dt))
            #print("7", pos_term.shape, neg_term.shape, random_term.shape, tmp[i, y_pos_idx, :].shape)
            self.user_subgroups[sub_idx].interest_obs[i, :] = self.user_subgroups[sub_idx].interest_obs[i, :] + pos_term + neg_term + random_term
            user_group_tracer.append({'item2rec': np.squeeze(item2rec[i, :, :]), 'feedback': y[i, :].reshape((1,-1)), 'interest_obs': self.user_subgroups[sub_idx].interest_obs[i, :]})
        self.user_subgroups[sub_idx].interest_obs = np.clip(self.user_subgroups[sub_idx].interest_obs, 0, 1)
        self.user_subgroups[sub_idx].interest_obs = self.user_subgroups[sub_idx].interest_obs/np.sum(self.user_subgroups[sub_idx].interest_obs, axis=1, keepdims=True)
        return user_group_tracer    

    def user_group_split(self, num):
        self.user_subgroups = []
        
        alpha = self.user_groups.alpha
        sim_type = self.user_groups.sim_type
        interest_int_list = np.split(self.user_groups.interest_int, num, axis=0)
        interest_obs_list = np.split(self.user_groups.interest_obs, num, axis=0)
        
        for i in range(num):
            self.user_subgroups.append(users_groups(alpha=alpha, interest_int=interest_int_list[i],\
            interest_obs=interest_obs_list[i], sim_type=sim_type))
        

    def tracer(self, user_group_tracer):        
        with open(self.other_para['results_path']+'run'+str(self.ts)+'.pkl', 'wb') as f:
            pickle.dump(user_group_tracer, f)
        
    def interaction(self):
        if not self.other_para['optimized']:
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
                u.interest_obs = np.clip(u.interest_obs, 0, 1)
                u.interest_obs = u.interest_obs/np.sum(u.interest_obs)

                user_group_tracer.append({'item2rec': item2rec, 'feedback': y, 'interest_obs': u.interest_obs})
            self.tracer(user_group_tracer = user_group_tracer)
        
        elif self.other_para['optimized'] > 1:
            num = self.other_para['optimized']

            user_group_tracer = []

            with Pool(num) as p:
                results = p.map(self.interaction_mp, range(num))
            
            for r in results:
                user_group_tracer = user_group_tracer+r #tracer
            
            self.tracer(user_group_tracer = user_group_tracer)
        
        elif self.other_para['optimized']:
            user_group_tracer = []

            # 推荐系统进行推荐
            item2rec, scores2rec = self.recsys.recommend(self.user_groups, method='batch_users')
            #print("2", item2rec.shape)
            # 用户进行反馈
            y = self.user_groups.feedback(item2rec, scores2rec) #(#users, #rec)
            #print("5", y.shape)

            #item2rec (#users, #rec, #topic)
            topic_idx = (item2rec>0).astype('float')
            tmp = (item2rec - self.user_groups.interest_obs[:, np.newaxis, :]) * topic_idx
            for i in range(self.num_users):
                y_pos_idx = np.where(y[i, :]==1)[0]
                y_neg_idx = np.where(y[i, :]==0)[0]
                #print("6", y_pos_idx, y_neg_idx)

                pos_term = (self.gamma_plus*np.squeeze(np.sum(tmp[i, y_pos_idx, :], axis=0))*self.dt) if y_pos_idx.shape[0]!=0 else np.zeros_like(self.user_groups.interest_obs[0, :])
                neg_term = (self.gamma_minus*np.squeeze(np.sum(tmp[i, y_neg_idx, :], axis=0))*self.dt) if y_neg_idx.shape[0]!=0 else np.zeros_like(self.user_groups.interest_obs[0, :])
                random_term = self.sigma*np.random.normal(size = self.user_groups.interest_obs[0, :].shape[0], loc=0, scale = np.sqrt(self.dt))
                #print("7", pos_term.shape, neg_term.shape, random_term.shape, tmp[i, y_pos_idx, :].shape)
                self.user_groups.interest_obs[i, :] = self.user_groups.interest_obs[i, :] + pos_term + neg_term + random_term
                
                self.user_groups.interest_obs[i, :] = np.clip(self.user_groups.interest_obs[i, :], 0, 1)
                self.user_groups.interest_obs[i, :] = self.user_groups.interest_obs[i, :]/np.sum(self.user_groups.interest_obs[i, :], keepdims=True)

                user_group_tracer.append({'item2rec': np.squeeze(item2rec[i, :, :]), 'feedback': y[i, :].reshape((1,-1)), 'interest_obs': self.user_groups.interest_obs[i, :]})
        
            #self.user_groups.interest_obs = np.clip(self.user_groups.interest_obs, 0, 1)
            #self.user_groups.interest_obs = self.user_groups.interest_obs/np.sum(self.user_groups.interest_obs, axis=1, keepdims=True)

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