import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
sns.set_style('ticks')
plt.rcParams['font.size'] = 20


def transform_files(b, gp, gm, s):
    with open('./tmp_re/a_0.00_b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.2f}/dynamic_v1.pkl'.format(b, gp, gm, s), 'rb') as f:
        all_time_step_result = pickle.load(f)
    end = all_time_step_result[-1]['entropy_diff_2']
    with open('./fig_demo/b_{:.2f}_gp_{:.2f}_gm_{:.2f}_s_{:.2f}.pkl'.format(b, gp, gm, s), 'wb') as f:
        pickle.dump(end,f)


def get_figs(beta=None, gamma_plus=None, gamma_minus=None, sigma=None, fig_type='beta'):
    if fig_type == 'beta':
        colors = ['#F25079',('#45BC9C','#008F92'),'#2A557F']
        for idx, b in enumerate(beta):
            with open('./fig_demo/b_{:.2f}_gp_1.00_gm_-0.10_s_0.01.pkl'.format(b), 'rb') as f:
                end = pickle.load(f)

            theory = pd.read_csv('./theory_results/a_0.00_b_{:.2f}_gp_1.00_gm_-0.10_s_0.01.csv'.format(b), names=['x','y'])
            
            
            if idx != 1:
                fig = plt.figure(figsize=(4,4))
                
                theory = pd.read_csv('./theory_results/a_0.00_b_{:.2f}_gp_1.00_gm_-0.10_s_0.01.csv'.format(b), names=['x','y'])
                plt.plot(theory['x'],theory['y'], color='r', linestyle='--', linewidth=2.5)

                sns.histplot(end, kde=False, bins=np.arange(0, 1.65, 0.05), alpha=0.8,
                color=colors[idx], stat='density', label='$\gamma_-$=0', legend=False, multiple="stack")
                plt.xticks([0, 0.5, 1, 1.5])
                plt.xlim(-0.1, 1.6)
                plt.yticks([0, 2, 4, 6])
                plt.ylim(0, 6)
                plt.ylabel('')
                plt.tight_layout()
                
            else:
                fig = plt.figure(figsize=(4,4))
                theory = pd.read_csv('./theory_results/a_0.00_b_{:.2f}_gp_1.00_gm_-0.10_s_0.01.csv'.format(b), names=['x','y'])
                plt.plot(theory['x'], theory['y'], color='r', linestyle='--', linewidth=2.5)
                min_x = theory.loc[(theory['x']<0.8)&(theory['x']>0.4)].sort_values(by='y')
                min_x = min_x.iloc[0,0]
                    
                #print(min_x)
                ax = sns.histplot(end, kde=False, bins=np.arange(0, 1.65, 0.05), alpha=0.8,
                                    color=colors[idx][0], stat='density', label='$\gamma_-$=0', legend=False, multiple="stack")
                
                for p in ax.patches:
                    if  p.get_x() < min_x:
                        p.set_alpha(0.8)
                        p.set_facecolor(colors[idx][1])
                        
                plt.xticks([0, 0.5, 1, 1.5])
                plt.xlim(-0.1, 1.6)
                plt.yticks([0, 2, 4, 6])
                plt.ylim(0, 6)
                plt.ylabel('')
                plt.tight_layout()
    elif fig_type == 'gamma_plus':
        colors = ['#F25079',('#45BC9C','#008F92'),'#2A557F']
        for idx, gp in enumerate(gamma_plus):
            with open('./fig_demo/b_7.00_gp_{:.2f}_gm_-0.10_s_0.01.pkl'.format(gp), 'rb') as f:
                end = pickle.load(f)

            if idx != 1:
                fig = plt.figure(figsize=(4,4))
                sns.histplot(end, kde=False, bins=np.arange(0, 2.05, 0.05), alpha=0.8,
                color=colors[idx], stat='density', label='$\gamma_-$=0', legend=False, multiple="stack")
                theory = pd.read_csv('./theory_results/a_0.00_b_7.00_gp_{:.2f}_gm_-0.10_s_0.01_gp.csv'.format(gp), names=['x','y'])
                plt.plot(theory['x'],theory['y'], color='r', linestyle='--', linewidth=2.5)
                
                
            else:
                fig = plt.figure(figsize=(4,4))
                if os.path.exists('./theory_results/a_0.00_b_7.00_gp_{:.2f}_gm_-0.10_s_0.01_gp.csv'.format(gp)):
                    theory = pd.read_csv('./theory_results/a_0.00_b_7.00_gp_{:.2f}_gm_-0.10_s_0.01_gp.csv'.format(gp), names=['x','y'])
                    plt.plot(theory['x'], theory['y'], color='r', linestyle='--', linewidth=2.5)
                    min_x = theory.loc[(theory['x']<0.7)&(theory['x']>0.25)].sort_values(by='y')
                    min_x = min_x.iloc[0,0]
                    #print(min_x)
                else:
                    pass
                
                ax = sns.histplot(end, kde=False, bins=np.arange(0, 2.05, 0.05), alpha=0.8,
                                    color=colors[idx][0], stat='density', label='$\gamma_-$=0', legend=False, multiple="stack")
                
                for p in ax.patches:
                    if  p.get_x() < min_x:
                        p.set_facecolor(colors[idx][1])
                        p.set_alpha(0.8)
            plt.xticks(np.arange(0,2.0,0.5))
            plt.yticks([0,2,4,6])
            plt.xlim(-0.1, 1.6)
            plt.ylabel('')
            plt.tight_layout()
    elif fig_type == 'gamma_minus':
        colors = ['#F25079',('#45BC9C','#008F92'),'#2A557F']
        colors = colors[::-1]

        for idx, gm in enumerate(gamma_minus):
            with open('./fig_demo/b_7.00_gp_1.00_gm_{:.2f}_s_0.01.pkl'.format(gm), 'rb') as f:
                end = pickle.load(f)
            
            if idx != 1:
                fig = plt.figure(figsize=(4,4))
                if os.path.exists('./theory_results/a_0.00_b_7.00_gp_1.00_gm_{:.2f}_s_0.01.csv'.format(gm)):
                    theory = pd.read_csv('./theory_results/a_0.00_b_7.00_gp_1.00_gm_{:.2f}_s_0.01.csv'.format(gm), names=['x','y'])
                    plt.plot(theory['x'], theory['y'], color='r', linestyle='--', linewidth=2.5)
                sns.histplot(end, kde=False, bins=np.arange(0, 1.65, 0.05),
                    color=colors[idx], stat='density', label='$\gamma_-$=0', legend=False)
            else:
                fig = plt.figure(figsize=(4,4))
                if os.path.exists('./theory_results/a_0.00_b_7.00_gp_1.00_gm_{:.2f}_s_0.01.csv'.format(gm)):
                    theory = pd.read_csv('./theory_results/a_0.00_b_7.00_gp_1.00_gm_{:.2f}_s_0.01.csv'.format(gm), names=['x','y'])
                    plt.plot(theory['x'], theory['y'], color='r', linestyle='--', linewidth=2.5)
                    min_x = theory.loc[(theory['x']<0.7)&(theory['x']>0.25)].sort_values(by='y')
                    min_x = min_x.iloc[0,0]
                
                ax = sns.histplot(end, kde=False, bins=np.arange(0, 1.65, 0.05),
                    color=colors[idx][0], stat='density', label='$\gamma_-$=0', legend=False)
                
                for p in ax.patches:
                    if  p.get_x() < min_x:
                        p.set_alpha(0.8)
                        p.set_facecolor(colors[idx][1])
            if idx == 0:
                #plt.grid(color='grey', linestyle='--', which='major')
                plt.xticks([0, 0.5, 1, 1.5])
                plt.yticks([0, 2, 4, 6])
                plt.xlim(-0.1, 1.6)
                plt.ylim(-0.2, 6)
                plt.ylabel('')
            elif idx == 2:
                plt.xticks([0, 0.5, 1, 1.5])
                plt.yticks([0, 2, 4])
                plt.xlim(-0.2, 1.3)
                plt.ylim(-0.1333, 4)
                plt.ylabel('')
                
            else:
                plt.xticks([0, 0.5, 1, 1.5])
                plt.yticks([0, 2, 4])
                plt.xlim(-0.1, 1.6)
                plt.ylim(-0.1333, 4)
                plt.ylabel('')
    elif fig_type == 'sigma':
        colors = ['#F25079',('#45BC9C','#008F92'),'#2A557F']
        colors = colors[::-1]
        for idx, s in enumerate(sigma):
            if len(str(s).split('.')[1])  == 3:
                print(s)
                with open('./fig_demo/b_10.00_gp_1.00_gm_-0.10_s_{:.3f}.pkl'.format(s), 'rb') as f:
                    end = pickle.load(f)
            
            elif s<=0.1:
                with open('./fig_demo/b_10.00_gp_1.00_gm_-0.10_s_{:.2f}.pkl'.format(s), 'rb') as f:
                    end = pickle.load(f)
            else:
                with open('./fig_demo/b_10.00_gp_1.00_gm_-0.10_s_{:.2f}.pkl'.format(s), 'rb') as f:
                    end = pickle.load(f)   
            
            if idx != 1:
                fig = plt.figure(figsize=(4,4))
                if os.path.exists('./theory_results/a_0.00_b_10.00_gp_1.00_gm_-0.10_s_{:.2f}.csv'.format(s)):
                    theory = pd.read_csv('./theory_results/a_0.00_b_10.00_gp_1.00_gm_-0.10_s_{:.2f}.csv'.format(s), names=['x','y'])
                    plt.plot(theory['x'],theory['y'], color='r', linestyle='--', linewidth=2.5)

                sns.histplot(end, kde=False, bins=np.arange(0, 1.65, 0.05),  
                color=colors[idx], stat='density', label='$\gamma_-$=0', legend=False, multiple="stack")
                plt.xticks([0, 0.5, 1, 1.5])
                plt.xlim(-0.1, 1.6)
                plt.yticks([0, 2, 4, 6])
                plt.ylim(0, 6)
                plt.ylabel('')
            else:
                fig = plt.figure(figsize=(4,4))
                if os.path.exists('./theory_results/a_0.00_b_10.00_gp_1.00_gm_-0.10_s_{:.2f}.csv'.format(s)):
                    theory = pd.read_csv('./theory_results/a_0.00_b_10.00_gp_1.00_gm_-0.10_s_{:.2f}.csv'.format(s), names=['x','y'])
                    plt.plot(theory['x'], theory['y'], color='r', linestyle='--', linewidth=2.5)
                    min_x = theory.loc[(theory['x']<1)&(theory['x']>0.25)].sort_values(by='y')
                    min_x = min_x.iloc[0,0]
                    

                ax = sns.histplot(end, kde=False, bins=np.arange(0, 1.65, 0.05), 
                                    color=colors[idx][0], stat='density', label='$\gamma_-$=0', legend=False, multiple="stack")
                
                for p in ax.patches:
                    if  p.get_x() < min_x:
                        p.set_alpha(0.8)
                        p.set_facecolor(colors[idx][1])
                
                plt.xticks([0, 0.5, 1, 1.5])
                plt.xlim(-0.1, 1.6)
                plt.yticks([0, 2, 4, 6])
                plt.ylim(0, 6)
                plt.ylabel('')


