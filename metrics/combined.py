# -*- coding: utf-8 -*-
"""

Created on Fri Jan 15 15:52:59 2021

@author: boris

"""

from feature_distribution import feature_distribution
from compute_wd import compute_wd
from compute_identifiability import compute_identifiability
from fid import compute_frechet_distance
from parzen import compute_parzen
from precision_recall import compute_prc
from prdc import compute_prdc
from evaluation import compute_alpha_precision

import torch
import numpy as np

if torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cpu'

def compute_metrics(X, Y, which_metric=None, wd_params=None, model=None, verbose=True):
    results = {}
    emb_types = ['']
    
    if model is not None:
        emb_types.append('_OC')    
    elif verbose:
        print('#####################!OC model not defined !##################')
        
    if wd_params is None:
        wd_params = dict()
        wd_params['iterations'] = 2000
        wd_params['h_dim'] = 30
        wd_params['z_dim'] = 10
        wd_params['mb_size'] = 128
    
    if which_metric is None:
            which_metric = [['WD','FD', 'PRDC', 'OC'], # normal
                            ['OC']]                   # additional OneClass
            
    for emb_index, emb in enumerate(emb_types):
        
        if emb_index == 1 and len(which_metric[1])>0:
            if verbose:
                print('Computing metrics for OC embedding')
                print('Embedding data into OC representation')
            model.to(device)
            with torch.no_grad():
                X = model(torch.tensor(X).float().to(device)).cpu().detach().numpy()
                Y = model(torch.tensor(Y).float().to(device)).cpu().detach().numpy()
            if verbose:
                print('Done embedding')
                print('X, std X', np.mean(X), np.std(X))
                print('Y, std Y', np.mean(Y), np.std(Y))
            
        elif verbose:
            print('Computing metrics for no additional OneClass embedding')
    
        
        
        # (1) Marginal distributions
        if 'marg' in which_metric[emb_index]:
            
            if verbose:
                print('Start computing marginal feature distributions')
            results[f'feat_dist{emb}'] = feature_distribution(X, Y)
            if verbose:
                print('Finish computing feature distributions')
                print(results[f'feat_dist{emb}'])
    
    
        # (2) Wasserstein Distance (WD)
        if 'WD' in which_metric[emb_index]:
            if verbose:
                print('Start computing Wasserstein Distance')
            results[f'wd_measure{emb}'] = compute_wd(X, Y, wd_params)
            if verbose:
                print('WD measure: ',results[f'wd_measure{emb}'])
        
        
        # (3) Identifiability 
        if 'ID' in which_metric[emb_index]:
            if verbose:
                print('Start computing identifiability')
            results[f'identifiability{emb}'] = compute_identifiability(X, Y)
            if verbose:
                print('Identifiability measure: ',results[f'identifiability{emb}'])
        
    
        # (4) Frechet distance
        if 'FD' in which_metric[emb_index] or 'FID' in which_metric[emb_index]:
            results[f'fid_value{emb}'] = compute_frechet_distance(X, Y)
            if verbose:
                print('Frechet distance', results[f'fid_value{emb}'])
                print('Frechet distance/dim', results[f'fid_value{emb}']/Y.shape[-1])
        
    
        # (5) Parzen
        if 'parzen' in which_metric[emb_index]:
            results[f'parzen_ll{emb}'], results[f'parzen_std{emb}'] = compute_parzen(X, Y, sigma=0.408)
            print(f'Parzen Log-Likelihood of test set = {results["parzen_ll"]}, se: {results["parzen_std"]}')
    
                
        # (6) Precision/Recall
        if 'PR' in which_metric[emb_index]:
            results[f'PR{emb}'] = compute_prc(X,Y)
        elif 'PRDC' in which_metric[emb_index]:
            if verbose:
                print('Start computing P&R and D&C')
            prdc_res = compute_prdc(X,Y)
            for key in prdc_res:
                if verbose:
                    print('PRDC:', key, prdc_res[key])
                results[key+emb] = prdc_res[key]
        
        # (7) OneClass
        if 'OC' in which_metric[emb_index] or 'OCalt' in which_metric[emb_index]:
            if emb_index==1:
                emb_center = model.c
            else:
                emb_center = np.mean(X,axis=0)
            if verbose:
                print('Start computing OC metrics')
            if 'OCalt' in which_metric[emb_index]:
                if 'OC' in which_metric[emb_index]:
                    print('Cannot calculate both alternative and non-alternative coverage in one run!')
                OC_res = compute_alpha_precision(X, Y, emb_center, alternative_coverage=True)
            else:
                OC_res = compute_alpha_precision(X, Y, emb_center, alternative_coverage=False)    
            alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authen = OC_res
            results[f'alphas{emb}'] = alphas
            results[f'alpha_pc{emb}'] = alpha_precision_curve
            results[f'beta_cv{emb}'] = beta_coverage_curve
            results[f'authen{emb}'] = authen
            results[f'Dpa{emb}'] = Delta_precision_alpha
            results[f'Dcb{emb}'] = Delta_coverage_beta
            if verbose:
                print('OneClass: Delta_precision_alpha', results[f'Dpa{emb}'])
                print('OneClass: Delta_coverage_beta  ', results[f'Dcb{emb}'])
                print('OneClass: authenticity    ', results[f'authen{emb}'])
        

    return results