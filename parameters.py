#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:42:32 2020

@author: jacobb
"""

import numpy as np
import torch
from scipy.special import comb

# This contains one single function that generates a dictionary of parameters, which is provided to the model on initialisation
def parameters():
    params = {}
    # -- World parameters
    # Does this world include the standing still action?
    params['has_static_action'] = True
    # Number of available actions, excluding the stand still action (since standing still has an action vector full of zeros, it won't add to the action vector dimension)
    params['n_actions'] = 4
    # Bias for explorative behaviour to pick the same action again, to encourage straight walks
    params['explore_bias'] = 2
    # Rate at which environments with shiny objects occur between training environments. Set to 0 for no shiny environments at all
    params['shiny_rate'] = 0
    # Discount factor in calculating Q-values to generate shiny object oriented behaviour
    params['shiny_gamma'] = 0.7
    # Inverse temperature for shiny object behaviour to pick actions based on Q-values
    params['shiny_beta'] = 1.5
    # Number of shiny objects in the arena
    params['shiny_n'] = 2
    # Number of times to return to a shiny object after finding it
    params['shiny_returns'] = 15
    # Group all shiny parameters together to pass them to the world object
    params['shiny'] = {'gamma' : params['shiny_gamma'], 'beta' : params['shiny_beta'], 'n' : params['shiny_n'], 'returns' : params['shiny_returns']}
      
    # -- Traning parameters
    # Number of walks to generate
    params['train_it'] = 20000
    # Number of steps to roll out before backpropagation through time
    params['n_rollout'] = 20
    # Batch size: number of walks for training simultaneously
    params['batch_size'] = 16    
    # Minimum length of a walk on one environment. Walk lengths are sampled uniformly from a window that shifts down until its lower limit is walk_it_min at the end of training
    params['walk_it_min'] = 25
    # Maximum length of a walk on one environment. Walk lengths are sampled uniformly from a window that starts with its upper limit at walk_it_max in the beginning of training, then shifts down
    params['walk_it_max'] = 300
    # Width of window from which walk lengths are sampled: at any moment, new walk lengths are sampled window_center +/- 0.5 * walk_it_window where window_center shifts down
    params['walk_it_window'] = 0.2 * (params['walk_it_max'] - params['walk_it_min'])
    # Weights of prediction losses
    params['loss_weights_x'] = 1
    # Weights of grounded location losses
    params['loss_weights_p'] = 1
    # Weights of abstract location losses
    params['loss_weights_g'] = 1
    # Weights of regularisation losses
    params['loss_weights_reg_g'] = 0.01 
    params['loss_weights_reg_p'] = 0.02
    # Weights of losses: re-balance contributions of L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p
    params['loss_weights'] = torch.tensor([params['loss_weights_p'], params['loss_weights_p'], params['loss_weights_x'], params['loss_weights_x'], params['loss_weights_x'], params['loss_weights_g'], params['loss_weights_reg_g'], params['loss_weights_reg_p']], dtype=torch.float)
    # Number of backprop iters until latent parameter losses (L_p_g, L_p_x, L_g) are all fully weighted
    params['loss_weights_p_g_it'] = 2000
    # Number of backptrop iters until regularisation losses are fully weighted
    params['loss_weights_reg_p_it'] = 4000
    params['loss_weights_reg_g_it'] = 40000000    
    # Number of backprop iters until eta is (rate of remembering) completely 'on'
    params['eta_it'] = 16000
    # Number of backprop iters until lambda (rate of forgetting) is completely 'on'
    params['lambda_it'] = 200
    # Determine how much to use an offset for the standard deviation of the inferred grounded location to reduce its influence
    params['p2g_scale_offset'] = 0
    # Additional value to offset standard deviation of inferred grounded location when inferring new abstract location, to reduce influence in precision weighted mean
    params['p2g_sig_val'] = 10000
    # Set number of iterations where offset scaling should be 0.5
    params['p2g_sig_half_it'] = 400
    # Set how fast offset scaling should decrease - after p2g_sig_half_it + p2g_sig_scale_it the offset scaling is down to ~0.25 (1/(1+e) to be exact)
    params['p2g_sig_scale_it'] = 200
    # Maximum learning rate 
    params['lr_max'] = 9.4e-4
    # Minimum learning rate 
    params['lr_min'] = 8e-5
    # Rate of learning rate decay
    params['lr_decay_rate'] = 0.5
    # Steps of learning rate decay
    params['lr_decay_steps'] = 4000
    
    # -- Model parameters 
    # Decide whether to sample, or assume no noise and simply take mean of all distributions
    params['do_sample'] = False
    # Decide whether to use inferred ground location while inferring new abstract location, instead of only previous grounded location (James's infer_g_type)
    params['use_p_inf'] = True    
    # Decide whether to use seperate grid modules that recieve shiny information for object vector cells. To disable OVC, set this False, and set n_ovc to [0 for _ in range(len(params['n_g_subsampled']))]
    params['separate_ovc'] = False
    # Standard deviation for initial initial g (which will then be learned)
    params['g_init_std'] = 0.5
    # Standard deviation to initialise hidden to output layer of MLP for inferring new abstract location from memory of grounded location
    params['g_mem_std'] = 0.1
    # Hidden layer size of MLP for abstract location transitions
    params['d_hidden_dim'] = 20
    
    # ---- Neuron and module parameters
    # Neurons for subsampled entorhinal abstract location f_g(g) for each frequency module
    params['n_g_subsampled'] = [10, 10, 8, 6, 6] 
    # Neurons for object vector cells. Neurons will get new modules if object vector cell modules are separated; otherwise, they are added to existing abstract location modules.
    # a) No additional modules, no additional object vector neurons (e.g. when not using shiny environments): [0 for _ in range(len(params['n_g_subsampled']))], and separate_ovc set to False
    # b) No additional modules, but n additional object vector neurons in each grid module: [n for _ in range(len(params['n_g_subsampled']))], and separate_ovc set to False
    # c) Additional separate object vector modules, with n, m neurons: [n, m], and separate_ovc set to True
    params['n_ovc'] = [0 for _ in range(len(params['n_g_subsampled']))]
    # Add neurons for object vector cells. Add new modules if object vector cells get separate modules, or else add neurons to existing modules
    params['n_g_subsampled'] = params['n_g_subsampled'] + params['n_ovc'] if params['separate_ovc'] else [grid + ovc for grid, ovc in zip(params['n_g_subsampled'],params['n_ovc'])]
    # Number of hierarchical frequency modules for object vector cells
    params['n_f_ovc'] = len(params['n_ovc']) if params['separate_ovc'] else 0
    # Number of hierarchical frequency modules for grid cells
    params['n_f_g'] = len(params['n_g_subsampled']) - params['n_f_ovc']    
    # Total number of modules
    params['n_f'] = len(params['n_g_subsampled'])
    # Number of neurons of entorhinal abstract location g for each frequency
    params['n_g'] = [3 * g for g in params['n_g_subsampled']]
    # Neurons for sensory observation x
    params['n_x'] = 45    
    # Neurons for compressed sensory experience x_c
    params['n_x_c'] = 10
    # Neurons for temporally filtered sensory experience x for each frequency
    params['n_x_f'] = [params['n_x_c'] for _ in range(params['n_f'])]
    # Neurons for hippocampal grounded location p for each frequency
    params['n_p'] = [g * x for g, x in zip(params['n_g_subsampled'], params['n_x_f'])]
    # Initial frequencies of each module. For ease of interpretation (higher number = higher frequency) this is 1 - the frequency as James uses it
    params['f_initial'] = [0.99, 0.3, 0.09, 0.03, 0.01]
    # Add frequencies of object vector cell modules, if object vector cells get separate modules
    params['f_initial'] = params['f_initial'] + params['f_initial'][0:params['n_f_ovc']]
    
    # ---- Memory parameters
    # Use common memory for generative and inference network
    params['common_memory'] = False
    # Hebbian rate of forgetting
    params['lambda'] = 0.9999    
    # Hebbian rate of remembering
    params['eta'] = 0.5   
    # Hebbian retrieval decay term
    params['kappa'] = 0.8
    # Number of iterations of attractor dynamics for memory retrieval
    params['i_attractor'] = params['n_f_g']
    # Maximum iterations of attractor dynamics per frequency in inference model, so you can early stop low-frequency modules. Set to None for no early stopping
    params['i_attractor_max_freq_inf'] = [params['i_attractor'] for _ in range(params['n_f'])]
    # Maximum iterations of attractor dynamics per frequency in generative model, so you can early stop low-frequency modules. Don't early stop for object vector cell modules.
    params['i_attractor_max_freq_gen'] = [params['i_attractor'] - freq_nr for freq_nr in range(params['n_f_g'])] + [params['i_attractor'] for _ in range(params['n_f_ovc'])]
        
    # --- Connectivity matrices
    # Set connections when forming Hebbian memory of grounded locations: from low frequency modules to high. High frequency modules come first (different from James!)
    params['p_update_mask'] = torch.zeros((np.sum(params['n_p']),np.sum(params['n_p'])), dtype=torch.float)
    n_p = np.cumsum(np.concatenate(([0],params['n_p'])))    
    # Entry M_ij (row i, col j) is the connection FROM cell i TO cell j. Memory is retrieved by h_t+1 = h_t * M, i.e. h_t+1_j = sum_i {connection from i to j * h_t_i}
    for f_from in range(params['n_f']):
        for f_to in range(params['n_f']):
            # For connections that involve separate object vector modules: these are connected to all normal modules, but hierarchically between object vector modules
            if f_from > params['n_f_g'] or f_to > params['n_f_g']:
                # If this is a connection between object vector modules: only allow for connection from low to high frequency
                if (f_from > params['n_f_g'] and f_to > params['n_f_g']):
                    if params['f_initial'][f_from] <= params['f_initial'][f_to]:
                        params['p_update_mask'][n_p[f_from]:n_p[f_from+1],n_p[f_to]:n_p[f_to+1]] = 1.0
                # If this is a connection to between object vector and normal modules: allow any connections, in both directions
                else:
                    params['p_update_mask'][n_p[f_from]:n_p[f_from+1],n_p[f_to]:n_p[f_to+1]] = 1.0
            # Else: this is a connection between abstract location frequency modules; only allow for connections if it goes from low to high frequency                
            else:
                if params['f_initial'][f_from] <= params['f_initial'][f_to]:
                    params['p_update_mask'][n_p[f_from]:n_p[f_from+1],n_p[f_to]:n_p[f_to+1]] = 1.0
    # During memory retrieval, hierarchical memory retrieval of grounded location is implemented by early-stopping low-frequency memory updates, using a mask for updates at every retrieval iteration
    params['p_retrieve_mask_inf'] = [torch.zeros(sum(params['n_p'])) for _ in range(params['i_attractor'])]
    params['p_retrieve_mask_gen'] = [torch.zeros(sum(params['n_p'])) for _ in range(params['i_attractor'])]
    # Build masks for each retrieval iteration
    for mask, max_iters in zip([params['p_retrieve_mask_inf'], params['p_retrieve_mask_gen']], [params['i_attractor_max_freq_inf'], params['i_attractor_max_freq_gen']]):
        # For each frequency, we get the number of update iterations, and insert ones in the mask for those iterations
        for f, max_i in enumerate(max_iters):
            # Update masks up to maximum iteration
            for i in range(max_i):
                mask[i][n_p[f]:n_p[f+1]] = 1.0                
    # In path integration, abstract location frequency modules can influence the transition of other modules hierarchically (low to high). Set for each frequency module from which other frequencies input is received
    params['g_connections'] = [[params['f_initial'][f_from] <= params['f_initial'][f_to] for f_from in range(params['n_f_g'])] + [False for _ in range(params['n_f_ovc'])] for f_to in range(params['n_f_g'])]
    # Add connections for separate object vector cell module: only between object vector cell modules - and make those hierarchical too
    params['g_connections'] = params['g_connections'] + [[False for _ in range(params['n_f_g'])] + [params['f_initial'][f_from] <= params['f_initial'][f_to] for f_from in range(params['n_f_g'], params['n_f'])] for f_to in range(params['n_f_g'], params['n_f'])]

    # ---- Static matrices            
    # Matrix for repeating abstract location g to do outer product with sensory information x with elementwise product. Also see (*) note at bottom
    params['W_repeat'] = [torch.tensor(np.kron(np.eye(params['n_g_subsampled'][f]),np.ones((1,params['n_x_f'][f]))), dtype=torch.float) for f in range(params['n_f'])]
    # Matrix for tiling sensory observation x to do outer product with abstract with elementwise product. Also see (*) note at bottom
    params['W_tile'] = [torch.tensor(np.kron(np.ones((1,params['n_g_subsampled'][f])),np.eye(params['n_x_f'][f])), dtype=torch.float) for f in range(params['n_f'])]    
    # Table for converting one-hot to two-hot compressed representation 
    params['two_hot_table'] = [[0]*(params['n_x_c']-2) + [1]*2]
    # We need a compressed code for each possible observation, but it's impossible to have more compressed codes than "n_x_c choose 2"
    for i in range(1, min(int(comb(params['n_x_c'],2)), params['n_x'])):
        # Copy previous code
        code = params['two_hot_table'][-1].copy()
        # Find latest occurrence of [0 1] in that code
        swap = [index for index in range(len(code)-1,-1,-1) if code[index:index+2] == [0,1]][0]
        # Swap those to get new code
        code[swap:swap+2] = [1,0]
        # If the first one was swapped: value after swapped pair is 1
        if swap+2 < len(code) and code[swap+2] == 1:
            # In that case: move the second 1 all the way back - reverse everything after the swapped pair
            code[swap+2:] = code[:swap+1:-1]
        # And append new code to array
        params['two_hot_table'].append(code)
    # Convert each code to column vector pytorch tensor
    params['two_hot_table'] = [torch.tensor(code) for code in params['two_hot_table']]
    # Downsampling matrix to go from grid cells to compressed grid cells for indexing memories by simply taking only the first n_g_subsampled grid cells
    params['g_downsample'] = [torch.cat([torch.eye(dim_out, dtype=torch.float),torch.zeros((dim_in-dim_out,dim_out), dtype=torch.float)]) for dim_in, dim_out in zip(params['n_g'],params['n_g_subsampled'])]
    return params

# This specifies how parameters are updated at every backpropagation iteration/gradient update
def parameter_iteration(iteration, params):
    # Calculate eta (rate of remembering) and lambda (rate of forgetting) for Hebbian memory updates
    eta = min((iteration+1)/params['eta_it'], 1) * params['eta']
    lamb = min((iteration+1)/params['lambda_it'], 1) * params['lambda']
    # Calculate current scaling of variance offset for ground location inference
    p2g_scale_offset = 1/(1+np.exp((iteration - params['p2g_sig_half_it'])/params['p2g_sig_scale_it']))
    # Calculate current learning rate
    lr = max(params['lr_min'] + (params['lr_max'] - params['lr_min']) * (params['lr_decay_rate'] ** (iteration / params['lr_decay_steps'])), params['lr_min'])
    # Calculate center of walk length window, within which the walk lenghts of new walks are uniformly sampled
    walk_length_center = params['walk_it_max'] - params['walk_it_window'] * 0.5 - min((iteration+1)/params['train_it'],1) * (params['walk_it_max'] - params['walk_it_min'] - params['walk_it_window'])
    # Calculate current loss weights
    L_p_g = min((iteration+1)/params['loss_weights_p_g_it'], 1) * params['loss_weights_p']
    L_p_x = min((iteration+1)/params['loss_weights_p_g_it'], 1) * params['loss_weights_p'] * (1 - p2g_scale_offset)
    L_x_gen = params['loss_weights_x']
    L_x_g = params['loss_weights_x']
    L_x_p = params['loss_weights_x']
    L_g = min((iteration+1)/params['loss_weights_p_g_it'], 1) * params['loss_weights_g']
    L_reg_g = (1 - min((iteration+1) / params['loss_weights_reg_g_it'], 1)) * params['loss_weights_reg_g']
    L_reg_p = (1 - min((iteration+1) / params['loss_weights_reg_p_it'], 1)) * params['loss_weights_reg_p']
    # And concatenate them in the order expected by the model
    loss_weights = torch.tensor([L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p])
    # Return all updated parameters
    return eta, lamb, p2g_scale_offset, lr, walk_length_center, loss_weights


'''
(*) Note on W_tile and W_repeat:
W_tile and W_repeat are for calculating outer products then vector flattening by matrix multiplication then elementwise product:
    g = np.random.rand(4,1)
    x = np.random.rand(3,1)
    out1 = np.matmul(g,np.transpose(x)).reshape((4*3,1))
    W_repeat = np.kron(np.eye(4),np.ones((3,1)))
    W_tile = np.kron(np.ones((4,1)),np.eye(3))
    out2 = np.matmul(W_repeat,g) * np.matmul(W_tile,x)
Or in the case of row vectors, which is what you'd do for batch calculation:
    g = g.T
    x = x.T
    out3 = np.matmul(np.transpose(g), x).reshape((1,4*3)) # Notice how this is not batch-proof!
    W_repeat = np.kron(np.eye(4), np.ones((1,3)))
    W_tile = np.kron(np.ones((1,4)),np.eye(3))
    out4 = np.matmul(g, W_repeat) * np.matmul(x,W_tile) # This is batch-proof        
'''     