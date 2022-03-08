#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:26:32 2020

This is a pytorch implementation of the Tolman-Eichenbaum Machine,
written by Jacob Bakermans after the original by James Whittington.
The referenced paper is the bioRxiv publication at https://www.biorxiv.org/content/10.1101/770495v2

Release v1.0.0: Fully functional pytorch model, without any extensions

@author: jacobb
"""
# Standard modules
import numpy as np
import torch
import pdb
import copy
from scipy.stats import truncnorm
# Custom modules
import utils

class Model(torch.nn.Module):
    def __init__(self, params):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(Model, self).__init__()
        # Copy hyperparameters (e.g. network sizes) from parameter dict, usually generated from parameters() in parameters.py
        self.hyper = copy.deepcopy(params)
        # Create trainable parameters
        self.init_trainable()
    
    def forward(self, walk, prev_iter = None, prev_M = None):
        # The previous iteration may contain walks without action. These are new walks, for which some parameters need to be reset.
        steps = self.init_walks(prev_iter)
        # Forward pass: perform a TEM iteration for each set of [place, observation, action], and produce inferred and generated variables for each step.
        for g, x, a in walk:
            # If there is no previous iteration at all: all walks are new, initialise a whole new iteration object
            if steps is None:
                # Use an Iteration object to set initial values before any real iterations, initialising M, x_inf as zero. Set actions to None blank to indicate there was no previous action
                steps = [self.init_iteration(g, x, [None for _ in range(len(a))], prev_M)]
            # Perform TEM iteration using transition from previous iteration
            L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf = self.iteration(x, g, steps[-1].a, steps[-1].M, steps[-1].x_inf, steps[-1].g_inf)
            # Store this iteration in iteration object in steps list
            steps.append(Iteration(g, x, a, L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf))    
        # The first step is either a step from a previous walk or initialisiation rubbish, so remove it
        steps = steps[1:]
        # Return steps, which is a list of Iteration objects
        return steps

    def iteration(self, x, locations, a_prev, M_prev, x_prev, g_prev):
        # First, do the transition step, as it will be necessary for both the inference and generative part of the model
        gt_gen, gt_inf = self.gen_g(a_prev, g_prev, locations)        
        # Run inference model: infer grounded location p_inf (hippocampus), abstract location g_inf (entorhinal). Also keep filtered sensory observation (x_inf), and retrieved grounded location p_inf_x
        x_inf, g_inf, p_inf_x, p_inf = self.inference(x, locations, M_prev, x_prev, gt_inf)                        
        # Run generative model: since generative model is only used for training purposes, it will generate from *inferred* variables instead of *generated* variables (as it would when used for generation)
        x_gen, x_logits, p_gen = self.generative(M_prev, p_inf, g_inf, gt_gen)
        # Update generative memory with generated and inferred grounded location. 
        M = [self.hebbian(M_prev[0], torch.cat(p_inf,dim=1), torch.cat(p_gen,dim=1))]
        # If using memory for grounded location inference: append inference memory
        if self.hyper['use_p_inf']:
            # Inference memory is identical to generative memory if using common memory, and updated separatedly if not            
            M.append(M[0] if self.hyper['common_memory'] else self.hebbian(M_prev[1], torch.cat(p_inf,dim=1), torch.cat(p_inf_x,dim=1), do_hierarchical_connections=False))
        # Calculate loss of this step
        L = self.loss(gt_gen, p_gen, x_logits, x, g_inf, p_inf, p_inf_x, M_prev)
        # Return all iteration values
        return L, M, gt_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf
        
    def inference(self, x, locations, M_prev, x_prev, g_gen):
        # Compress sensory observation from one-hot to two-hot (or alternatively, whatever an MLP makes of it)
        x_c = self.f_c(x)
        # Temporally filter sensory observation by mixing it with previous experience 
        x_f = self.x_prev2x(x_prev, x_c)
        # Prepare sensory experience for input to memory by normalisation and weighting
        x_ = self.x2x_(x_f)
        # Retrieve grounded location from memory by doing pattern completion on current sensory experience
        p_x = self.attractor(x_, M_prev[1], retrieve_it_mask=self.hyper['p_retrieve_mask_inf']) if self.hyper['use_p_inf'] else None
        # Infer abstract location by combining previous abstract location and grounded location retrieved from memory by current sensory experience
        g = self.inf_g(p_x, g_gen, x, locations)
        # Prepare abstract location for input to memory by downsampling and weighting
        g_ = self.g2g_(g)
        # Infer grounded location from sensory experience and inferred abstract location
        p = self.inf_p(x_, g_)
        # Return variables in order that they were created
        return x_f, g, p_x, p    

    def generative(self, M_prev, p_inf, g_inf, g_gen):
        # Generate observation from inferred grounded location, using only the highest frequency. Also keep non-softmaxed logits which are used in the loss later
        x_p, x_p_logits = self.gen_x(p_inf[0])
        # Retrieve grounded location from memory by pattern completion on inferred abstract location
        p_g_inf = self.gen_p(g_inf, M_prev[0]) # was p_mem_gen
        # And generate observation from the grounded location retrieved from inferred abstract location
        x_g, x_g_logits = self.gen_x(p_g_inf[0])
        # Retreive grounded location from memory by pattern completion on abstract location by transitioning
        p_g_gen = self.gen_p(g_gen, M_prev[0])
        # Generate observation from sampled grounded location
        x_gt, x_gt_logits = self.gen_x(p_g_gen[0])
        # Return all generated observations and their corresponding logits
        return (x_p, x_g, x_gt), (x_p_logits, x_g_logits, x_gt_logits), p_g_inf

    def loss(self, g_gen, p_gen, x_logits, x, g_inf, p_inf, p_inf_x, M_prev):
        # Calculate loss function, separately for each component because you might want to reweight contributions later                
        # L_p_gen is squared error loss between inferred grounded location and grounded location retrieved from inferred abstract location
        L_p_g = torch.sum(torch.stack(utils.squared_error(p_inf, p_gen), dim=0), dim=0)
        # L_p_inf is squared error loss between inferred grounded location and grounded location retrieved from sensory experience
        L_p_x = torch.sum(torch.stack(utils.squared_error(p_inf, p_inf_x), dim=0), dim=0) if self.hyper['use_p_inf'] else torch.zeros_like(L_p_g)
        # L_g is squared error loss between generated abstract location and inferred abstract location
        L_g = torch.sum(torch.stack(utils.squared_error(g_inf, g_gen), dim=0), dim=0)         
        # L_x is a cross-entropy loss between sensory experience and different model predictions. First get true labels from sensory experience
        labels = torch.argmax(x, 1)            
        # L_x_gen: losses generated by generative model from g_prev -> g -> p -> x
        L_x_gen = utils.cross_entropy(x_logits[2], labels)
        # L_x_g: Losses generated by generative model from g_inf -> p -> x
        L_x_g = utils.cross_entropy(x_logits[1], labels)
        # L_x_p: Losses generated by generative model from p_inf -> x
        L_x_p = utils.cross_entropy(x_logits[0], labels)
        # L_reg are regularisation losses, L_reg_g on L2 norm of g
        L_reg_g = torch.sum(torch.stack([torch.sum(g ** 2, dim=1) for g in g_inf], dim=0), dim=0)
        # And L_reg_p regularisation on L1 norm of p
        L_reg_p = torch.sum(torch.stack([torch.sum(torch.abs(p), dim=1) for p in p_inf], dim=0), dim=0)
        # Return total loss as list of losses, so you can possibly reweight them
        L = [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]
        return L

    def init_trainable(self):
        # Scale factor in Laplacian transform for each frequency module. High frequency comes first, low frequency comes last. Learn inverse sigmoid instead of scale factor directly, so domain of alpha is -inf, inf
        self.alpha = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(np.log(self.hyper['f_initial'][f] / (1 - self.hyper['f_initial'][f])), dtype=torch.float)) for f in range(self.hyper['n_f'])])
        # Entorhinal preference weights
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))
        # Entorhinal preference bias
        self.b_x = torch.nn.Parameter(torch.zeros(self.hyper['n_x_c']))
        # Frequency module specific scaling of sensory experience before input to hippocampus
        self.w_p = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.0)) for f in range(self.hyper['n_f'])])        
        # Initial activity of abstract location cells when entering a new environment, like a prior on g. Initialise with truncated normal
        self.g_init = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=self.hyper['n_g'][f], loc=0, scale=self.hyper['g_init_std']), dtype=torch.float)) for f in range(self.hyper['n_f'])])
        # Log of standard deviation of abstract location cells when entering a new environment; standard deviation of the prior on g. Initialise with truncated normal
        self.logsig_g_init = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=self.hyper['n_g'][f], loc=0, scale=self.hyper['g_init_std']), dtype=torch.float)) for f in range(self.hyper['n_f'])])                
        # MLP for transition weights (not in paper, but recommended by James so you can learn about similarities between actions). Size is given by grid connections
        self.MLP_D_a = MLP([self.hyper['n_actions'] for _ in range(self.hyper['n_f'])],
                            [sum([self.hyper['n_g'][f_from] for f_from in range(self.hyper['n_f']) if self.hyper['g_connections'][f_to][f_from]])*self.hyper['n_g'][f_to] for f_to in range(self.hyper['n_f'])],
                            activation=[torch.tanh, None],
                            hidden_dim=[self.hyper['d_hidden_dim'] for _ in range(self.hyper['n_f'])],
                            bias=[True, False])        
        # Initialise the hidden to output weights as zero, so initially you simply keep the current abstract location to predict the next abstract location
        self.MLP_D_a.set_weights(1, 0.0)
        # Transition weights without specifying an action for use in generative model with shiny objects
        self.D_no_a = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(sum([self.hyper['n_g'][f_from] for f_from in range(self.hyper['n_f']) if self.hyper['g_connections'][f_to][f_from]])*self.hyper['n_g'][f_to])) for f_to in range(self.hyper['n_f'])])
        # MLP for standard deviation of transition sample
        self.MLP_sigma_g_path = MLP(self.hyper['n_g'], self.hyper['n_g'], activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.hyper['n_g']])
        # MLP for standard devation of grounded location from retrieved memory sample        
        self.MLP_sigma_p = MLP(self.hyper['n_p'], self.hyper['n_p'], activation=[torch.tanh, torch.exp])
        # MLP to generate mean of abstract location from downsampled abstract location, obtained by summing grounded location over sensory preferences in inference model
        self.MLP_mu_g_mem = MLP(self.hyper['n_g_subsampled'], self.hyper['n_g'], hidden_dim=[2 * g for g in self.hyper['n_g']])
        # Initialise weights in last layer of MLP_mu_g_mem as truncated normal for each frequency module
        self.MLP_mu_g_mem.set_weights(-1, [torch.tensor(truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=self.hyper['g_mem_std']), dtype=torch.float) for f in range(self.hyper['n_f'])])
        # MLP to generate standard deviation of abstract location from two measures (generated observation error and inferred abstract location vector norm) of memory quality
        self.MLP_sigma_g_mem = MLP([2 for _ in self.hyper['n_g_subsampled']], self.hyper['n_g'], activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.hyper['n_g']])
        # MLP to generate mean of abstract location directly from shiny object presence. Outputs to object vector cell modules if they're separated, else to all abstract location modules
        self.MLP_mu_g_shiny = MLP([1 for _ in range(self.hyper['n_f_ovc'] if self.hyper['separate_ovc'] else self.hyper['n_f'])], 
                                  [n_g for n_g in self.hyper['n_g'][(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0):]], 
                                  hidden_dim=[2*n_g for n_g in self.hyper['n_g'][(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0):]])
        # MLP to generate standard deviation of abstract location directly from shiny object presence. Outputs to object vector cell modules if they're separated, else to all abstract location modules
        self.MLP_sigma_g_shiny = MLP([1 for _ in range(self.hyper['n_f_ovc'] if self.hyper['separate_ovc'] else self.hyper['n_f'])],
                                     [n_g for n_g in self.hyper['n_g'][(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0):]],
                                     hidden_dim=[2*n_g for n_g in self.hyper['n_g'][(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0):]], activation=[torch.tanh, torch.exp])
        # MLP for decompressing highest frequency sensory experience to sensory observation
        self.MLP_c_star = MLP(self.hyper['n_x_f'][0], self.hyper['n_x'], hidden_dim=20 * self.hyper['n_x_c'])
    
    def init_iteration(self, g, x, a, M):
        # On the very first iteration, update the batch size based on the data. This is useful when doing analysis on the network with different batch sizes compared to training
        self.hyper['batch_size'] = x.shape[0]
        # Initalise hebbian memory connectivity matrix [M_gen, M_inf] if it wasn't initialised yet
        if M is None:
            # Create new empty memory dict for generative network: zero connectivity matrix M_0, then empty list of the memory vectors a and b for each iteration for efficient hebbian memory computation
            M = [torch.zeros((self.hyper['batch_size'],sum(self.hyper['n_p']),sum(self.hyper['n_p'])), dtype=torch.float)]
            # Append inference memory only if memory is used in grounded location inference
            if self.hyper['use_p_inf']:
                # If inference and generative network share common memory: reuse same connectivity, and same memory vectors. Else, create a new empty memory list for inference network
                M.append(M[0] if self.hyper['common_memory'] else torch.zeros((self.hyper['batch_size'],sum(self.hyper['n_p']),sum(self.hyper['n_p'])), dtype=torch.float)) 
        # Initialise previous abstract location by stacking abstract location prior
        g_inf = [torch.stack([self.g_init[f] for _ in range(self.hyper['batch_size'])]) for f in range(self.hyper['n_f'])]
        # Initialise previous sensory experience with zeros, as there is no data yet for temporal smoothing
        x_inf = [torch.zeros((self.hyper['batch_size'], self.hyper['n_x_f'][f])) for f in range(self.hyper['n_f'])]        
        # And construct new iteration for that g, x, a, and M
        return Iteration(g=g, x=x, a=a, M=M, x_inf=x_inf, g_inf=g_inf)    
    
    def init_walks(self, prev_iter):
        # Only reset parameters for previous iteration if a previous iteration was actually provided - if it wasn't, all parameters will be reset when creating a fresh Iteration object in init_iteration
        if prev_iter is not None:   
            # The supplied previous iteration might have new walks starting, with empty actions. For these walks some parameters need to be reset
            for a_i, a in enumerate(prev_iter[0].a):
                # A new walk is indicated by having a None action in the previous iteration
                if a is None:
                    # Reset the initial connectivity matrix for this walk
                    for M in prev_iter[0].M:
                        M[a_i,:,:] = 0         
                    # Reset the abstract location for this walk
                    for f, g_inf in enumerate(prev_iter[0].g_inf):
                        g_inf[a_i,:] = self.g_init[f]
                    # Reset the sensory experience for this walk
                    for f, x_inf in enumerate(prev_iter[0].x_inf):
                        x_inf[a_i,:] = torch.zeros(self.hyper['n_x_f'][f])
        # Return the iteration with reset parameters (or simply the empty array if prev_iter was empty)
        return prev_iter
                    
    def gen_g(self, a_prev, g_prev, locations):
        # Transition from previous abstract location to new abstract location using weights specific to action taken for each frequency module
        mu_g = self.f_mu_g_path(a_prev, g_prev)
        sigma_g = self.f_sigma_g_path(a_prev, g_prev)
        # Either sample new abstract location g or simply take the mean of distribution in noiseless case.
        g = [mu_g[f] + sigma_g[f] * np.random.randn() if self.hyper['do_sample'] else mu_g[f] for f in range(self.hyper['n_f'])]
        # But for environments with shiny objects, the transition to the new abstract location shouldn't have access to the action direction in the generative model
        shiny_envs = [location['shiny'] is not None for location in locations]
        # If there are any shiny environments, the abstract locations for the generative model will need to be re-calculated without providing actions for those
        g_gen =  self.f_mu_g_path(a_prev, g_prev, no_direc=shiny_envs) if any(shiny_envs) else g
        # Return generated abstract location after transition
        return g_gen, (g, sigma_g)
    
    def gen_p(self, g, M_prev):
        # We want to use g as an index for memory retrieval, but it doesn't have the right dimensions (these are grid cells, we need place cells). We need g_ instead
        g_ = self.g2g_(g)
        # Retreive memory: do pattern completion on abstract location to get grounded location    
        mu_p = self.attractor(g_, M_prev, retrieve_it_mask=self.hyper['p_retrieve_mask_gen'])
        sigma_p = self.f_sigma_p(mu_p)
        # Either sample new grounded location p or simply take the mean of distribution in noiseless case
        p = [mu_p[f] + sigma_p[f] * np.random.randn() if self.hyper['do_sample'] else mu_p[f] for f in range(self.hyper['n_f'])]
        # Return pattern-completed grounded location p after memory retrieval
        return p

    def gen_x(self, p):
        # Get categorical distribution over observations from grounded location
        # If you actually want to sample observation, you need a reparaterisation trick for categorical distributions
        # Sampling would be the correct way to do this, since observations are discrete, and it's also what the TEM paper says
        # However, it looks like you could also get away with using categorical distribution directly as an approximation of the one-hot observations
        if self.hyper['do_sample']:
            x, logits = self.f_x(p) # This is a placeholder! Should be done using reparameterisation trick (like https://blog.evjang.com/2016/11/tutorial-categorical-variational.html)
        else:
            x, logits = self.f_x(p)
        # Return one-hot (or almost one-hot...) observation obtained from grounded location, and also the non-softmaxed logits
        return x, logits
        
    def inf_g(self, p_x, g_gen, x, locations):
        # Infer abstract location from the combination of [grounded location retrieved from memory by sensory experience] ...
        if self.hyper['use_p_inf']:
            # Not in paper, but makes sense from symmetry with f_x: first get g from p by "summing over sensory preferences" g = p * W_repeat^T
            g_downsampled = [torch.matmul(p_x[f], torch.t(self.hyper['W_repeat'][f])) for f in range(self.hyper['n_f'])]      
            # Then use abstract location after summing over sensory preferences as input to MLP to obtain the inferred abstract location from memory
            mu_g_mem = self.f_mu_g_mem(g_downsampled)
            # Not in paper, but this greatly improves zero-shot inference: provide the uncertainty function of the inferred abstract location with measures of memory quality
            with torch.no_grad():
                # For the first measure, use the grounded location inferred from memory to generate an observation
                x_hat, x_hat_logits = self.gen_x(p_x[0])            
                # Then calculate the error between the generated observation and the actual observation: if the memory is working well, this error should be small                
                err = utils.squared_error(x, x_hat)
            # The second measure is the vector norm of the inferred abstract location; good memories should have similar vector norms. Concatenate the two measures as input for the abstract location uncertainty function
            sigma_g_input = [torch.cat((torch.sum(g ** 2, dim=1, keepdim=True), torch.unsqueeze(err, dim=1)), dim=1) for g in mu_g_mem]            
            # Not in paper, but recommended by James for stability: get final mean of inferred abstract location by clamping activations between -1 and 1                
            mu_g_mem = self.f_g_clamp(mu_g_mem)
            # And get standard deviation/uncertainty of inferred abstract location by providing uncertainty function with memory quality measures
            sigma_g_mem = self.f_sigma_g_mem(sigma_g_input)        
        # ... and [previous abstract location and action (path integration)]  
        mu_g_path = g_gen[0]
        sigma_g_path = g_gen[1]
        # Infer abstract location by combining previous abstract location and grounded location retrieved from memory by current sensory experience
        mu_g, sigma_g = [], []
        for f in range(self.hyper['n_f']):
            if self.hyper['use_p_inf']:
                # Then get full gaussian distribution of inferred abstract location by calculating precision weighted mean
                mu, sigma = utils.inv_var_weight([mu_g_path[f], mu_g_mem[f]],[sigma_g_path[f], sigma_g_mem[f]])
            else:
                # Or simply completely ignore the inference memory here, to test if things are working
                mu, sigma = mu_g_path[f], sigma_g_path[f]
            # Append mu and sigma to list for all frequency modules
            mu_g.append(mu)
            sigma_g.append(sigma)
        # Finally (though not in paper), also add object vector cell information to inferred abstract location for environments with shiny objects
        shiny_envs = [location['shiny'] is not None for location in locations]
        if any(shiny_envs):            
            # Find for which environments the current location has a shiny object
            shiny_locations = torch.unsqueeze(torch.stack([torch.tensor(location['shiny'], dtype=torch.float) for location in locations if location['shiny'] is not None]), dim=-1)
            # Get abstract location for environments with shiny objects and feed to each of the object vector cell modules
            mu_g_shiny = self.f_mu_g_shiny([shiny_locations for _ in range(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else self.hyper['n_f'])])
            sigma_g_shiny = self.f_sigma_g_shiny([shiny_locations for _ in range(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else self.hyper['n_f'])])
            # Update only object vector modules with shiny-inferred abstract location: start from offset if object vector modules are separate
            module_start = self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0
            # Inverse variance weighting is associative, so I can just do additional inverse variance weighting to the previously obtained mu and sigma - but only for object vector cell modules!
            for f in range(module_start, self.hyper['n_f']):
                # Add inferred abstract location from shiny objects to previously obtained position, only for environments with shiny objects
                mu, sigma = utils.inv_var_weight([mu_g[f][shiny_envs,:], mu_g_shiny[f - module_start]], [sigma_g[f][shiny_envs,:], sigma_g_shiny[f - module_start]])                
                # In order to update only the environments with shiny objects, without in-place value assignment, construct a mask of shiny environments
                mask = torch.zeros_like(mu_g[f], dtype=torch.bool)
                mask[shiny_envs,:] = True
                # Use mask to update the shiny environment entries in inferred abstract locations
                mu_g[f] = mu_g[f].masked_scatter(mask,mu) 
                sigma_g[f] = sigma_g[f].masked_scatter(mask,sigma) 
        # Either sample inferred abstract location from combined (precision weighted) distribution or just take mean
        g = [mu_g[f] + sigma_g[f] * np.random.randn() if self.hyper['do_sample'] else mu_g[f] for f in range(self.hyper['n_f'])]
        # Return abstract location inferred from grounded location from memory and previous abstract location
        return g
        
    def inf_p(self, x_, g_):
        # Infer grounded location from sensory experience and inferred abstract location for each module       
        p = []
        # Use the same transformation for each frequency module: leaky relu for sparsity
        for f in range(self.hyper['n_f']):
            mu_p = self.f_p(g_[f] * x_[f]) # This is element-wise multiplication
            sigma_p = 0 # Unclear from paper (typo?). Some undefined function f that takes two arguments: f(f_n(x),g)
            # Either sample inferred grounded location or just take mean
            if self.hyper['do_sample']:
                p.append(mu_p + sigma_p * np.random.randn())
            else:
                p.append(mu_p)
        # Return new memory constructed from sensory experience and inferred abstract location
        return p

    def x_prev2x(self, x_prev, x_c):
        # Calculate factor for filtering from sigmoid of learned parameter
        alpha = [torch.nn.Sigmoid()(self.alpha[f]) for f in range(self.hyper['n_f'])]
        # Do exponential temporal filtering for each frequency modulemod
        x = [(1 - alpha[f]) * x_prev[f] + alpha[f] * x_c for f in range(self.hyper['n_f'])]
        return x
    
    def x2x_(self, x):
        # Prepare sensory input for input to memory by weighting and normalisation for each frequency module        
        # Get normalised sensory input for each frequency module 
        normalised = self.f_n(x)
        # Then reshape and reweight (use sigmoid to keep weight between 0 and 1) each frequency module separately: matrix multiplication by W_tile prepares x for outer product with g by element-wise multiplication    
        x_ = [torch.nn.Sigmoid()(self.w_p[f]) * torch.matmul(normalised[f],self.hyper['W_tile'][f]) for f in range(self.hyper['n_f'])]        
        return x_

    def g2g_(self, g):
        # Prepares abstract location for input to memory by reshaping and down-sampling for each frequency module
        # Get downsampled abstract location for each frequency module
        downsampled = self.f_g(g)
        # Then reshape and reweight each frequency module separately
        g_ = [torch.matmul(downsampled[f], self.hyper['W_repeat'][f]) for f in range(self.hyper['n_f'])]
        return g_            
        
    def f_mu_g_path(self, a_prev, g_prev, no_direc=None):
        # If there are no environments where the transition direction needs to be omitted (e.g. no shiny objects, or in inference model: set to all false
        no_direc = [False for _ in a_prev] if no_direc is None else no_direc
        # Remove all Nones from a_prev: these are walks where there was no previous action, so no step needs to be calculated for those
        a_prev_step = [a if a is not None else 0 for a in a_prev]
        # And also keep track of which walks these valid step actions are for
        a_do_step = [a != None for a in a_prev]        
        # Transform list of actions into batch of one-hot row vectors. 
        if self.hyper['has_static_action']:
            # If this world has static actions: whenever action 0 (standing still) appears, the action vector should be all zeros. All other actions should have a 1 in the label-1 entry
            a = torch.zeros((len(a_prev_step),self.hyper['n_actions'])).scatter_(1, torch.clamp(torch.tensor(a_prev_step).unsqueeze(1)-1,min=0), 1.0*(torch.tensor(a_prev_step).unsqueeze(1)>0))
        else:
            # Without static actions: each action label should become a one-hot vector for that label
            a = torch.zeros((len(a_prev_step),self.hyper['n_actions'])).scatter_(1, torch.tensor(a_prev_step).unsqueeze(1), 1.0)
        # Get vector of transition weights by feeding actions into MLP        
        D_a = self.MLP_D_a([a for _ in range(self.hyper['n_f'])])
        # Replace transition weights by non-directional transition weights in environments where transition direction needs to be omitted (can set only if any no_direc)
        for f in range(self.hyper['n_f']):
            D_a[f][no_direc,:] = self.D_no_a[f]
        # Reshape transition weight vector into transition matrix. The number of rows in the transition matrix is given by the incoming abstract location connections for each frequency module
        D_a = [torch.reshape(D_a[f_to],(-1, sum([self.hyper['n_g'][f_from] for f_from in range(self.hyper['n_f']) if self.hyper['g_connections'][f_to][f_from]]), self.hyper['n_g'][f_to])) for f_to in range(self.hyper['n_f'])]        
        # Select the frequency modules of the previous abstract location that are connected to each frequency module, to 
        g_in = [torch.unsqueeze(torch.cat([g_prev[f_from] for f_from in range(self.hyper['n_f']) if self.hyper['g_connections'][f_to][f_from]], dim=1),1) for f_to in range(self.hyper['n_f'])]        
        # Reshape transition weight vector into transition matrix. The number of rows in the transition matrix is given by the incoming abstract location connections for each frequency module
        delta = [torch.squeeze(torch.matmul(g, T)) for g, T in zip(g_in, D_a)]
        # Not in the paper, but recommended by James for stability: use inferred code as *difference* in abstract location. Calculate new abstract location from previous abstract location and difference
        g_step = [g + d if g.dim() > 1 else torch.unsqueeze(g + d, 0) for g, d in zip(g_prev, delta)]
        # Not in paper, but recommended by James for stability: clamp activations between -1 and 1
        g_step = self.f_g_clamp(g_step)
        # Build new abstract location from result of transition if there was one, or from prior on abstract location if there wasn't
        return [torch.stack([g_step[f][batch_i, :] if do_step else self.g_init[f] for batch_i, do_step in enumerate(a_do_step)]) for f in range(self.hyper['n_f'])]
    
    def f_sigma_g_path(self, a_prev, g_prev):
        # Keep track of which walks these valid step actions are for
        a_do_step = [a != None for a in a_prev]
        # Multi layer perceptron to generate standard deviation from all previous abstract locations, including those that were just initialised and not real previous locations
        from_g = self.MLP_sigma_g_path(g_prev)
        # And take exponent to get prior sigma for the walks that didn't have a previous location
        from_prior = [torch.exp(logsig) for logsig in self.logsig_g_init]
        # Now select the standard deviation generated from the previous abstract location if there was one, and the prior standard deviation on abstract location otherwise
        return [torch.stack([from_g[f][batch_i, :] if do_step else from_prior[f] for batch_i, do_step in enumerate(a_do_step)]) for f in range(self.hyper['n_f'])]

    def f_mu_g_mem(self, g_downsampled):
        # Multi layer perceptron to generate mean of abstract location from down-sampled abstract location, obtained by summing over sensory dimension of grounded location
        return self.MLP_mu_g_mem(g_downsampled)
    
    def f_sigma_g_mem(self, g_downsampled):
        # Multi layer perceptron to generate standard deviation of abstract location from down-sampled abstract location, obtained by summing over sensory dimension of grounded location
        sigma = self.MLP_sigma_g_mem(g_downsampled)
        # Not in paper, but also offset this sigma over training, so you can reduce influence of inferred p early on
        return [sigma[f] + self.hyper['p2g_scale_offset'] * self.hyper['p2g_sig_val'] for f in range(self.hyper['n_f'])]
    
    def f_mu_g_shiny(self, shiny):
        # Multi layer perceptron to generate mean of abstract location from boolean location shiny-ness
        mu_g = self.MLP_mu_g_shiny(shiny)
        # Take absolute because James wants object vector cells to be positive
        mu_g = [torch.abs(mu) for mu in mu_g]
        # Then apply clamp and leaky relu to get object vector module activations, like it's done for ground location activations
        g = self.f_p(mu_g)
        return g
        
    def f_sigma_g_shiny(self, shiny):
        # Multi layer perceptron to generate standard deviation of abstract location from boolean location shiny-ness        
        return self.MLP_sigma_g_shiny(shiny)    
    
    def f_sigma_p(self, p):
        # Multi layer perceptron to generate standard deviation of grounded location retrieval 
        return self.MLP_sigma_p(p)
    
    def f_x(self, p):
        # Calculate categorical probability distribution over observations for a given ground location
        # p has dimensions n_p[0]. We'll need to transform those to temporally filtered sensory experience, before we can decompress
        # p is the flattened (by concatenating rows - like reading sentences) outer product of g and x (p = g^T * x). 
        # Therefore to get the sensory experience x for a grounded location p, sum over all abstract locations g for each component of x
        # That's what the paper means when it says "sum over entorhinal preferences". It can be done with the transpose of W_tile
        x = self.w_x * torch.matmul(p, torch.t(self.hyper['W_tile'][0])) + self.b_x
        # Then we need to decompress the temporally filtered sensory experience into a single current experience prediction     
        logits = self.f_c_star(x)
        # We'll keep both the logits (domain -inf, inf) and probabilities (domain 0, 1) because both are needed later on
        probability = utils.softmax(logits) 
        return probability, logits
    
    def f_c_star(self, compressed):
        # Multi layer perceptron to decompress sensory experience at highest frequency 
        return self.MLP_c_star(compressed)
    
    def f_c(self, decompressed):
        # Compress sensory observation from one-hot provided by world to two-hot for ease of computation
        return torch.stack([self.hyper['two_hot_table'][i] for i in torch.argmax(decompressed, dim=1)], dim=0)
    
    def f_n(self, x):
        # Normalise sensory observation for each frequency module
        normalised = [utils.normalise(utils.relu(x[f] - torch.mean(x[f]))) for f in range(self.hyper['n_f'])]
        return normalised
    
    def f_g(self, g):
        # Downsample abstract location for each frequency module
        downsampled = [torch.matmul(g[f], self.hyper['g_downsample'][f]) for f in range(self.hyper['n_f'])]
        return downsampled    
    
    def f_g_clamp(self, g):
        # Calculate activation for abstract location, thresholding between -1 and 1
        activation = [torch.clamp(g_f, min=-1, max=1) for g_f in g]
        return activation    
    
    def f_p(self, p):
        # Calculate activation for inferred grounded location, using a leaky relu for sparsity. Either apply to full multi-frequency grounded location or single frequency module
        activation = [utils.leaky_relu(torch.clamp(p_f, min=-1, max=1)) for p_f in p] if type(p) is list else utils.leaky_relu(torch.clamp(p, min=-1, max=1)) 
        return activation        
    
    def attractor(self, p_query, M, retrieve_it_mask=None):        
        # Retreive grounded location from attractor network memory with weights M by pattern-completing query    
        # For example, initial attractor input can come from abstract location (g_) or sensory experience (x_)                        
        # Start by flattening query grounded locations across frequency modules
        h_t = torch.cat(p_query, dim=1)
        # Apply activation function to initial memory index
        h_t = self.f_p(h_t)        
        # Hierarchical retrieval (not in paper) is implemented by early stopping retrieval for low frequencies, using a mask. If not specified: initialise mask as all 1s
        retrieve_it_mask = [torch.ones(sum(self.hyper['n_p'])) for _ in range(self.hyper['n_p'])] if retrieve_it_mask is None else retrieve_it_mask
        # Iterate attractor dynamics to do pattern completion
        for tau in range(self.hyper['i_attractor']):
            # Apply one iteration of attractor dynamics, but only where there is a 1 in the mask. NB retrieve_it_mask entries have only one row, but are broadcasted to batch_size
            h_t = (1-retrieve_it_mask[tau])*h_t + retrieve_it_mask[tau]*(self.f_p(self.hyper['kappa'] * h_t + torch.squeeze(torch.matmul(torch.unsqueeze(h_t,1), M))))
        # Make helper list of cumulative neurons per frequency module for grounded locations
        n_p = np.cumsum(np.concatenate(([0],self.hyper['n_p'])))                
        # Now re-cast the grounded location into different frequency modules, since memory retrieval turned it into one long vector
        p = [h_t[:,n_p[f]:n_p[f+1]] for f in range(self.hyper['n_f'])]
        return p;
    
    def hebbian(self, M_prev, p_inferred, p_generated, do_hierarchical_connections=True):
        # Create new ground memory for attractor network by setting weights to outer product of learned vectors
        # p_inferred corresponds to p in the paper, and p_generated corresponds to p^. 
        # The order of p + p^ and p - p^ is reversed since these are row vectors, instead of column vectors in the paper.
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p_inferred + p_generated, 2),torch.unsqueeze(p_inferred - p_generated,1)))
        # Multiply by connection vector, e.g. only keeping weights from low to high frequencies for hierarchical retrieval
        if do_hierarchical_connections:
            M_new = M_new * self.hyper['p_update_mask']
        # Store grounded location in attractor network memory with weights M by Hebbian learning of pattern        
        M = torch.clamp(self.hyper['lambda'] * M_prev + self.hyper['eta'] * M_new, min=-1, max=1)
        return M;

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation=(torch.nn.functional.elu, None), hidden_dim=None, bias=(True, True)):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(MLP, self).__init__()        
        # Check if this network consists of module: are input and output dimensions lists? If not, make them (but remember it wasn't)
        if type(in_dim) is list:
            self.is_list = True
        else:
            in_dim = [in_dim]
            out_dim = [out_dim]
            self.is_list = False
        # Find number of modules
        self.N = len(in_dim)
        # Create weights (input->hidden, hidden->output) for each module
        self.w = torch.nn.ModuleList([])
        for n in range(self.N):
            # If number of hidden dimensions is not specified: mean of input and output
            if hidden_dim is None:
                hidden = int(np.mean([in_dim[n],out_dim[n]]))
            else:
                hidden = hidden_dim[n] if self.is_list else hidden_dim         
            # Each module has two sets of weights: input->hidden and hidden->output
            self.w.append(torch.nn.ModuleList([torch.nn.Linear(in_dim[n], hidden, bias=bias[0]), torch.nn.Linear(hidden, out_dim[n], bias=bias[1])]))
        # Copy activation function for hidden layer and output layer
        self.activation = activation
        # Initialise all weights
        with torch.no_grad():
            for from_layer in range(2):
                for n in range(self.N):
                    # Set weights to xavier initalisation
                    torch.nn.init.xavier_normal_(self.w[n][from_layer].weight)
                    # Set biases to 0
                    if bias[from_layer]:
                        self.w[n][from_layer].bias.fill_(0.0)        
    
    def set_weights(self, from_layer, value):
        # If single value is provided: copy it for each module
        if type(value) is not list:
            input_value = [value for n in range(self.N)]
        else:
            input_value = value
        # Run through all modules and set weights starting from requested layer to the specified value
        with torch.no_grad():
            # MLP is setup as follows: w[module][layer] is Linear object, w[module][layer].weight is Parameter object for linear weights, w[module][layer].weight.data is tensor of weight values
            for n in range(self.N):
                # If a tensor is provided: copy the tensor to the weights                
                if type(input_value[n]) is torch.Tensor:
                    self.w[n][from_layer].weight.copy_(input_value[n])                    
                # If only a single value is provided: set that value everywhere
                else:
                    self.w[n][from_layer].weight.fill_(input_value[n]) 
        
    def forward(self, data):
        # Make input data into list, if this network doesn't consist of modules
        if self.is_list:
            input_data = data
        else:
            input_data = [data]
        # Run input through network for each module
        output = []
        for n in range(self.N):
            # Pass through first weights from input to hidden layer
            module_output = self.w[n][0](input_data[n])
            # Apply hidden layer activation
            if self.activation[0] is not None:
                module_output = self.activation[0](module_output)
            # Pass through second weights from hidden to output layer
            module_output = self.w[n][1](module_output)
            # Apply output layer activation
            if self.activation[1] is not None:
                module_output = self.activation[1](module_output)
            # Transpose output again to go back to column vectors instead of row vectors
            output.append(module_output) 
        # If this network doesn't consist of modules: select output from first module to return
        if not self.is_list:
            output = output[0]
        # And return output
        return output     

class LSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers = 1, n_a = 4):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(LSTM, self).__init__()
        # LSTM layer
        self.lstm = torch.nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        # Hidden to output
        self.lin = torch.nn.Linear(hidden_dim, out_dim)
        # Copy number of actions, will be needed for input data vector
        self.n_a = n_a
        
    def forward(self, data, prev_hidden = None):
        # If previous hidden and cell state are not provided: initialise them randomly
        if prev_hidden is None:
            hidden_state = torch.randn(self.lstm.num_layers, data.shape[0], self.lstm.hidden_size)
            cell_state = torch.randn(self.lstm.num_layers, data.shape[0], self.lstm.hidden_size)
            prev_hidden = (hidden_state, cell_state)
        # Run input through lstm
        lstm_out, lstm_hidden = self.lstm(data, prev_hidden)
        # Apply linear network to lstm output to get output: prediction at each timestep
        lin_out = self.lin(lstm_out)
        # And since we want a one-hot prediciton: do softmax on top
        out = utils.softmax(lin_out)
        # Return output and hidden state
        return out, lstm_hidden

    def prepare_data(self, data_in):
        # Transform list of actions of each step into batch of one-hot row vectors
        actions = [torch.zeros((len(step[2]),self.n_a)).scatter_(1, torch.tensor(step[2]).unsqueeze(1), 1.0) for step in data_in]
        # Concatenate observation and action together along column direction in each step
        vectors = [torch.cat((step[1], action), dim=1) for step, action in zip(data_in, actions)]
        # Then stack all these together along the second dimension, which is sequence length
        data = torch.stack(vectors, dim=1)
        # Return data in [batch_size, seq_len, input_dim] dimension as expected by lstm        
        return data  

class Iteration:
    def __init__(self, g = None, x = None, a = None, L = None, M = None, g_gen = None, p_gen = None, x_gen = None, x_logits = None, x_inf = None, g_inf = None, p_inf = None):
        # Copy all inputs
        self.g = g
        self.x = x
        self.a = a
        self.L = L
        self.M = M
        self.g_gen = g_gen
        self.p_gen = p_gen
        self.x_gen = x_gen
        self.x_logits = x_logits
        self.x_inf = x_inf
        self.g_inf = g_inf
        self.p_inf = p_inf

    def correct(self):
        # Detach observation and all predictions
        observation = self.x.detach().numpy()
        predictions = [tensor.detach().numpy() for tensor in self.x_gen]
        # Did the model predict the right observation in this iteration?
        accuracy = [np.argmax(prediction, axis=-1) == np.argmax(observation, axis=-1) for prediction in predictions]
        return accuracy
    
    def detach(self):
        # Detach all tensors contained in this iteration
        self.L = [tensor.detach() for tensor in self.L]
        self.M = [tensor.detach() for tensor in self.M]
        self.g_gen = [tensor.detach() for tensor in self.g_gen]
        self.p_gen = [tensor.detach() for tensor in self.p_gen]
        self.x_gen = [tensor.detach() for tensor in self.x_gen]
        self.x_inf = [tensor.detach() for tensor in self.x_inf]
        self.g_inf = [tensor.detach() for tensor in self.g_inf]
        self.p_inf = [tensor.detach() for tensor in self.p_inf]
        # Return self after detaching everything
        return self
        