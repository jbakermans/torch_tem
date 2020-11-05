#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:57:45 2020

@author: jacobb
"""

# Standard library imports
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import glob, os, shutil
import importlib.util
# Own module imports
import world
import utils
import parameters
import model as model    

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Either load a trained model and continue training, or start afresh
load_existing_model = False;
if load_existing_model:
    # Choose which trained model to load
    date = '2020-10-06' # 2020-07-05 run 0 for successful node agent
    run = '2'
    i_start = 40
    
    # Set all paths from existing run 
    run_path, train_path, model_path, save_path, script_path, envs_path = utils.set_directories(date, run)
    
    # Load the model: use import library to import module from specified path
    model_spec = importlib.util.spec_from_file_location("model", script_path + '/model.py')
    model = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model)
    
    # Load the parameters of the model
    params = torch.load(model_path + '/params_' + str(i_start) + '.pt')
    # But certain parameters (like total nr of training iterations) may need to be copied from the current set of parameters
    new_params = {'train_it':40000}
    # Update those in params
    for key in new_params:
        params[key] = new_params[key]
    
    # Create a new tem model with the loaded parameters
    tem = model.Model(params)
    # Load the model weights after training
    model_weights = torch.load(model_path + '/tem_' + str(i_start) + '.pt')
    # Set the model weights to the loaded trained model weights
    tem.load_state_dict(model_weights)
    
    # Make list of all the environments that this model was trained on
    envs = list(glob.iglob(envs_path + '/*'))
    
    # And increase starting iteration by 1, since the loaded model already carried out the current starting iteration
    i_start = i_start + 1
else:
    # Start training from step 0
    i_start = 0
    
    # Create directories for storing all information about the current run
    run_path, train_path, model_path, save_path, script_path, envs_path = utils.make_directories()
    # Save all python files in current directory to script directory
    files = glob.iglob(os.path.join('.', '*.py'))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, os.path.join(script_path, file))        
            
    # Initalise hyperparameters for model
    params = parameters.parameters()
    # Save parameters
    np.save(os.path.join(save_path, 'params'), params)       
    
    # And create instance of TEM with those parameters
    tem = model.Model(params)
    
    # Create list of environments that we will sample from during training to provide TEM with trajectory input
    envs = ['./envs/5x5.json']
    # Save all environment files that are being used in training in the script directory
    for file in set(envs):
        shutil.copy2(file, os.path.join(envs_path, os.path.basename(file)))    

# Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
writer = SummaryWriter(train_path)
# Create a logger to write log output to file
logger = utils.make_logger(run_path)

# Make an ADAM optimizer for TEM
adam = torch.optim.Adam(tem.parameters(), lr = params['lr_max'])

# Make set of environments: one for each batch, randomly choosing to use shiny objects or not
environments = [world.World(graph, randomise_observations=True, shiny=(params['shiny'] if np.random.rand() < params['shiny_rate'] else None)) for graph in np.random.choice(envs,params['batch_size'])]
# Initialise whether a state has been visited for each world
visited = [[False for _ in range(env.n_locations)] for env in environments]
# And make a single walk for each environment, where walk lengths can be any between the min and max length to de-sychronise world switches
walks = [env.generate_walks(params['n_rollout']*np.random.randint(params['walk_it_min'], params['walk_it_max']), 1)[0] for env in environments]
# Initialise the previous iteration as None: we start from the beginning of the walk, so there is no previous iteration yet
prev_iter = None

# Train TEM on walks in different environment
for i in range(i_start, params['train_it']):
    
    # Get start time for function timing
    start_time = time.time()
    # Get updated parameters for this backprop iteration
    eta_new, lambda_new, p2g_scale_offset, lr, walk_length_center, loss_weights = parameters.parameter_iteration(i, params)
    # Update eta and lambda
    tem.hyper['eta'] = eta_new
    tem.hyper['lambda'] = lambda_new
    # Update scaling of offset for variance of inferred grounded position
    tem.hyper['p2g_scale_offset'] = p2g_scale_offset
    # Update learning rate (the neater torch-way of doing this would be a scheduler, but this is quick and easy)
    for param_group in adam.param_groups:
        param_group['lr'] = lr            
    
    # Make an empty chunk that will be fed to TEM in this backprop iteration
    chunk = []
    # For each environment: fill chunk by popping the first batch_size steps of the walk
    for env_i, walk in enumerate(walks):
        # Make sure this walk has enough steps in it for a whole backprop iteration
        if len(walk) < params['n_rollout']:
            # If it doesn't: create a new environment 
            environments[env_i] = world.World(envs[np.random.randint(len(envs))], randomise_observations=True, shiny=(params['shiny'] if np.random.rand() < params['shiny_rate'] else None))
            # Initialise whether a state has been visited for each world
            visited[env_i] = [False for _ in range(environments[env_i].n_locations)]            
            # Generate a new walk on that environment
            walk = environments[env_i].generate_walks(params['n_rollout']*np.random.randint(walk_length_center - params['walk_it_window'] * 0.5, walk_length_center + params['walk_it_window'] * 0.5), 1)[0]
            # And store it in walks array
            walks[env_i] = walk
            # Finally, set the action of the previous iteration for this environment to zero, to indicate that this is a new walk
            prev_iter[0].a[env_i] = None
            # Log progress
            logger.info('Iteration {:d}: new walk of length {:d} for batch entry {:d}'.format(i, len(walk), env_i))                
        # Now pop the first n_rollout steps from this walk and append them to the chunk
        for step in range(params['n_rollout']):
            # For the first environment: simply copy the components (g, x, a) of each step
            if len(chunk) < params['n_rollout']:
                chunk.append([[comp] for comp in walk.pop(0)])
            # For all next environments: add the components to the existing list of components for each step
            else:
                for comp_i, comp in enumerate(walk.pop(0)):
                    chunk[step][comp_i].append(comp)
    # Stack all observations (x, component 1) into tensors along the first dimension for batch processing
    for i_step, step in enumerate(chunk):
        chunk[i_step][1] = torch.stack(step[1], dim=0)    
        
    # Forward-pass this walk through the network
    forward = tem(chunk, prev_iter)    
    
    # Accumulate loss from forward pass
    loss = torch.tensor(0.0)
    # Make vector for plotting losses
    plot_loss = 0
    # Collect all losses 
    for step in forward:            
        # Make list of losses included in this step
        step_loss = []        
        # Only include loss for locations that have been visited before
        for env_i, env_visited in enumerate(visited):
            if env_visited[step.g[env_i]['id']]:
                step_loss.append(loss_weights*torch.stack([l[env_i] for l in step.L]))
            else:
                env_visited[step.g[env_i]['id']] = True
        # Stack losses in this step along first dimension, then average across that dimension to get mean loss for this step
        step_loss = torch.tensor(0) if not step_loss else torch.mean(torch.stack(step_loss, dim=0), dim=0)
        # Save all separate components of loss for monitoring
        plot_loss = plot_loss + step_loss.detach().numpy()
        # And sum all components, then add them to total loss of this step
        loss = loss + torch.sum(step_loss)

    # Reset gradients
    adam.zero_grad()
    # Do backward pass to calculate gradients with respect to total loss of this chunk
    loss.backward(retain_graph=True)
    # Then do optimiser step to update parameters of model
    adam.step()
    # Update the previous iteration for the next chunk with the final step of this chunk, removing all operation history
    prev_iter = [forward[-1].detach()]
    
    # Compute model accuracies
    acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
    acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]        
    # Log progress 
    if i % 10 == 0:
        # Write series of messages to logger from this backprop iteration
        logger.info('Finished backprop iter {:d} in {:.2f} seconds.'.format(i,time.time()-start_time))
        logger.info('Loss: {:.2f}. <p_g> {:.2f} <p_x> {:.2f} <x_gen> {:.2f} <x_g> {:.2f} <x_p> {:.2f} <g> {:.2f} <reg_g> {:.2f} <reg_p> {:.2f}'.format(loss.detach().numpy(), *plot_loss))
        logger.info('Accuracy: <p> {:.2f}% <g> {:.2f}% <gt> {:.2f}%'.format(acc_p, acc_g, acc_gt))
        logger.info('Parameters: <max_hebb> {:.2f} <eta> {:.2f} <lambda> {:.2f} <p2g_scale_offset> {:.2f}'.format(np.max(np.abs(prev_iter[0].M[0].numpy())), tem.hyper['eta'], tem.hyper['lambda'], tem.hyper['p2g_scale_offset']))
        logger.info('Weights:' + str([w for w in loss_weights.numpy()]))
        logger.info(' ')
        # Also write progress to tensorboard, and all loss components. Order: [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]
        writer.add_scalar('Losses/Total', loss.detach().numpy(), i)
        writer.add_scalar('Losses/p_g', plot_loss[0], i)
        writer.add_scalar('Losses/p_x', plot_loss[1], i)
        writer.add_scalar('Losses/x_gen', plot_loss[2], i)
        writer.add_scalar('Losses/x_g', plot_loss[3], i)
        writer.add_scalar('Losses/x_p', plot_loss[4], i)
        writer.add_scalar('Losses/g', plot_loss[5], i)
        writer.add_scalar('Losses/reg_g', plot_loss[6], i)
        writer.add_scalar('Losses/reg_p', plot_loss[7], i)
        writer.add_scalar('Accuracies/p', acc_p, i)
        writer.add_scalar('Accuracies/g', acc_g, i)
        writer.add_scalar('Accuracies/gt', acc_gt, i)
    # Also store the internal state (all learnable parameters) and the hyperparameters periodically 
    if i % 1000 == 0:
        torch.save(tem.state_dict(), model_path + '/tem_' + str(i) + '.pt')
        torch.save(tem.hyper, model_path + '/params_' + str(i) + '.pt')

# Save the final state of the model after training has finished
torch.save(tem.state_dict(), model_path + '/tem_' + str(i) + '.pt')
torch.save(tem.hyper, model_path + '/params_' + str(i) + '.pt')