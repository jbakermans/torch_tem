#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:35:57 2020

@author: jacobb
"""

# Standard library imports
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import importlib.util
# Own module imports. Note how model module is not imported, since we'll used the model from the training run
import world
import analyse
import plot
import os
import time
from datetime import datetime

def test_model(date, run, env_to_plot, index='0', which_plots=['zero-shot', 'rate_maps', 'acc_to_from', 'occupancy'], seed=0, save_dir=None, columns=25):
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Choose which trained model to load
    #date = '2022-03-07' # 2020-10-13 run 0 for successful node agent
    #run = '0'
    #index = '0'

    # Load the model: use import library to import module from specified pathr
    model_spec = importlib.util.spec_from_file_location("model", '/cumulus/cristofer/TEM_data/' + date + '/run' + run + '/script/model.py')
    model = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model)

    # Load the parameters of the model
    params = torch.load('/cumulus/cristofer/TEM_data/' + date + '/run' + run + '/model/params_' + index + '.pt')
    # Create a new tem model with the loaded parameters
    tem = model.Model(params)
    # Load the model weights after training
    model_weights = torch.load('/cumulus/cristofer/TEM_data/' + date + '/run' + run + '/model/tem_' + index + '.pt')
    # Set the model weights to the loaded trained model weights
    tem.load_state_dict(model_weights)
    # Make sure model is in evaluate mode (not crucial because it doesn't currently use dropout or batchnorm layers)
    tem.eval()

    # Make list of all the environments that this model was trained on
    envs = list(glob.iglob('/cumulus/cristofer/TEM_data/' + date + '/run' + run + '/script/envs/*'))
    # Set which environments will include shiny objects
    shiny_envs = [False, False, False, False]
    # Set the number of walks to execute in parallel (batch size)
    n_walks = len(shiny_envs)
    # Select environments from the environments included in training
    environments = [world.World(graph, randomise_observations=params['randomise_observations'], shiny=(params['shiny'] if shiny_envs[env_i] else None), specify_behavior=params['specify_behavior'], behavior_type=params['behavior_type'], seed=seed) for env_i, graph in enumerate(np.random.choice(envs, n_walks))]
    # Determine the length of each walk
    walk_len = np.median([env.n_locations * 50 for env in environments]).astype(int)
    # And generate walks for each environment
    walks = [env.generate_walks(walk_len, 1)[0] for env in environments]

    # Generate model input from specified walk and environment: group steps from all environments together to feed to model in parallel
    model_input = [[[[walks[i][j][k]][0] for i in range(len(walks))] for k in range(3)] for j in range(walk_len)]
    for i_step, step in enumerate(model_input):
        model_input[i_step][1] = torch.stack(step[1], dim=0)

    # Run a forward pass through the model using this data, without accumulating gradients
    with torch.no_grad():
        forward = tem(model_input, prev_iter=None)

    # Decide whether to include stay-still actions as valid occasions for inference
    include_stay_still = False
    
    # Choose which environment to plot
    #env_to_plot = 1
    # And when averaging environments, e.g. for calculating average accuracy, decide which environments to include
    envs_to_avg = shiny_envs if shiny_envs[env_to_plot] else [not shiny_env for shiny_env in shiny_envs]

    #print('type(forward): {0}\n'.format(type(forward)))
    #print('forward: {0}'.format(forward))
    if 'zero-shot' in which_plots:
        # Compare trained model performance to a node agent and an edge agent
        correct_model, correct_node, correct_edge = analyse.compare_to_agents(forward, tem, environments, include_stay_still=include_stay_still)
        
        # Analyse occurrences of zero-shot inference: predict the right observation arriving from a visited node with a new action
        zero_shot = analyse.zero_shot(forward, tem, environments, include_stay_still=include_stay_still)
        
        # Plot results of agent comparison and zero-shot inference analysis
        filt_size = 41
        plt.figure()
        plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_model) if envs_to_avg[env_i]]),0)[1:], filt_size), label='tem')
        plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_node) if envs_to_avg[env_i]]),0)[1:], filt_size), label='node')
        plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_edge) if envs_to_avg[env_i]]),0)[1:], filt_size), label='edge')
        plt.ylim(0, 1.1)
        plt.legend()
        
        performance = np.mean([np.mean(env) for env_i, env in enumerate(zero_shot) if envs_to_avg[env_i]]) * 100
        performance = f'{performance:.4f}'
        
        plt.title('{0} zero-shot inference: '.format(environments[0].env_type) + performance + '%; index {0}'.format(index))
        if save_dir:
            plt.savefig(save_dir + 'index_{0}_seed_{1}_performance.svg'.format(index, seed))
            plt.savefig(save_dir + 'index_{0}_seed_{1}_performance.png'.format(index, seed))
        
        plt.show()
        
            
    if 'rate_maps' in which_plots:
        # Generate rate maps
        g, p = analyse.rate_map(forward, tem, environments)
        #print('p: {0}'.format(p))
        # Plot rate maps for all cells
        plot.plot_cells(p[env_to_plot], g[env_to_plot], environments[env_to_plot], n_f_ovc=(params['n_f_ovc'] if 'n_f_ovc' in params else 0), columns=columns, save_dir=save_dir, index=index, seed=seed)
    
    if 'acc_to_from' in which_plots:
        # Calculate accuracy leaving from and arriving to each location
        from_acc, to_acc = analyse.location_accuracy(forward, tem, environments)
        
        # Plot accuracy separated by location
        plt.figure()
        ax = plt.subplot(1,2,1)
        plot.plot_map(environments[env_to_plot], np.array(to_acc[env_to_plot]), ax)
        ax.set_title('Accuracy to location')
        ax = plt.subplot(1,2,2)
        plot.plot_map(environments[env_to_plot], np.array(from_acc[env_to_plot]), ax)
        ax.set_title('Accuracy from location')
    
    if 'occupancy' in which_plots:
        # Generate occupancy maps: how much time TEM spends at every location
        occupation = analyse.location_occupation(forward, tem, environments)
        
        # Plot occupation per location, then add walks on top
        ax = plot.plot_map(environments[env_to_plot], np.array(occupation[env_to_plot])/sum(occupation[env_to_plot])*environments[env_to_plot].n_locations, 
                           min_val=0, max_val=2, ax=None, shape='square', radius=1/np.sqrt(environments[env_to_plot].n_locations))
        ax = plot.plot_walk(environments[env_to_plot], walks[env_to_plot], ax=ax, n_steps=max(1, int(len(walks[env_to_plot])/500)))
        plt.title('Walk and average occupation')
        

        
# Take model_directory_path and test all of the model params that haven't
# been tested yet
def test_all_untested_models(model_directory_path, date, run, env_to_plot, which_plots=['zero-shot', 'rate_maps'], seed=0, columns=15, already_tested=[], save_dir=None):
    
    
    # Get the index for all finished tem_files in model_directory_path
    try:
        files = [f for f in os.listdir(model_directory_path) if os.path.isfile(os.path.join(model_directory_path, f))]
        tem_files = [f for f in files if f[0] == 't']
        # List of all indexes in directory
        param_indexes = [int(f[4:-3]) for f in tem_files]
        param_indexes.sort()
        param_indexes = [str(param_index) for param_index in param_indexes]
    except FileNotFoundError:
        raise Exception('Invalid model_directory_path')

    # If there are more param_indexes than previously
    if len(param_indexes) > len(already_tested):
        # Get slice that only contains indexes that have not been tested
        untested_param_indexes = param_indexes[len(already_tested):]
        print('Found these untested indexes: {0}\nAttempting to test all at {1}'.format(untested_param_indexes, datetime.now().strftime("%H:%M:%S, %Y-%m-%d")))
        for param_index in untested_param_indexes:
            
            test_model(date=date, run=run, env_to_plot=env_to_plot, index=param_index, which_plots=which_plots, seed=seed, save_dir=save_dir, columns=columns)

            already_tested.append(param_index)
            
    # Return already tested items so that to input in next call
    return already_tested

# Regularly test new iterations of model for test_period amount of time
def test_recurring(save_dir, model_directory_path, date, run, env_to_plot, which_plots=['zero-shot', 'rate_maps'], seed=0, columns=15, already_tested=[], test_period=43200, test_interval=36000):
    
    # If nothing has yet been saved, make a directory to hold the files
    if len(already_tested) == 0:
        os.makedirs(save_dir)
        
    start_time = time.time()
    already_tested = test_all_untested_models(model_directory_path=model_directory_path, date=date, run=run, env_to_plot=env_to_plot, which_plots=which_plots, seed=seed, columns=columns, already_tested=already_tested, save_dir=save_dir)

    while time.time() - start_time < test_period:
        
        already_tested = test_all_untested_models(model_directory_path=model_directory_path, date=date, run=run, env_to_plot=env_to_plot, which_plots=which_plots, seed=seed, columns=columns, already_tested=already_tested, save_dir=save_dir)
        time.sleep(test_interval - ((time.time() - start_time) % test_interval))

    print('Finished testing at {0}'.format(datetime.now().strftime("%H:%M:%S, %Y-%m-%d")))
    
    return already_tested

def get_forward(date, run, env_to_plot, index='0', seed=0):
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Choose which trained model to load
    #date = '2022-03-07' # 2020-10-13 run 0 for successful node agent
    #run = '0'
    #index = '0'

    # Load the model: use import library to import module from specified path
    model_spec = importlib.util.spec_from_file_location("model", '/cumulus/cristofer/TEM_data/' + date + '/run' + run + '/script/model.py')
    model = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model)

    # Load the parameters of the model
    params = torch.load('/cumulus/cristofer/TEM_data/' + date + '/run' + run + '/model/params_' + index + '.pt')
    # Create a new tem model with the loaded parameters
    tem = model.Model(params)
    # Load the model weights after training
    model_weights = torch.load('/cumulus/cristofer/TEM_data/' + date + '/run' + run + '/model/tem_' + index + '.pt')
    # Set the model weights to the loaded trained model weights
    tem.load_state_dict(model_weights)
    # Make sure model is in evaluate mode (not crucial because it doesn't currently use dropout or batchnorm layers)
    tem.eval()

    # Make list of all the environments that this model was trained on
    envs = list(glob.iglob('/cumulus/cristofer/TEM_data/' + date + '/run' + run + '/script/envs/*'))
    # Set which environments will include shiny objects
    shiny_envs = [False, False, False, False]
    # Set the number of walks to execute in parallel (batch size)
    n_walks = len(shiny_envs)
    # Select environments from the environments included in training
    environments = [world.World(graph, randomise_observations=params['randomise_observations'], shiny=(params['shiny'] if shiny_envs[env_i] else None), specify_behavior=params['specify_behavior'], behavior_type=params['behavior_type'], seed=seed) for env_i, graph in enumerate(np.random.choice(envs, n_walks))]
    # Determine the length of each walk
    walk_len = np.median([env.n_locations * 50 for env in environments]).astype(int)
    # And generate walks for each environment
    walks = [env.generate_walks(walk_len, 1)[0] for env in environments]

    # Generate model input from specified walk and environment: group steps from all environments together to feed to model in parallel
    model_input = [[[[walks[i][j][k]][0] for i in range(len(walks))] for k in range(3)] for j in range(walk_len)]
    for i_step, step in enumerate(model_input):
        model_input[i_step][1] = torch.stack(step[1], dim=0)

    # Run a forward pass through the model using this data, without accumulating gradients
    with torch.no_grad():
        forward = tem(model_input, prev_iter=None)

    return forward, tem, environments