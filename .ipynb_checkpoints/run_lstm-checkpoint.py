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
# Own module imports
import world
import parameters
import model
import plot

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Create world: 4x4 grid with actions [North, East, South, West] with random policy, with 15 sensory experiences
grid = world.World('./graphs/5x5.json', 45)

# Initalise hyperparameters for model
params = parameters.parameters(grid)

# Create lstm, to see if that learns well
lstm = model.LSTM(params['n_x'] + params['n_actions'], 100, params['n_x'], n_a = params['n_actions'])

# Create set of training worlds, as many as there are batches
environments = [world.World('./graphs/5x5.json', 45) for batch in range(params['n_batches'])]

# Create walks on each world
walks = [env.generate_walks(params['walk_length'], params['n_walks']) for env in environments]

# Create batched walks: instead of having walks separated by environment, collect them by environment
batches = [[[[],[],[]] for l in range(params['walk_length'])] for w in range(params['n_walks'])]
for env in walks:
    for i_walk, walk in enumerate(env):
        for i_step, step in enumerate(walk):
            for i_comp, component in enumerate(step):
                # Append state, observation, action across environments
                batches[i_walk][i_step][i_comp].append(component)
# Stack all observations into tensors along the first dimension for batch processing
for i_walk, walk in enumerate(batches):
    for i_step, step in enumerate(walk):
        batches[i_walk][i_step][1] = torch.stack(step[1], dim=0)

# Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
writer = SummaryWriter()

# Make an ADAM optimizer for the LSTM
adam = torch.optim.Adam(lstm.parameters(), lr = 0.1)

# Create learning rate scheduler that reduces learning rate over training
lr_factor = lambda epoch: 0.75
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(adam,lr_factor)

# Train LSTM
for i, walk in enumerate(batches):
    # Don't feed walk all at once; instead, feed limited number of forward rollouts, then backprop through time
    chunks = [[i, min(i + params['n_rollout'],len(walk))] for i in range(0, len(walk), params['n_rollout'])]
    # Initialise the previous hidden state as none: at the beginning of a walk, there is no hidden state yet
    prev_hidden = None
    # Run through all chunks that we are going to backprop for
    for j, [start, stop] in enumerate(chunks):
        # Get start time for function timing
        start_time = time.time()
        # Prepare data for feeding into lstm
        data = lstm.prepare_data(walk[start:stop])
        # Forward-pass this data through the network
        predictions, prev_hidden = lstm(data, prev_hidden)
        # Calculate loss from forward pass: difference between predicted and real observation at each step
        loss = torch.nn.BCELoss()(predictions[:,:-1,:], data[:,1:,:params['n_x']])
        # Reset gradients
        adam.zero_grad()
        # Do backward pass to calculate gradients with respect to total loss of this chunk
        loss.backward(retain_graph=True)    
        # Then do optimiser step to update parameters of model
        adam.step()
        # And detach previous hidden state to prevent gradients going back forever
        prev_hidden = tuple([hidden.detach() for hidden in prev_hidden])
        # Calculate accuracy: how often was the best guess from the predictions correct?
        accuracy = torch.mean((torch.argmax(data[:,1:,:params['n_x']], dim=-1) == torch.argmax(predictions[:,:-1,:], dim=-1)).type(torch.float)).numpy()
        # Show progress
        if j % 10 == 0:
            print('Finished walk {:d}, chunk {:d} in {:.2f} seconds.\n'.format(i,j,time.time()-start_time) +
                  'Loss: {:.2f}, accuracy: {:.2f} %'.format(loss.detach().numpy(), accuracy * 100.0))
        # Also write progress to tensorboard
        writer.add_scalar('Walk ' + str(i + 1) + '/Loss', loss.detach().numpy(), j)
        writer.add_scalar('Walk ' + str(i + 1) + '/Accuracy', accuracy * 100, j) 
    # Also step the learning rate down after each walk
    scheduler.step()
