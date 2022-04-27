#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:33:06 2020

@author: jacobb
"""
import json
import numpy as np
import torch
import copy
from scipy.sparse.csgraph import shortest_path

# Functions for generating data that TEM trains on: sequences of [state,observation,action] tuples

class World:
    def __init__(self, env, randomise_observations=False, replace=False, randomise_policy=False, shiny=None, specify_behavior=False, behavior_type=None, seed=0):
        # If the environment is provided as a filename: load the corresponding file. If it's no filename, it's assumed to be an environment dictionary
        if type(env) == str or type(env) == np.str_:
            # Filename provided, load graph from json file
            file = open(env, 'r') 
            json_text = file.read()
            env = json.loads(json_text)
            file.close()
        
        # Now env holds a dictionary that describes this world
        try:
            # Copy expected fields to object attributes
            self.env_type = env['env_type']
            self.adjacency = env['adjacency']
            self.locations = env['locations']
            self.n_actions = env['n_actions']
            self.n_locations = env['n_locations']
            self.n_observations = env['n_observations']
            self.specify_behavior = specify_behavior
            self.behavior_type = behavior_type
            self.replace = replace
            self.seed = seed
            
            if self.env_type == 'loop_laps':
                # If this is a loop_laps track, set the number of laps and length of each lap
                self.lap_num = env['lap_num']
                self.lap_len = env['lap_len']
                
        except (KeyError, TypeError) as e:
            # If any of the expected fields is missing: treat this as an invalid environment
            print('Invalid environment: bad dictionary\n', e)            
            # Initialise all environment fields for an empty environment
            self.adjacency = []
            self.locations = []
            self.n_actions = 0
            self.n_locations = 0
            self.n_observations = 0
        
        # If requested: shuffle observations from original assignments
        if randomise_observations:
            self.observations_randomise()
        
        # If requested: randomise policy by setting equal probability for each action
        if randomise_policy:
            self.policy_random()

        # Copy the shiny input
        self.shiny = copy.deepcopy(shiny)
        # If there's no shiny data provided: initialise this world as a non-shiny environement
        if self.shiny is None:
            # TEM needs to know that this is a non-shiny environment (e.g. for providing actions to generative model), so set shiny to None for each location
            for location in self.locations:
                location['shiny'] = None
        # If shiny data is provided: initialise shiny properties
        else:
            # Initially make all locations non-shiny
            for location in self.locations:
                location['shiny'] = False            
            # Calculate all graph distances, since shiny objects aren't allowed to be too close together
            dist_matrix = shortest_path(csgraph=np.array(self.adjacency), directed=False)
            # Initialise the list of shiny locations as empty
            self.shiny['locations'] = []
            # Then select shiny locations by adding them one-by-one, with the constraint that they can't be too close to each other
            while len(self.shiny['locations']) < self.shiny['n']:
                new = np.random.randint(self.n_locations)
                too_close = [dist_matrix[new,existing] < np.max(dist_matrix) / self.shiny['n'] for existing in self.shiny['locations']]
                if not any(too_close):
                    self.shiny['locations'].append(new)                    
            # Set those locations to be shiny
            for shiny_location in self.shiny['locations']:
                self.locations[shiny_location]['shiny'] = True
            # Get objects at shiny locations
            self.shiny['objects'] = [self.locations[location]['observation'] for location in self.shiny['locations']]
            # Make list of objects that are not shiny
            not_shiny = [observation for observation in range(self.n_observations) if observation not in self.shiny['objects'] ]
            # Update observations so there is no non-shiny occurence of the shiny objects
            for location in self.locations:
                # Update a non-shiny location if it has a shiny object observation
                if location['id'] not in self.shiny['locations'] and location['observation'] in self.shiny['objects']:
                    # Pick new observation from non-shiny objects                    
                    location['observation'] = np.random.choice(not_shiny)
            # Generate a policy towards each of the shiny objects
            self.shiny['policies'] = [self.policy_distance(shiny_location) for shiny_location in self.shiny['locations']]
    
    def observations_randomise(self):
        if self.env_type == 'loop_laps':
            # Go through one lap plus one state and set unique stimulus for each location
            # We go one extra state after the lap because the 0th state is truly unique - it contains the reward stimulus
            randomized_observations = np.random.choice(np.arange(1, self.n_observations), self.n_observations-1, replace=self.replace)
            for i, location in enumerate(self.locations[1:self.lap_len+1]):
                location['observation'] = randomized_observations[i]
                
            # Every n states should be identical. So starting after lap_len + 1 we set observations to be same as those one lap ago
            for state_id, location in enumerate(self.locations[self.lap_len+1:]):
                # enumerate will start at 0, but this state_id starts at lap_len + 1
                state_id = state_id + self.lap_len + 1
                # Set observation to be identical to that of previous lap
                location['observation'] = self.locations[state_id - self.lap_len]['observation']
        
        elif self.env_type == 'multi_w_exploration':
            randomized_observations = np.random.choice(np.arange(1, self.n_observations), self.n_observations-1, replace=self.replace)
            observation_i = 0
            for location in self.locations:
                # Only randomize non-rewards
                if location['observation'] != 0:
                    location['observation'] = randomized_observations[observation_i]
                    observation_i = observation_i + 1
        else:
            # Run through every abstract location
            for location in self.locations:
                # Pick random observation from any of the observations. May be unique or not depending on replace
                location['observation'] = np.random.choice(self.n_observations, replace=self.replace)
        return self
    
    def policy_random(self):
        # Run through every abstract location
        for location in self.locations:
            # Count the number of actions that can transition anywhere for this location
            count = sum([sum(action['transition']) > 0 for action in location['actions']])
            # Run through all actions at this location to update their probability
            for action in location['actions']:
                # If this action transitions anywhere: it is an available action, so set its probability to 1/count
                action['probability'] = 1.0/count if sum(action['transition']) > 0 else 0
        return self
    
    def policy_learned(self, reward_locations):
        # This generates a Q-learned policy towards reward locations.
        # Prepare new set of locations to hold policies towards reward locations
        new_locations, reward_locations = self.get_reward(reward_locations)
        # Initialise state-action values Q at 0
        for location in new_locations:
            for action in location['actions']:
                action['Q'] = 0
        # Do value iteration in order to find a policy toward a given location
        iters = 10*self.n_locations
        # Run value iterations by looping through all actions iteratively
        for i in range(iters):
            # Deepcopy the current Q-values so they are the same for all updates (don't update values that you later need)
            prev_locations = copy.deepcopy(new_locations)
            for location in new_locations:
                for action in location['actions']:
                    # Q-value update from value iteration of Bellman equation: Q(s,a) <- sum_across_s'(p(s,a,s') * (r(s') + gamma * max_across_a'(Q(s', a'))))
                    action['Q'] = sum([probability * ((new_location in reward_locations) + self.shiny['gamma'] * max([new_action['Q'] for new_action in prev_locations[new_location]['actions']])) for new_location, probability in enumerate(action['transition'])])
        # Calculate policy from softmax over Q-values for every state
        for location in new_locations:
            exp = np.exp(self.shiny['beta'] * np.array([action['Q'] if action['probability']>0 else -np.inf for action in location['actions']]))
            for action, probability in zip(location['actions'], exp/sum(exp)):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action['probability'] = probability
        # Return new locations with updated policy for given reward locations
        return new_locations

    def policy_distance(self, reward_locations):
        # This generates a distance-based policy towards reward locations, which is much faster than Q-learning but ignores policy and transition probabilities
        # Prepare new set of locations to hold policies towards reward locations
        new_locations, reward_locations = self.get_reward(reward_locations)
        # Create boolean vector of reward locations for matrix indexing
        is_reward_location = np.zeros(self.n_locations, dtype=bool)
        is_reward_location[reward_locations] = True
        # Calculate distances between all locations based on adjacency matrix - this doesn't take transition probabilities into account!
        dist_matrix = shortest_path(csgraph=np.array(self.adjacency), directed=True)                      
        # Fill out minimum distance to any reward state for each action
        for location in new_locations:
            for action in location['actions']:
                action['d'] = np.min(dist_matrix[is_reward_location, np.array(action['transition']) > 0]) if any(action['transition']) else np.inf
        # Calculate policy from softmax over negative distances for every action
        for location in new_locations:
            exp = np.exp(self.shiny['beta'] * np.array([-action['d'] if action['probability']>0 else -np.inf for action in location['actions']]))
            for action, probability in zip(location['actions'], exp/sum(exp)):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action['probability'] = probability
        # Return new locations with updated policy for given reward locations
        return new_locations
        
    def generate_walks(self, walk_length=10, n_walk=100, repeat_bias_factor=2):
        # Generate walk by sampling actions according to policy, then next state according to graph
        walks = [] # This is going to contain a list of (state, observation, action) tuples
        for currWalk in range(n_walk):
            new_walk = []
            # If shiny hasn't been specified: there are no shiny objects, generate default policy
            if self.shiny is None:
                if self.specify_behavior:
                    new_walk = self.walk_input(new_walk, walk_length, self.behavior_type, self.seed)
                else:
                    new_walk = self.walk_default(new_walk, walk_length, repeat_bias_factor)
            # If shiny was specified: use policy that uses shiny policy to approach shiny objects sequentially
            else:

                    new_walk = self.walk_shiny(new_walk, walk_length, repeat_bias_factor)
            # Clean up walk a bit by only keep essential location dictionary entries
            # Keep the observations so we can check if they are reward (observation=0)
            # This lets us scale loss for reward 
            for step in new_walk[:-1]:
                step[0] = {'id': step[0]['id'],'observation':step[0]['observation'],'shiny':step[0]['shiny']}
            # Append new walk to list of walks
            walks.append(new_walk)
        return walks

    def walk_default(self, walk, walk_length, repeat_bias_factor=2):
        # Finish the provided walk until it contains walk_length steps
        for curr_step in range(walk_length - len(walk)):
            # Get new location based on previous action and location
            new_location = self.get_location(walk)
            # Get new observation at new location
            new_observation = self.get_observation(new_location)
            # Get new action based on policy at new location
            new_action = self.get_action(new_location, walk)
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action])
        # Return the final walk
        return walk
    
    def walk_input(self, walk, walk_length, behavior_type='sweeping', seed=0):
        
        if behavior_type == 'sweeping':
            return self.walk_sweeping_arms(walk, walk_length, seed)
        elif behavior_type == 'random_arms':
            return self.walk_random_arms(walk, walk_length, seed)

        raise Exception('walk_input requires a specified policy type. None was received.')
    
    def walk_sweeping_arms(self, walk, walk_length, seed):
        all_arms = np.arange(6)
        gen = np.random.default_rng(seed)
        curr_arm = gen.choice(all_arms)
        curr_dir = gen.choice([-1, 1]) if curr_arm % 5 else int(((curr_arm<2.5)-0.5)*2)
        
        # Arm number converts to state by factor of 5
        curr_arm_end_state = curr_arm * 5
        walk.append(self.locations[curr_arm_end_state])
        
        # Calculate graph properties for shortest path calculation between arms
        adjacency = np.array(self.adjacency) 
        dists, pred = shortest_path(adjacency, directed=False, method='FW', return_predecessors=True)
        
        while len(walk) < walk_length+1:
            
            # Choose next arm, but don't include current arm in possibilities to avoid repeat visits
            new_arm = curr_arm + curr_dir
            new_arm_end_state = new_arm * 5
            new_dir = curr_dir if new_arm % 5 else int(((new_arm<2.5)-0.5)*2)
        
            # Using the adjacency matrix, create a trajectory using a shortest path traversal from end of curr_arm to end of new_arm
            #trajectory = self.get_path(pred, curr_arm_end_state, new_arm_end_state)[1:]
            #print(trajectory)
            trajectory = [self.locations[state_id] for state_id in self.get_path(pred, curr_arm_end_state, new_arm_end_state)][1:]
            
            # Update curr_arm now that new_arm has been used
            curr_arm = new_arm
            curr_arm_end_state = curr_arm * 5
            
            curr_dir = new_dir
            
            walk.extend(trajectory)
            
        # The final trajectory could have made the walk too long. Truncate walk to be the correct size. Make an extra step so action assignment works
        walk = walk[:walk_length+1]
        # identify observations at each location in completed walk
        walk_observations = [self.get_observation(loc) for loc in walk]
        # From every state i in the walk, identify the action that caused i -> i+1
        walk_actions = [self.get_action_between(walk[i], walk[i+1]) for i in np.arange(len(walk)-1)]
        walk = walk[:walk_length]
        walk_observations = walk_observations[:walk_length]
        walk = [[walk[i], walk_observations[i], walk_actions[i]] for i in np.arange(len(walk))]
        
        return walk
    
    # Take a start state and an end state. Identify which action caused the transition from start to end
    def get_action_between(self, start_state, end_state):
        actions = start_state["actions"]
        end_state_arr = np.eye(self.n_locations)[:, end_state["id"]]
        
        # Retrieve the action id that would have led to this transition. indexing into actions and then "id" is redundant, but kept in case action ids
        # ever don't align with indexes in actions list
        try:
            action_id = actions[np.argwhere([np.array_equal(end_state_arr, action["transition"]) for action in actions])[0, 0]]["id"]
            return action_id
        except IndexError:
            raise IndexError("Impossible transition")

    
    def walk_random_arms(self, walk, walk_length, seed):

        # Arms are identified using the state at their ends, which happen to be arm_num*5
        all_arms = np.arange(6)
        gen = np.random.default_rng(seed)
        curr_arm = gen.choice(all_arms)

        # Arm number converts to state by factor of 5
        curr_arm_end_state = curr_arm * 5
        walk.append(self.locations[curr_arm_end_state])
        
        # Calculate graph properties for shortest path calculation between arms
        adjacency = self.adjacency
        dists, pred = shortest_path(adjacency, directed=False, method='FW', return_predecessors=True)

        while len(walk) < walk_length+1:

            # Choose next arm, but don't include current arm in possibilities to avoid repeat visits
            available_arms = np.delete(all_arms, curr_arm)
            new_arm = gen.choice(available_arms)
            new_arm_end_state = new_arm * 5
            
            # Using the adjacency matrix, create a trajectory using a shortest path traversal from end of curr_arm to end of new_arm
            trajectory = [self.locations[state_id] for state_id in self.get_path(pred, curr_arm_end_state, new_arm_end_state)][1:]

            # Update curr_arm now that new_arm has been used
            curr_arm = new_arm
            curr_arm_end_state = curr_arm * 5

            walk.extend(trajectory)
            
        # The final trajectory could have made the walk too long. Truncate walk to be the correct size. Make an extra step so action assignment works
        walk = walk[:walk_length+1]
        # identify observations at each location in completed walk
        walk_observations = [self.get_observation(loc) for loc in walk]
        # From every state i in the walk, identify the action that caused i -> i+1
        walk_actions = [self.get_action_between(walk[i], walk[i+1]) for i in np.arange(len(walk)-1)]
        walk = walk[:walk_length]
        walk_observations = walk_observations[:walk_length]
        walk = [[walk[i], walk_observations[i], walk_actions[i]] for i in np.arange(len(walk))]
        
        return walk
    
    def get_path(self, pred, start, goal):
        path = [goal]
        k = goal
        while pred[start, k] != -9999:
            path.append(pred[start, k])
            k = pred[start, k]
        return path[::-1]
    
    def walk_shiny(self, walk, walk_length, repeat_bias_factor=2):
        # Pick current shiny object to approach
        shiny_current = np.random.randint(self.shiny['n'])
        # Reset number of iterations to hang around an object once found
        shiny_returns = self.shiny['returns']
        # Finish the provided walk until it contains walk_length steps
        for curr_step in range(walk_length - len(walk)):
            # Get new location based on previous action and location
            new_location = self.get_location(walk)
            # Check if the shiny object was found in this step
            if new_location['id'] == self.shiny['locations'][shiny_current]:
                # After shiny object is found, start counting down for hanging around
                shiny_returns -= 1            
            # Check if it's time to select new object to approach
            if shiny_returns < 0:
                # Pick new current shiny object to approach
                shiny_current = np.random.randint(self.shiny['n'])
                # Reset number of iterations to hang around an object once found
                shiny_returns = self.shiny['returns']                            
            # Get new observation at new location
            new_observation = self.get_observation(new_location)
            # Get new action based on policy of new location towards shiny object
            new_action = self.get_action(self.shiny['policies'][shiny_current][new_location['id']], walk)
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action])
        # Return the final walk
        return walk
    
    def get_location(self, walk):
        # First step: start at random location. However, if this is a loop_lap environment, always start at 2nd state (to avoid reward state)
        if len(walk) == 0:
            new_location = 1 if self.env_type == 'loop_laps' else np.random.randint(self.n_locations)
        # Any other step: get new location from previous location and action
        else:                        
            new_location = int(np.flatnonzero(np.cumsum(walk[-1][0]['actions'][walk[-1][2]]['transition'])>np.random.rand())[0])
        # Return the location dictionary of the new location
        return self.locations[new_location]
    
    def get_observation(self, new_location):
        # Find sensory observation for new state, and store it as one-hot vector
        new_observation = np.eye(self.n_observations)[new_location['observation']]
        # Create a new observation by converting the new observation to a torch tensor
        new_observation = torch.tensor(new_observation, dtype=torch.float).view((new_observation.shape[0]))
        # Return the new observation
        return new_observation
        
    def get_action(self, new_location, walk, repeat_bias_factor=2):
        # Build policy from action probability of each action of provided location dictionary
        policy = np.array([action['probability'] for action in new_location['actions']])
        # Add a bias for repeating previous action to walk in straight lines, only if (this is not the first step) and (the previous action was a move)
        policy[[] if len(walk) == 0 or new_location['id'] == walk[-1][0]['id'] else walk[-1][2]] *= repeat_bias_factor
        # And renormalise policy (note that for unavailable actions, the policy was 0 and remains 0, so in that case no renormalisation needed)
        policy = policy / sum(policy) if sum(policy) > 0 else policy
        # Select action in new state
        new_action = int(np.flatnonzero(np.cumsum(policy)>np.random.rand())[0])
        # Return the new action
        return new_action
    
    def get_reward(self, reward_locations):
        # Stick reward location into a list if there is only one reward location. Use multiple reward locations simultaneously for e.g. wall attraction
        reward_locations = [reward_locations] if type(reward_locations) is not list else reward_locations
        # Copy locations for updated policy towards goal
        new_locations = copy.deepcopy(self.locations)
        # Disable self-actions at reward locations because they will be very attractive
        for reward_location in reward_locations:
            # Check for each action if it's a self-action
            for action in new_locations[reward_location]['actions']:
                if action['transition'][reward_location] == 1:
                    action['probability'] = 0
            # Count total action probability to renormalise after disabling self-action
            total_probability = sum([action['probability'] for action in new_locations[reward_location]['actions']])
            # Renormalise action probabilities
            for action in new_locations[reward_location]['actions']:
                action['probability'] = action['probability'] / total_probability if total_probability > 0 else action['probability']
        return new_locations, reward_locations
        
        
        
        
        
        