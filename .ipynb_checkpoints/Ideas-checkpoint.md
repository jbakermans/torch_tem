### Ideas

1. Does pretraining TEM on gridworld and then linear track speed up learning on exploration?
    - basically teach it all of the spatial experiences the rats get
    - gut feeling is that this might help
    - but the new bioRxiv paper points out that TEM doesn't transfer across spatially dissimilar environments
    - Is there a way to identify sub-structures that are preserved, even if not all structural features are preserved?
    - for example, N->W->S->E is possible in gridworld but not 6-arm W
    - HOWEVER, W->W->E->E is possible in both. 
    - maybe the top loop is not transferable, but the bottom loop is
    - is there a way to transfer this?
    
2. TEM does not achieve great generalization performance when trained on dissimilar environments
    - for example, training on gridworld followed by spatial alternation may not be particularly useful
    - but I think that this animals actually do capitalize off of learning in structurally dissimilar environments. What are they capitalizing on?
    


---------------------------------------------------------------------------------------------

1. Let's formalize the possibility that rats undergo an evolution in their 'hypothesized state space' during spatial alternation
    - observation: rats identify the 3 contingency arms quickly
    - Do they explore as if it's a 3-arm bandit?
    - if so, what does the state space look like for a 3 arm bandit
    - more likely - they identify that home arm always gives reward but think that outer arms are probabilistic
 
---------------------------------------------------------------------------------------------

1. What does the state space for probabilistic tasks look like?
    - Would TEM or CSCG learn this?
    
    
    
---------------------------------------------------------------------------------------------

1. I suspect that if we get path equivalence in TEM, path equivalent cells will be more common near the reward points (in the latter half of the trajectory). Is this true in our data?
2. Are path equivalent cells a result of some sort of regularization method? For instance, in cognitively 'easy' tasks, we get more path equivalent cells because of some sort of regularization to use fewer cells for encoding if possible?
3. Whereas they disappear in more demanding tasks since the task demands outweigh the regularization bias?
4. This is vague, I know

---------------------------------------------------------------------------------------------

1. Animal learns to do task - perhaps with model-free RL (idk, just making this up)
2. Now the behavioral policy reflects the structure of the task - you are doing it correctly so the task must be recoverable from your behavior
3. At this point, MEC starts to try and learn a good representation of this task
4. Once it does, hippocampus splitter cells reflect MEC's understanding of the abstract task structure
5. Thus, splitter cells do not reflect memories of previous arms. Instead they reflect that MEC has learned the structure of the space
6. With this disambiguation in mind, we can recognize that the behavioral implications of splitter cells are NOT improved W-alternation performance
7. Instead, the behavioral prediction is improved STRUCTURAL GENERALIZATION and therefore generalization on NEW W-alternation tasks


---------------------------------------------------------------------------------------------

1. Is there anything to extending these models to probabilistic environments?
2. Once the state space is known in these tasks, the environments are deterministic