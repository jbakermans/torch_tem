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
3. One idea is to do spatial alternation with probabilistic rewards even under an optimal policy
4. What would this let us study?

---------------------------------------------------------------------------------------------

1. One open question is how an animal identifies which context it's in. IE, when it has learned multiple tasks and therefore multiple MEC abstract task spaces, how does it know which to use?
2. We can try to look at this a little in the return to exploration phase of our experiments
3. At this point, animals have learned at least 2 abstract state spaces
    a. spatial_alternation
    b. multi_w_exploration
4. Furthermore, we notice that scn2a animals continue to perform spatial alternation even in return to alternation
5. How do they know when to switch?
7. When animals move to alternation after return to exploration, what neural phenomena exist when they infer which task space to use? - immediately switch to the spatial_alternation 'mode'
8. So is continued alternation behavior due to
    a. no explorative tendencies and therefore no knowledge about change in task structure?
    b. similar explorative tendencies but an inability to switch out the MEC map being used

---------------------------------------------------------------------------------------------

1. Honestly, what is the point of having a good state space representation if it can only be learned from near-optimal behavior?
2. I think that the benefits of good representation cannot be identified when the animals are simply continuing to do the original spatial alternation
3. Good representation's benefits are likely only exposed in more complicated scenarios
4. What are these scenarios?
5. Something about changing goal objects within the state space?
6. Can probabilistic reward address this?

7. Is it possible that good representations are most useful when goals change? IE you can navigate to a different sensory stimulus in the state space more easily?
8. If this is the case, how do you get a rat to do this? With humans, you can just say "your salient cue is now a banana, navigate to that"
9. With rats, you could instead have two different types of rewards - milk and water.
10. At different times, food restrict xor water restrict them
11. This should induce differences in sensory cue salience, and perhaps expose the importance of good representations
12. Very wishy-washy - need to think through this more carefully