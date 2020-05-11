# generalized graph-based MDP framework
Part of a class project for Reintegrating AI with George Konidaris

## Important notes

 - GridWorlds run best on the single-process VI, while Taxis run best on multi-process VI. The default for abstractions in general is single-thread, but if you want to run abstracted Taxis I recommend manually changing it (in build_option_a_s, just swap in the VI function used above in build_option
 - Taxi domain is slow af, but with the multi-processed VI it's almost twice as fast
 - I left some example bits in ReAI.py, I recommend using those as a base to build experiments build experiments from
 - When you run [some mdp].show(goal=[specify some goal]), it will run VI with that goal then use pyplot to draw the value and policy tables. For taxi, these are both 5-dimensional so it just draws the slices that correspond to "The agent has not yet retrieved the passenger and the passenger is at the specified location" and "The agent is carrying the passenger"
