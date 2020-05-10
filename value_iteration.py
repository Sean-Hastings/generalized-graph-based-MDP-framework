import cProfile
from multiprocessing.pool import Pool
from copy import deepcopy
import numpy as np
from graph import *


''' https://artint.info/html/ArtInt_227.html '''
def value_iterate(graph, goal_id, gamma=.99, epsilon=.01, debug=False):
    epsilon = epsilon * gamma
    sz = graph.size
    if debug:
        pr = cProfile.Profile()
        pr.enable()
    value = [1] * len(graph.states)
    next_value = [0] * len(graph.states)
    policy = [-1] * len(graph.states)
    to_update = [abs(next_value[i] - value[i]) > epsilon for i in range(len(value))]
    iter = 0
    while any(to_update):
        value = deepcopy(next_value)
        for i_state, state in enumerate(graph.states):
            adj = graph.get_adjacent(i_state)
            if (any([to_update[ad] for ad in adj])) and (not (all([state.actions[0] == act for act in state.actions]))):
                action_value = [0] * len(state.actions)
                action_max = 0
                for i_action, action in enumerate(state.actions):
                    if all([dest == i_state for dest in action.destinations]):
                        continue
                    action_value[i_action] = sum([action.probabilities[i_sp]*((action.destinations[i_sp] == goal_id)+(1-(action.destinations[i_sp] == goal_id))*value[action.destinations[i_sp]]) for i_sp in range(len(action.destinations))])
                    if action_value[i_action] > action_max:
                        action_max = action_value[i_action]
                        policy[i_state] = i_action
                next_value[i_state] = action_max*gamma
        to_update = [abs(next_value[i] - value[i]) > epsilon for i in range(len(value))]

        if debug:
            iter += 1
            print('=========')
            print(iter)
            print(sum(to_update))

    if debug:
        pr.disable()
        pr.print_stats(sort='cumtime')
    return next_value, policy


def evaluate(args):
    value, goal_id, gamma, epsilon, policy, to_update, state = args
    i_state = state.id
    if not (all([state.actions[0] == act for act in state.actions])):
        action_value = [0] * len(state.actions)
        action_max = 0
        for i_action, action in enumerate(state.actions):
            if all([dest == i_state for dest in action.destinations]):
                continue
            action_value[i_action] = sum([action.probabilities[i_sp]*((action.destinations[i_sp] == goal_id)+(1-(action.destinations[i_sp] == goal_id))*value[action.destinations[i_sp]]) for i_sp in range(len(action.destinations))])
            if action_value[i_action] > action_max:
                action_max = action_value[i_action]
                policy = i_action
        action_max = action_max * gamma
        to_update = abs(action_max - value[i_state]) > epsilon
        return i_state, action_max, policy, to_update
    else:
        return i_state, value[i_state], policy, 0




''' https://artint.info/html/ArtInt_227.html '''
def value_iterate_threaded(graph, goal_id, gamma=.99, epsilon=.01, debug=False):
    epsilon = gamma*epsilon
    if debug:
        pr = cProfile.Profile()
        pr.enable()
    value = [1] * len(graph.states)
    next_value = [0] * len(graph.states)
    policy = np.array([-1] * len(graph.states))
    to_update = [abs(next_value[i] - value[i]) > epsilon for i in range(len(value))]
    iter = 0
    with Pool(processes=8) as pool:
        while any(to_update):
            value = deepcopy(next_value)
            for i, v, p, t in pool.imap_unordered(evaluate, [(value, goal_id, gamma, epsilon, policy[i], to_update[i], graph.states[i]) for i in range(len(value)) if any([to_update[ad] for ad in graph.get_adjacent(i)])], chunksize=sum(to_update)):
                next_value[i] = v
                policy[i] = p
                to_update[i] = t
            iter += 1
            print('===========')
            print(iter)
            print(sum(to_update))

    if debug:
        pr.disable()
        pr.print_stats(sort='cumtime')
    return next_value, policy
