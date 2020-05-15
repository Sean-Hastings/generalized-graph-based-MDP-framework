from value_iteration import *
from graph import *
from domains import *
from matplotlib import pyplot as plt


def build_option(graph, valid_states, goal_states, debug=False):
    new_graph = graph.rebuild([graph.convert_state(s) for s in valid_states])
    goal_states = [graph.convert_state(g) for g in goal_states]
    value, policy = value_iterate_threaded(new_graph, goal_states, gamma=.99, debug=debug)
    return (value, policy)


def build_option_a_s(a_s, goal_states, debug=False):
    goal_states = [a_s.convert_state(g) for g in goal_states]
    value, policy = value_iterate(a_s, goal_states, debug=debug)
    return (value, policy)


def build_abstract_state(graph, ground_states):
    return graph.rebuild(ground_states)


class Abstraction():
    def __init__(self, graph, partitions, debug=False):
        self.ground_graph = graph
        self.partitions = [[graph.convert_state(state) for state in partition] for partition in partitions]
        flat_partitions = [state for partition in self.partitions for state in partition]
        assert all([(0 <= state) and (state < len(graph.states)) for state in flat_partitions])
        assert all([state.id in flat_partitions for state in graph.states])

        # partition is a list of lists of integers representing positions of each abstract state's primitive
        self.abstract_states = [build_abstract_state(graph, partition) for partition in self.partitions]
        self.options = [[build_option_a_s(a_s, goal, debug=debug) for goal in self.partitions] for a_s in self.abstract_states]

        na_states = len(partitions)
        dests = list(range(na_states))
        states = []
        for i in range(na_states):
            actions = []
            for j in range(na_states):
                probabilities = [0]*j+[any([o > 0 for o in self.options[i][j][0]])]+[0]*(na_states-j-1)
                if sum(probabilities) == 0:
                    probabilities[i] = 1
                actions += [Action(dests, probabilities)]
            states += [State(i, actions)]

        self.abstract_graph = DSAG(states)
        self.abstract_policy = None

    def convert_state(self, goal):
        for i, partition in enumerate(self.partitions):
            if goal in partition:
                return i
        raise Exception('goal not in any partition, this implies a bug somewhere as it should not be possible.')

    def plan(self, goal=0, debug=False):
        if isinstance(goal, (tuple, list)):
            if isinstance(goal[0], (tuple, list)):
                goal = [self.ground_graph.convert_state(g) for g in goal]
            else:
                goal = [self.ground_graph.convert_state(goal)]
        else:
            goal = [goal]
        abstract_goal = [self.convert_state(g) for g in goal]
        for i in abstract_goal:
            self.options[i][i] = build_option_a_s(self.abstract_states[i], goal, debug=debug)

        abstract_value, self.abstract_policy = value_iterate(self.abstract_graph, abstract_goal, debug=debug)


        ground_value = [0] * len(self.ground_graph.states)
        ground_policy = [0] * len(ground_value)

        for i_state, i_action in enumerate(self.abstract_policy):
            if i_state in abstract_goal:
                i_action = i_state
            val, pol = self.options[i_state][i_action]
            for id_ in self.partitions[i_state]:
                ground_value[id_] = val[id_]
                ground_policy[id_] = pol[id_]


        return ground_value, ground_policy

    def show(self, goal=0, debug=False):
        self.plan(goal, debug)

        for i_a_s in range(len(self.partitions)):
            #abstract_state = self.abstract_states[i_a_s]
            options = self.options[i_a_s]
            for option in options:
                if isinstance(self.ground_graph, GridWorld):
                    plt.imshow(np.array(option).reshape(self.ground_graph.size, self.ground_graph.size))
                    plt.show()
