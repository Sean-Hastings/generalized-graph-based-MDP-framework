from value_iteration import *
from graph import *
from domains import *
from matplotlib import pyplot as plt


def build_option(graph, valid_states, goal_states, debug=False):
    new_graph = graph.rebuild([graph.convert_state(s) for s in valid_states])
    goal_states = [graph.convert_state(g) for g in goal_states]
    _, policy = value_iterate_threaded(new_graph, goal_states, gamma=.99, debug=debug)
    return policy


def build_option_a_s(a_s, goal_states, debug=False):
    goal_states = [a_s.convert_state(g) for g in goal_states]
    _, policy = value_iterate(a_s, goal_states, debug=debug)
    return policy


def build_abstract_state(graph, ground_states):
    return graph.rebuild(ground_states)


class Abstraction():
    def __init__(self, graph, partitions, debug=False):
        self.ground_graph = graph
        self.partitions = [[graph.convert_state(state) for state in partition] for partition in partitions]
        flat_partitions = [state for partition in self.partitions for state in partition]
        assert all([(0 <= state) and (state < len(graph.states)) for state in flat_partitions])
        assert all([state.id in flat_partitions for state in graph.states])

        na_states = len(partitions)
        dests = list(range(na_states))
        self.abstract_states = [build_abstract_state(graph, partition) for partition in self.partitions]
        self.options = [[build_option_a_s(a_s, goal, debug=debug) for goal in self.partitions] for a_s in self.abstract_states]
        self.abstract_graph = DSAG([State(i, [Action(dests, [any([o >= 0 for o in self.options[i][j]]) for j in range(na_states)])]) for i in range(na_states)])
        self.abstract_policy = None

    def convert_state(self, goal):
        for i, partition in enumerate(self.partitions):
            if goal in partition:
                return i
        raise Exception('goal not in any partition, this implies a bug somewhere a_s it should not be possible.')

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

        _, self.abstract_policy = value_iterate(self.abstract_graph, abstract_goal, debug=debug)

    def show(self, goal=0, debug=False):
        self.plan(goal, debug)

        for i_a_s in range(len(self.partitions)):
            #abstract_state = self.abstract_states[i_a_s]
            options = self.options[i_a_s]
            for option in options:
                if isinstance(self.ground_graph, GridWorld):
                    plt.imshow(np.array(option).reshape(self.ground_graph.size, self.ground_graph.size))
                    plt.show()
