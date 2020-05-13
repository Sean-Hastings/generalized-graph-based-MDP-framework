from math import isclose
from copy import deepcopy
import numpy as np


class State():
    def __init__(self, id_, actions):
        self.id = id_
        self.actions = actions

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'State(%d, %s)' % (self.id, self.actions)


class Action():
    def __init__(self, destinations, probabilities):
        assert len(destinations) == len(probabilities)
        prob_sum = sum(probabilities)
        self.destinations = destinations
        self.probabilities = [prob / prob_sum for prob in probabilities]

    def __eq__(self, other):
        return self.destinations == other.destinations and self.probabilities == other.probabilities

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Action(%s)' % str([(self.destinations[i], ('%.2f' % self.probabilities[i])) for i in range(len(self.destinations))])

    def execute(self):
        next_state = np.random.choice(self.destinations, p=self.probabilities)
        return next_state


''' Directed State Action Graph (MDP without rewards) '''
class DSAG():
    def __init__(self, states, build_args=None):
        assert isinstance(states, list)
        assert all([len(state.actions) == len(states[0].actions) for state in states])
        self.states = states
        self.build_args = build_args

    def convert_state(self, goal):
        return goal

    def get_adjacent(self, i_state):
        return [s.id for s in self.states if any([any([d == i_state for d in a.destinations]) for a in s.actions])]

    def show(self, goal=0, debug=False):
        goal = self.convert_state(goal)

        for i_state, state in enumerate(self.states):
            print(state)

    def rebuild(self, newstates):
        newstates = [self.convert_state(s) for s in newstates]
        build_args = deepcopy(self.build_args)
        build_args['sinks'] += [state.id for state in self.states if state.id not in newstates]
        return self.build(*build_args.values())

    @staticmethod
    def build(*build_args):
        return DSAG([], build_args)

    def get_num_edges(self,cluster_1, cluster_2):
        num_edges = 0
        for a in cluster_1:
            num_edges += cluster_2.intersection(set(self.get_adjacent(a))).__len__()
        for a in cluster_2:
            num_edges += cluster_1.intersection(set(self.get_adjacent(a))).__len__()
        return num_edges

    def get_partitions(self,k):
        init_cluster_ct = len(self.states)
        saved_vals = dict()
        clusters = [{state} for state in self.states]
        while init_cluster_ct > k:
            print("cluster count",len(clusters))
            best_val = -float("inf")
            best_copy = None
            for cluster_1 in clusters:
                for cluster_2 in clusters:
                    if cluster_1 != cluster_2:
                        if str((cluster_2, cluster_1)) in saved_vals:
                            saved_vals[str((cluster_1, cluster_2))] = saved_vals[str((cluster_2, cluster_1))]
                        else:
                            num_edges = self.get_num_edges(cluster_1, cluster_2)
                            val = self.value_function(cluster_1.__len__(), cluster_2.__len__(), num_edges)
                            saved_vals[str((cluster_1, cluster_2))] = val

            for combine1 in clusters:
                for combine2 in clusters:
                    if combine1 != combine2:
                        clusters_copy = clusters.copy()
                        clusters_copy.remove(combine1)
                        clusters_copy.remove(combine2)
                        clusters_copy.append(combine1.union(combine2))
                        utility = self.get_utility(clusters_copy,saved_vals)
                        if utility > best_val:
                            best_val = utility
                            best_copy = clusters_copy

            clusters = best_copy
            init_cluster_ct = len(clusters)

        return clusters

    def get_utility(self,clusters,saved_vals):
        utility = 0
        for cluster_1 in clusters:
            for cluster_2 in clusters:
                if cluster_1 != cluster_2:
                    if str((cluster_2, cluster_1)) in saved_vals:
                        saved_vals[str((cluster_1, cluster_2))] = saved_vals[str((cluster_2, cluster_1))]
                    elif str((cluster_1, cluster_2)) not in saved_vals:
                        num_edges = self.get_num_edges(cluster_1, cluster_2)
                        val = self.value_function(cluster_1.__len__(), cluster_2.__len__(), num_edges)
                        saved_vals[str((cluster_1, cluster_2))] = val
                    utility += saved_vals[str((cluster_1, cluster_2))]
        return utility


    def value_function(self,size1,size2,num_edges):
        # if num_edges > 0:
        #     print("value function")
        #     print(size1,size2,num_edges)
        if num_edges == 0:
            return 0
        return min(size1,size2)*np.log(max(size1,size2))/num_edges






if __name__ == '__main__':
    a = State(0, [Action([0],[1])])
    b = DSAG([a])
    b.show(debug=True)
