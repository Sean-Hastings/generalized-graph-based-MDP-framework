from math import isclose
from copy import deepcopy
import itertools
import numpy as np
import time

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
        self.adj_memos = {}

    def convert_state(self, goal):
        return goal

    def get_adjacent(self, i_state):
        if not i_state.id in self.adj_memos:
            self.adj_memos[i_state.id] = [s.id for s in self.states if any([any([d == i_state for d in a.destinations]) for a in s.actions])]
        return self.adj_memos[i_state.id]

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
        saved_edges = dict()
        clusters = {frozenset({state}) for state in self.states}
        while init_cluster_ct > k:
            print("cluster count", len(clusters))
            best_val = -float("inf")
            best_copy = None

            best_pair = {None, None}

            pairs = itertools.combinations(clusters, 2)
            for cluster_1, cluster_2 in pairs:
                if cluster_1 != cluster_2:
                    num_edges = self.get_num_edges(cluster_1, cluster_2)
                    saved_edges[(cluster_1, cluster_2)] = num_edges
                    saved_edges[(cluster_2, cluster_1)] = num_edges


            pairs = itertools.combinations(clusters, 2)

            print("second loop")
            for combine1, combine2 in pairs:
                #t = time.time()
                u = self.dutil(clusters, combine1, combine2, saved_edges)
                if u > best_val:
                    best_val = u
                    best_pair = {combine1, combine2}
                    # clusters_copy = clusters.copy()
                    # clusters_copy.remove(combine1)
                    # clusters_copy.remove(combine2)
                    # clusters_copy.add(combine1.union(combine2))
                    # utility = self.get_utility(clusters_copy, saved_vals)
                    # if utility > best_val:
                    #     best_val = utility
                    #     best_copy = clusters_copy
                #print(time.time() - t)


            c1, c2 = best_pair
            clusters.remove(c1)
            clusters.remove(c2)
            new_cluster = c1.union(c2)
            clusters.add(new_cluster)

            for c in clusters:
                if c != new_cluster:
                    num_edges = saved_edges[(c1, c)] + saved_edges[(c2, c)]
                    saved_edges[(new_cluster, c)] = num_edges
                    saved_edges[(c, new_cluster)] = num_edges


            # clusters = best_copy
            init_cluster_ct = len(clusters)

        return clusters


    def get_val(self, cluster_1, cluster_2, saved_edges):
        # if (cluster_1, cluster_2) not in saved_vals:
        #                 num_edges = self.get_num_edges(cluster_1, cluster_2)
        #                 val = self.value_function(cluster_1.__len__(), cluster_2.__len__(), num_edges)
        #                 saved_vals[(cluster_1, cluster_2)] = val
        #                 saved_vals[(cluster_2, cluster_1)] = val
        # return  saved_vals[(cluster_1, cluster_2)]
         num_edges = self.get_num_edges(cluster_1, cluster_2)
         return self.value_function(cluster_1.__len__(), cluster_2.__len__(), num_edges)


    def dutil(self, clusters, cluster1, cluster2, saved_edges):
        u0 = 0
        for c in clusters:
            if c != cluster1:
                u0 += self.value_function(len(cluster1), len(c), saved_edges[(cluster1, c)])
        for c in clusters:
            if c != cluster2 and c != cluster1:
                u0 +=  self.value_function(len(cluster2), len(c), saved_edges[(cluster2, c)])

        new_cluster = cluster1.union(cluster2)

        u=0
        for c in clusters:
            if c != cluster1 and c != cluster2:
                u  +=  self.value_function(len(new_cluster), len(c),
                                           saved_edges[(cluster2, c)] + saved_edges[(cluster1, c)])

        return u - u0



    def get_utility(self, clusters, saved_vals):
        utility = 0
        pairs = itertools.combinations(clusters, 2)
        for cluster_1, cluster_2 in pairs:
                if cluster_1 != cluster_2:
                    if (cluster_1, cluster_2) not in saved_vals:
                        num_edges = self.get_num_edges(cluster_1, cluster_2)
                        val = self.value_function(cluster_1.__len__(), cluster_2.__len__(), num_edges)
                        saved_vals[(cluster_1, cluster_2)] = val
                        saved_vals[(cluster_2, cluster_1)] = val
                        utility += val
                    else:
                        utility += saved_vals[(cluster_1, cluster_2)]
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
