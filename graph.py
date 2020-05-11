from math import isclose
from copy import deepcopy


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

if __name__ == '__main__':
    a = State(0, [Action([0],[1])])
    b = DSAG([a])
    b.show(debug=True)
