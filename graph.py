from math import isclose


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
        assert isclose(sum(probabilities), 1)
        self.destinations = destinations
        self.probabilities = probabilities

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
    def __init__(self, states):
        assert isinstance(states, list)
        assert all([len(state.actions) == len(states[0].actions) for state in states])
        self.states = states

    def convert_goal(self, goal):
        if isinstance(goal, (list, tuple)):
            goal = self.size*goal[0] + goal[1]
        return goal

    def get_adjacent(self, i_state):
        return [s.id for s in self.states if any([any([d == i_state for d in a.destinations]) for a in s.actions])]

    def show(self, goal=0, debug=False):
        goal = self.convert_goal(goal)
        #value, policy = value_iterate(self, goal, debug=debug)

        for i_state, state in enumerate(self.states):
            print(state)#, value[i_state], policy[i_state])

if __name__ == '__main__':
    a = State(0, [Action([0],[1])])
    b = DSAG([a])
    b.show(debug=True)
