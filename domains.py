from graph import *
from value_iteration import *
import numpy as np
from matplotlib import pyplot as plt



class GridWorld(DSAG):
    def __init__(self, states, build_args=None):
        super().__init__(states, build_args)
        self.size = np.sqrt(len(states))
        assert int(self.size) == self.size
        self.size = int(self.size)

    def convert_state(self, goal):
        if isinstance(goal, (list, tuple)):
            goal = self.size*goal[0] + goal[1]
        return goal

    def get_num_edges(self,cluster_1, cluster_2):
        num_edges = 0
        map_1 = set([a.id for a in cluster_1])
        map_2 = set([a.id for a in cluster_2])
        for a in cluster_1:
            num_edges += map_2.intersection(set(self.get_adjacent(a.id))).__len__()
        for a in cluster_2:
            num_edges += map_1.intersection(set(self.get_adjacent(a.id))).__len__()
        return num_edges

    def get_adjacent(self, i_state):
        adj = [i_state]

        r = i_state // self.size
        c = i_state % self.size

        adj += [max(r-1, 0)*self.size + c]
        adj += [min(r+1, self.size-1)*self.size + c]
        adj += [max(c-1, 0) + r*self.size]
        adj += [min(c+1, self.size-1) + r*self.size]

        return adj

    def plan(self, goal=0, gamma=.99, debug=False):
        if isinstance(goal, (tuple, list)):
            if isinstance(goal[0], (tuple, list)):
                goal = [self.convert_state(g) for g in goal]
            else:
                goal = [self.convert_state(goal)]
        else:
            goal = [goal]
        return value_iterate(self, goal, gamma=gamma, debug=debug)

    def show(self, goal=(0,0), gamma=.99, debug=False):
        value, policy = self.plan(goal, gamma, debug)

        print('Values:')
        plt.imshow(np.array(value).reshape(self.size, self.size))
        plt.show()

        print('Policy:')
        plt.imshow(np.array(policy).reshape(self.size, self.size))
        plt.show()

        '''
        print('key: wall, up, down, left, right')
        plt.imshow(np.array([-1,0,1,2,3]).reshape(1,5))
        '''

        if debug:
            print(np.array(policy).reshape(self.size, self.size))
            print(np.array(value).reshape(self.size, self.size))

    @staticmethod
    def build(*build_args):
        return build_gridworld(*build_args)


class Taxi(DSAG):
    def __init__(self, states, indices, build_args=None):
        super().__init__(states, build_args)
        self.indices = indices
        self.size = indices.shape[0]

    def convert_state(self, goal=(0,0), debug=False):
        if isinstance(goal, (list, tuple)):
            if len(goal) == 2:
                return self.indices[goal[0], goal[1], goal[0], goal[1], 0]
            else:
                return self.indices[goal[0], goal[1], goal[2], goal[3], goal[4]]
        else:
            return goal

    def get_adjacent(self, i_state):
        ar = i_state // (self.size**3 * 2)
        ac = i_state // (self.size**2 * 2) % self.size
        pr = i_state // (self.size**1 * 2) % self.size
        pc = i_state // 2 % self.size
        h  = i_state % 2

        adj = [self.indices[max(ar-1, 0), ac, pr, pc, h],
            self.indices[min(ar+1, self.size-1), ac, pr, pc, h],
            self.indices[ar, max(ac-1, 0), pr, pc, h],
            self.indices[ar, min(ac+1, self.size-1), pr, pc, h],
            self.indices[ar, ac, max(pr-1, 0), pc, h],
            self.indices[ar, ac, min(pr+1, self.size-1), pc, h],
            self.indices[ar, ac, pr, max(pc-1, 0), h],
            self.indices[ar, ac, pr, min(pc+1, self.size-1), h],
            self.indices[ar, ac, pr, pc, 0],
            self.indices[ar, ac, pr, pc, 1]]
        if h:
            adj += [self.indices[max(ar-1, 0), ac, max(pr-1, 0), pc, h],
                self.indices[min(ar+1, self.size-1), ac, min(pr+1, self.size-1), pc, h],
                self.indices[ar, max(ac-1, 0), pr, max(pc-1, 0), h],
                self.indices[ar, min(ac+1, self.size-1), pr, min(pc+1, self.size-1), h]]

        return adj

    def plan(self, goal=(0,0)):
        if isinstance(goal, (tuple, list)):
            if isinstance(goal[0], (tuple, list)):
                goal = [self.convert_state(g) for g in goal]
            else:
                goal = [self.convert_state(goal)]
        else:
            goal = [goal]
        return value_iterate_threaded(self, goal, gamma=gamma, debug=debug)

    def show(self, passenger=(0,0), goal=(0,0), gamma=.99, debug=False):
        value, policy = self.plan(goal, debug)

        rows = np.array(sum([[i]*self.size for i in range(self.size)], []))
        cols = np.concatenate([np.arange(self.size)]*self.size)

        print('Values:')
        print('Getting Passenger:')
        plt.imshow(np.array(value).reshape(self.size, self.size, self.size, self.size, 2)[:, :, passenger[0], passenger[1], 0])
        plt.show()
        print('Transporting Passenger:')
        plt.imshow(np.array(value).reshape(self.size, self.size, self.size, self.size, 2)[rows, cols, rows, cols, 1].reshape(self.size, self.size))
        plt.show()

        print('Policy:')
        print('Getting Passenger:')
        plt.imshow(np.array(policy).reshape(self.size, self.size, self.size, self.size, 2)[:, :, passenger[0], passenger[1], 0])
        plt.show()
        print('Transporting Passenger:')
        plt.imshow(np.array(policy).reshape(self.size, self.size, self.size, self.size, 2)[rows, cols, rows, cols, 1].reshape(self.size, self.size))
        plt.show()

        '''
        print('key: wall, up, down, left, right, pick, place')
        plt.imshow(np.array([-1,0,1,2,3,4,5]).reshape(1,7))
        plt.show()
        '''

        if debug:
            print(np.array(value).reshape(self.size, self.size, self.size, self.size, 2)[:, :, passenger[0], passenger[1], 0])
            print(np.array(value).reshape(self.size, self.size, self.size, self.size, 2)[rows, cols, rows, cols, 1].reshape(self.size, self.size))
            print(np.array(policy).reshape(self.size, self.size, self.size, self.size, 2)[:, :, passenger[0], passenger[1], 0])
            print(np.array(policy).reshape(self.size, self.size, self.size, self.size, 2)[rows, cols, rows, cols, 1].reshape(self.size, self.size))

    @staticmethod
    def build(*build_args):
        return build_taxi(*build_args)



def build_gridworld(size, slip_rate=.1, walls=[], sinks=[], debug=False):
    sinks = [(s[0]*size+s[1] if isinstance(s, (list, tuple)) else s) for s in sinks]
    states = [None]*size**2
    for i in range(size):
        for j in range(size):
            id_ = size*i+j
            up    = i-1
            down  = i+1
            left  = j-1
            right = j+1
            if (i, j) in walls or id_ in sinks:
                states[id_] = State(id_, [Action([size*i+j]*4, [1/4]*4)]*4)
            else:
                if i == 0 or (i-1, j) in walls:
                    up = i
                if i == (size-1) or (i+1, j) in walls:
                    down = i
                if j == 0 or (i, j-1) in walls:
                    left = j
                if j == (size-1) or (i, j+1) in walls:
                    right = j
                actions = [None]*4
                dests = [up*size+j, down*size+j, i*size+left, i*size+right]
                probs = [1-slip_rate, slip_rate/3]
                actions[0] = Action(dests, [probs[0], probs[1], probs[1], probs[1]])
                actions[1] = Action(dests, [probs[1], probs[0], probs[1], probs[1]])
                actions[2] = Action(dests, [probs[1], probs[1], probs[0], probs[1]])
                actions[3] = Action(dests, [probs[1], probs[1], probs[1], probs[0]])
                states[id_] = State(id_, actions)

                if debug:
                    print(i, j, [(dest//size, dest%size) for dest in actions[0].destinations])

    return GridWorld(states, {'size':size, 'slip_rate':slip_rate, 'walls':walls, 'sinks':sinks})


def build_four_rooms(size, slip_rate=.1, walls=[], sinks=[], debug=False):
    pos = size//2
    n_walls = [(pos, i) for i in range(size)] + [(i, pos) for i in range(size)]
    n_walls = n_walls[:-pos*3//2] + n_walls[-pos*3//2+1:]
    n_walls = n_walls[:-pos//2] + n_walls[-pos//2+1:]
    n_walls = n_walls[:pos*3//2] + n_walls[pos*3//2+1:]
    n_walls = n_walls[:pos//2] + n_walls[pos//2+1:]
    walls += n_walls

    if debug:
        print(walls)

    return build_gridworld(size, slip_rate, walls, sinks, debug)


def build_taxi(size, slip_rate=.1, walls=[], sinks=[], debug=False):
    indices = np.arange(size**4*2).reshape(size, size, size, size, 2)
    wells = [s for s in sinks if isinstance(s, (list, tuple)) and len(s) == 2]
    for well in wells:
        sinks += list(indices[well[0],well[1], :, :, :].flatten())
        sinks += list(indices[:, :, well[0],well[1], :].flatten())
    sinks = [s for s in sinks if not (isinstance(s, (list, tuple)) and len(s) == 2)]
    sinks = [(indices[s[0],s[1],s[2],s[3],s[4]] if isinstance(s, (list, tuple)) else s) for s in sinks]
    states = [None] * indices.size
    for ar in range(size):
        for ac in range(size):
            for pr in range(size):
                for pc in range(size):
                    for h in range(2):
                        id_ = indices[ar, ac, pr, pc, h]
                        invalid = (not (pr==ar and pc==ac)) and h
                        if (ar, ac) in walls or (pr, pc) in walls or id_ in sinks or invalid:
                            states[id_] = State(id_, [Action([id_]*6, [1/6]*6)]*6)
                        else:
                            pick  = 1 if all([pr==ar, pc==ac]) else 0
                            place = 0
                            up    = -1
                            down  = 1
                            left  = -1
                            right = 1
                            if ar == 0 or (ar+up, ac) in walls:
                                up = 0
                            if ar == (size-1) or (ar+down, ac) in walls:
                                down = 0
                            if ac == 0 or (ar, ac+left) in walls:
                                left = 0
                            if ac == (size-1) or (ar, ac+right) in walls:
                                right = 0

                            aup    = ar+up
                            adown  = ar+down
                            aleft  = ac+left
                            aright = ac+right
                            if h:
                                pup    = pr+up
                                pdown  = pr+down
                                pleft  = pc+left
                                pright = pc+right
                            else:
                                pup    = pr
                                pdown  = pr
                                pleft  = pc
                                pright = pc
                            actions = [None]*6
                            dests = [indices[aup,   ac,     pup,   pc,     h],
                                     indices[adown, ac,     pdown, pc,     h],
                                     indices[ar,    aleft,  pr,    pleft,  h],
                                     indices[ar,    aright, pr,    pright, h],
                                     indices[ar,    ac,     pr,    pc,     pick],
                                     indices[ar,    ac,     pr,    pc,     place]]
                            probs = [1-slip_rate, slip_rate/5]
                            actions = [Action(dests, [probs[1]]*i + [probs[0]] + [probs[1]]*(5-i)) for i in range(6)]
                            states[id_] = State(id_, actions)

    return Taxi(states, indices, {'size':size, 'slip_rate':slip_rate, 'walls':walls, 'sinks':sinks})
