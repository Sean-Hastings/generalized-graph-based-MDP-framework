from graph import *
from value_iteration import *
from domains import *
from abstraction import *
from time import perf_counter
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np



if __name__ == '__main__':
    #a = build_gridworld(25, slip_rate=.1, walls=[(i,i) for i in range(15)])
    #a.show(21*19, debug=False)


    #b = build_four_rooms(25, walls=[(i,i) for i in range(20)])
    #b.show((10, 5), debug=False)
    #c = b.rebuild(list(range(25*21)))
    #c.show((10, 5), debug=False)


    #d = build_taxi(10, slip_rate=.25, walls=[(2, i) for i in range(7)]+[(i+3, 6) for i in range(4)], sinks=[(6,5), (6,4), (6,3)], debug=False)
    #d.show(passenger=(4,4), goal=[(0,0), (3,0,4,4,0)], debug=True)

    #e = d.rebuild([(ar, ac, pr, pc, h) for ar in range(5) for ac in range(10) for pr in range(5) for pc in range(10) for h in range(2)])
    #e.show(passenger=(4,4), goal=[(0,0), (5,9,4,4,0)], debug=True)

    #f = Abstraction(d, [[(ar, ac, pr, pc, 0) for ar in range(10) for ac in range(10) for pr in range(10) for pc in range(10)], [(ar, ac, pr, pc, 1) for ar in range(10) for ac in range(10) for pr in range(10) for pc in range(10)]], debug=True)
    #print('abstracted')
    #f.plan(goal=(9,9,9,9,1))
    #print('abstract planned')

    num_states = 10
    a = build_four_rooms(num_states, slip_rate=.1)
    #a.show(21*19, debug=False) # Comment this to stop it showing the pyplot stuff

    # partitions = [[(i, j) for i in range(12) for j in range(12)],
    #               [(j, i) for i in range(12) for j in range(12, 25)],
    #               [(i, j) for j in range(12, 25) for i in range(12)],
    #               [(i, j) for j in range(12, 25) for i in range(12, 25)]]
    # num_states = 20
    # num_abstract = 20
    num_abstract = 4
    partitions = a.get_partitions(num_abstract)

    #b = Abstraction(a, partitions)
    #b.show(goal=(10,5))
    partitions = [[x for x in p] for p in partitions]

    goals = [(1,0), (0,1)]
    # goals += [(i, i-5) for i in range(5, 25)]
    # goals += [(i-5, i) for i in range(5, 25)]
    # goals += [(24-i, 20+i) for i in range(5)]

    cluster_arr = np.zeros((num_states, num_states))

    i = 1
    for part in partitions:
        for square in part:
            cluster_arr[square // num_states, square % num_states] = i
        i += 1

    plt.imshow(cluster_arr)
    plt.show()

    # This line of code adds back wall states to one of the partitions because the rest of the code
    # expects wall states to be included in the partitions. 
    for wall in a.get_walls():
        partitions[0].append(wall)


    start_time = perf_counter()
    for goal in goals:
        value, policy = a.plan(goal)
    end_time = perf_counter()
    delta_time = end_time - start_time
    print('The un-abstracted version took %.2f seconds to plan over the given goals' % delta_time)
    # plt.imshow(np.array(value).reshape(25, 25))
    # plt.show()
    # plt.imshow(np.array(policy).reshape(25, 25))
    # plt.show()

    plt.imshow(np.array(value).reshape(num_states, num_states))
    plt.show()
    plt.imshow(np.array(policy).reshape(num_states, num_states))
    plt.show()

    start_time = perf_counter()
    b = Abstraction(a, partitions)
    print('Finished building abstraction')
    for goal in goals:
        value, policy = b.plan(goal)
    end_time = perf_counter()
    delta_time = end_time - start_time
    print('The abstracted version took %.2f seconds to plan over the given goals' % delta_time)
    plt.imshow(np.array(value).reshape(num_states, num_states))
    plt.show()
    plt.imshow(np.array(policy).reshape(num_states, num_states))
    plt.show()
    # plt.imshow(np.array(value).reshape(25, 25))
    # plt.show()
    # plt.imshow(np.array(policy).reshape(25, 25))
    # plt.show()