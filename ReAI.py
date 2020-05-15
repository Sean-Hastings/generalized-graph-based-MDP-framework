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

    num_states = 15
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

    goals = [(r, c) for r in range(num_states) for c in range(num_states)]

    cluster_arr = np.zeros((num_states, num_states), dtype=np.float64)

    i = 1.0
    for part in partitions:
        for square in part:
            cluster_arr[square // num_states, square % num_states] = i+0.0
        i += 1.0

    #plt.imshow(cluster_arr)
    #plt.show()

    # This line of code adds back wall states to one of the partitions because the rest of the code
    # expects wall states to be included in the partitions.
    for wall in a.get_walls():
        partitions[0].append(wall)


    start_time = perf_counter()
    un_vps = []
    for goal in goals:
        un_vps += [a.plan(goal)]
    end_time = perf_counter()
    delta_time = end_time - start_time
    print('The un-abstracted version took %.3f seconds per goal to plan over the given goals' % (delta_time / num_states**2))
    # plt.imshow(np.array(value).reshape(25, 25))
    # plt.show()
    # plt.imshow(np.array(policy).reshape(25, 25))
    # plt.show()

    #plt.imshow(np.array(un_vps[0][0]).reshape(num_states, num_states))
    #plt.show()
    #plt.imshow(np.array(un_vps[0][1]).reshape(num_states, num_states))
    #plt.show()

    start_time = perf_counter()
    b = Abstraction(a, partitions)
    end_time = perf_counter()
    delta_time = end_time - start_time
    print('The abstraction took %.3f seconds to build (not including clustering)' % delta_time)
    start_time = perf_counter()
    print('Finished building abstraction')
    a_vps = []
    for goal in goals:
        a_vps += [b.plan(goal)]
    end_time = perf_counter()
    delta_time = end_time - start_time
    print('The abstracted version took %.3f seconds per goal to plan over the given goals' % (delta_time / num_states**2))

    for i, a_vp in enumerate(a_vps):
        value, policy = a_vp
        for j in range(num_states**2):
            for r in range(num_states):
                for c in range(num_states):
                    if (r,c) != goals[i]:
                        id_ = r*num_states+c
                        state = a.states[id_]
                        value[id_] = sum([value[state.actions[policy[id_]].destinations[l]] * state.actions[policy[id_]].probabilities[l] for l in range(4)]) * .99

    #plt.imshow(np.array(a_vps[0][0]).reshape(num_states, num_states))
    #plt.show()
    #plt.imshow(np.array(a_vps[0][1]).reshape(num_states, num_states))
    #plt.show()

    vl = []
    vl1 = []
    vl2 = []
    for i, ua_vp in enumerate(zip(un_vps, a_vps)):
        u_vp, a_vp = ua_vp
        if np.mean(np.array(u_vp[0]).reshape(num_states, num_states)) > 0:
            vl1 += [np.mean(np.array(u_vp[0]).reshape(num_states, num_states))]
            vl2 += [np.mean(np.array(a_vp[0]).reshape(num_states, num_states))]
            vl += [np.mean(np.array(u_vp[0]).reshape(num_states, num_states) - np.array(a_vp[0]).reshape(num_states, num_states))]
            #print('Mean value loss for goal "%s": %.3f' % (str(goals[i]), vl[-1]))
            f = plt.figure(0)
            plt.subplot(2, 3, 1)
            plt.imshow(np.array(u_vp[0]).reshape(num_states, num_states))
            plt.colorbar()
            plt.subplot(2, 3, 2)
            plt.imshow(np.array(u_vp[1], dtype=np.float64).reshape(num_states, num_states))
            plt.subplot(2, 3, 3)
            plt.imshow(cluster_arr)
            plt.subplot(2, 3, 4)
            plt.imshow(np.array(a_vp[0]).reshape(num_states, num_states))
            plt.colorbar()
            plt.subplot(2, 3, 5)
            plt.imshow(np.array(a_vp[1], dtype=np.float64).reshape(num_states, num_states))
            plt.subplot(2, 3, 6)
            plt.imshow(np.array(u_vp[0]).reshape(num_states, num_states) - np.array(a_vp[0]).reshape(num_states, num_states))
            plt.colorbar()
            f.tight_layout()
            plt.savefig(str(i)+'.png')
            plt.clf()

    print('Mean value loss across goals: %.3f' % np.mean(np.array(vl)))
    print('Mean un-abstracted value across goals: %.3f' % np.mean(np.array(vl1)))
    print('Mean abstracted value across goals: %.3f' % np.mean(np.array(vl2)))
    # plt.imshow(np.array(value).reshape(25, 25))
    # plt.show()
    # plt.imshow(np.array(policy).reshape(25, 25))
    # plt.show()
