from graph import *
from value_iteration import *
from domains import *

if __name__ == '__main__':
    '''
    a = build_gridworld(25, slip_rate=.1, walls=[(i,i) for i in range(15)])
    a.show(21*19, debug=False)


    b = build_four_rooms(25, walls=[(i,i) for i in range(22)])
    b.show((10, 14), debug=False)
    '''

    c = build_taxi(10, slip_rate=.1, walls=[(2, i) for i in range(7)]+[(i+3, 6) for i in range(4)], debug=False)
    c.show(passenger=(4,4), goal=(0,0), debug=True)
