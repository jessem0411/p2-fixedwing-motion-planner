from rrt_model import SE2Tree, SE2, CircleObstacleArray, local_planner_straight_line
from rrt_view import draw_se2_tree, draw_circle_obstacles, draw_se2, draw_se2_path
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin
from math import sqrt

def main():
    obstacles = CircleObstacleArray.generate_uniform(
        x_range= (1,9), y_range= (1,9), r_range=(0.5,1), samples=10
    )

    start = SE2(0,0,0)
    goal = SE2(10, 10, 1)

    root = SE2Tree(start)
    node = root
    for i in range(5000):
        end_goal = np.random.rand() < 0.1
        if end_goal:
            tmp_goal = goal
        else:
            tmp_goal = SE2.rand((0,10),(0,10),(-np.pi,np.pi))
        d , closest = root.find_closest(tmp_goal)
        if( end_goal and d<1):
            break
        pose_new = local_planner_straight_line(
            start=closest.pose, goal=tmp_goal,
            distance=1, dtheta_max=0.5
        )
        if(pose_new is None):
            continue
        close_i, close_d = obstacles.closest_obstacle(pose_new.x, pose_new.y)
        if( close_d < 0):
            continue
        else:
            child = SE2Tree(pose_new)
            closest.add_child(child)

    #trajectory generation
        #A matrix 
    a = np.array([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5],
        [0, 0, 2, 0, 0, 0], 
        [0, 0, 2, 6, 12, 20]
    ])

    #separate x, y, theta
    data = closest.path()

    thetas = [d.theta for d in data]
    posx = [d.x for d in data]
    posy = [d.y for d in data]

    velx, vely = calc_initial(thetas)
    #b matrix 
    bx = []
    by = []
    accel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    for i in range(0, len(posx) - 1):
        bx.append(posx[i])
        bx.append(posx[i + 1])
        bx.append(velx[i])
        bx.append(velx[i + 1])
        bx.append(accel[i])
        bx.append(accel[i+1])

        by.append(posy[i])
        by.append(posy[i + 1])
        by.append(vely[i])
        by.append(vely[i + 1])
        by.append(accel[i])
        by.append(accel[i+1])

    b = [bx, by]

    #polynomial values
    xvals = calc_values(a, bx)
    yvals = calc_values(a, by)

    t = np.linspace(0, 1)
    
    #smoothed trajectory
    posxnew = [x for row in xvals for x in calc_pos(row, t)]
    posynew = [y for row in yvals for y in calc_pos(row, t)]
    velxnew = [x for row in xvals for x in calc_vel(row, t)]
    velynew = [y for row in yvals for y in calc_vel(row, t)]
    accelxnew = [x for row in xvals for x in calc_accel(row, t)]
    accelynew = [y for row in yvals for y in calc_accel(row, t)]
    xtripledot = [x for row in xvals for x in calc_tripledot(row,t)]
    ytripledot = [y for row in yvals for y in calc_tripledot(row, t)]
    plot_position(posxnew, posynew, 'Smoothed Trajectory')
    plot_roll(t, velxnew, velynew, accelxnew, accelynew, 'Roll Angle vs. Time')
    plot_rollrate(t, velxnew, velynew, accelxnew, accelynew, xtripledot, ytripledot, 'Roll Rate vs. Time')

    ax = plt.gca()
    h1 = draw_se2(ax,start,color='b', label='start')
    h2 = draw_se2(ax,goal,color='g', label='goal')
    h3 = draw_circle_obstacles(ax, obstacles, color='r', label='obstacle')[0]
    h4 = draw_se2_tree(ax, root, color='k', label='tree')[0]
    h5 = draw_se2_path(ax, closest.path(),'y.-', label = 'path', alpha = 0.5, linewidth=5)
    plt.plot(posxnew,posynew)
    plt.legend(handles=[h1,h2,h3,h4,h5], loc = 'upper left', ncol=2)
    plt.xlabel('x,m')
    plt.ylabel('y,m')
    plt.title('RRT')
    plt.axis([-2,12,-2,12])
    plt.grid()
    plt.show()
    print('done')

def calc_initial(theta):
    #calculates initial velocity conditions
    initial_velx = []
    initial_vely = []

    for x in range(0, len(theta)):
        velx = cos(theta[x])
        initial_velx.append(velx)

        vely = sin(theta[x])
        initial_vely.append(vely)
    
    return initial_velx, initial_vely

def calc_values(A, b):
    #calculates trajectory values given A matrix and B initial conditions
    values = []
    b = np.transpose(b)
    start = 0
    for x in range(len(b) + 1):
        if x % 6 == 0 and x > 0: 
            tempb = b[start:x]
            out = np.linalg.inv(A).dot(tempb)
            values.append(out)
            
            start += 6
        else: 
            continue 
            
    return values

def calc_pos(val, t, n=5):
    #calculates position given polynomial constant, time and degree of polynomial (n)
    result = []

    for dt in t:
        v = 0
        for i in range(n + 1):
            v += val[i] * (dt ** i)
        result.append(v)
    
    return result


def calc_vel(val, t, n=5):
    result = []

    for dt in t:
        v = 0
        for i in range(n):
            v += val[i+1] * (dt ** i) * (i+1)
        result.append(v)

    return result
def calc_accel(val, t, n=5):
    result = []

    for dt in t:
        v = 0
        for i in range(n-1):
            v += val[i+2] * (dt ** i) * (i+2) * (i+1)
        result.append(v)

    return result

def calc_tripledot(val, t, n=5):
    result = []

    for dt in t:
        v = 0
        for i in range(n-2):
            v += val[i+3] * (dt ** i) * (i+3) * (i+2) * (i+1)
        result.append(v)

    return result

def plot_position(x, y, start):
    #plots position
    plt.plot(x, y)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title( start)
    plt.xlim(-2,12)
    plt.ylim(-2,12)
    plt.grid()
    plt.show()

def plot_roll(t, velx, vely, accelx, accely, start):
    t = np.linspace(0,1,len(velx))
    psi = np.arctan2(vely, velx) * 180 /np.pi
    V = np.sqrt(np.power(velx,2) + np.power(vely,2))
    psidot = [(velx[i]*accely[i]-vely[i]*accelx[i])/np.power(V[i],2) for i in range(len(velx))]
    phi = np.arctan2(V*psidot,9.8) * 180/np.pi
    plt.plot(t, phi)
    plt.xlabel('time [s]')
    plt.ylabel('Roll Angle [degrees]')
    plt.title(start)
    plt.grid()
    plt.show()

def plot_rollrate(t, velx, vely, accelx, accely, xtripledot, ytripledot, start):
    t = np.linspace(0,1,len(velx))
    V = np.sqrt(np.power(velx,2) + np.power(vely,2))
    Vdot = [1/(2*V[i])*(2*velx[i]*accelx[i] - 2*vely[i]*accely[i]) for i in range(len(velx))]
    psidot = [(velx[i]*accely[i]-vely[i]*accelx[i])/np.power(V[i],2) for i in range(len(velx))]
    psiddot = [((velx[i]*ytripledot[i] - vely[i]*xtripledot[i])*np.power(V[i],2) -
               (velx[i]*accely[i]-vely[i]*accelx[i])*(2*velx[i]*accelx[i]+2*vely[i]*accely[i]))/np.power(V[i],4)
               for i in range(len(velx))]
    phidot = [1/(1+(V[i]*psidot[i]/9.8)**2)*(Vdot[i]*psidot[i] + V[i]*psiddot[i])/9.8*180/np.pi for i in range(len(velx))]
    plt.plot(t, phidot)
    plt.xlabel('time [s]')
    plt.ylabel('Roll Rate [degrees/s]')
    plt.title(start)
    plt.grid()
    plt.show()

main()
