from typing import List, Tuple
from rrt_model import SE2Tree, SE2, CircleObstacleArray, local_planner_straight_line
import numpy as np
def get_traj(poses: List[SE2]):
    x_list = []
    y_list = []
    time_list = []
    x0 = poses[0].x
    y0 = poses[0].y
    theta0 = poses[0].theta
    xdot0 = 1 #velocity x
    ydot0 = 1 #velocity y
    xddot0 = 0.2 #roll
    xdddot0 = 0.1 #roll rate
    xdot1 = 0.5 #velocity x
    prev = SE2(x0,xdot0,theta0)
    for i in poses:
        distance = 0
        x_list.append(i.x)
        y_list.append(i.y)
        distance = np.sqrt(np.square(i.x-prev.x) + np.square(i.y-prev.y))
        velocity = 1
        time_list.append(distance/velocity)
        prev = i
    S = np.hstack([0, np.cumsum(time_list)])