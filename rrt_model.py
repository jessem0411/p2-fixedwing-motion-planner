import numpy as np
from numpy.random import rand
from typing import List, Tuple

class SE2:
    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

    def __repr__(self):
        return repr((self.x, self.y, self.theta))

    @classmethod
    def rand(cls, x_range: Tuple[float,float],
             y_range: Tuple[float,float], theta_range: Tuple[float,float]):
        min_val = np.array([x_range[0], y_range[0], theta_range[0]])
        max_val = np.array([x_range[1], y_range[1], theta_range[1]])
        diff = max_val - min_val
        new_val = rand(3).dot(np.diag(diff)) + min_val
        return cls(x=new_val[0], y=new_val[1], theta=new_val[2])

def local_planner_straight_line(
        start: SE2, goal: SE2, distance: float, dtheta_max: float) -> SE2 :
    p0 = np.array([start.x, start.y])
    p1 = np.array([goal.x, goal.y])
    dp = p1 - p0
    dp = dp/np.linalg.norm(dp)
    theta = np.arctan2(dp[1], dp[0])
    dtheta = (theta - start.theta + np.pi) % (2*np.pi) - np.pi
    p = p0 + dp*distance
    if np.abs(dtheta) > dtheta_max:
        return None
    else:
        return SE2(p[0], p[1], theta)

class SE2Tree:
    def __init__(self, pose: SE2):
        self.parent = None   # type: SE2Tree
        self.pose = pose    # type: SE2
        self.children = []  # type: List[SE2Tree]

    def add_child(self, child) -> None:
        assert isinstance(child, SE2Tree)
        child.parent = self
        self.children.append(child)

    def find_closest(self, pose:SE2):
        d = np.linalg.norm([self.pose.x-pose.x, self.pose.y-pose.y])
        closest = self
        for child in self.children:
            d_child, closest_to_child = child.find_closest(pose)
            if d_child < d:
                d = d_child
                closest = closest_to_child
        return d, closest

    def path(self) -> List[SE2]:
        if self.parent is None:
            return [self.pose]
        else:
            return self.parent.path() + [self.pose]

class CircleObstacleArray:
    def __init__(self, obstacles: np.array):
        assert obstacles.shape[1] == 3
        self.data = obstacles

    @classmethod
    def generate_uniform(cls, x_range: Tuple[float,float],
                         y_range: Tuple[float,float], r_range: Tuple[float,float],
                         samples:int = 10):
        min_val = np.array([x_range[0], y_range[0], r_range[0]])
        max_val = np.array([x_range[1], y_range[1], r_range[1]])
        return cls(np.random.rand(samples,3).dot(
            np.diag(max_val-min_val)) + min_val)

    def closest_obstacle(self, x: float, y: float) -> Tuple[int,float]:
        dist = np.linalg.norm(self.data[:, :2] - np.array([
            x, y]), axis = 1) - self.data[:,2]
        i_min = np.argmin(dist)
        return int(i_min), dist[i_min]

    def __repr__(self):
        return repr(self.data)