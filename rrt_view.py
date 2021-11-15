import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import  Patch
import numpy as np
from typing import List

from rrt_model import SE2Tree, SE2, CircleObstacleArray

def draw_circle_obstacles(ax: plt.Axes, obstacles: CircleObstacleArray, *args,**kwargs):
    col = PatchCollection([plt.Circle((x,y),radius)
                           for x, y, radius in obstacles.data],*args,**kwargs)
    ax.add_collection(col)
    return Patch(*args,**kwargs), col

def draw_se2(ax: plt.Axes, pose: SE2, *args, **kwargs):
    return ax.quiver(
        pose.x, pose.y, np.cos(pose.theta), np.sin(pose.theta), *args, **kwargs)

def draw_se2_path(ax:plt.Axes, poses: List[SE2], *args, **kwargs):
    return ax.plot([pose.x for pose in poses],[pose.y for pose in poses], *args, **kwargs)[0]

def draw_se2_tree(ax: plt.Axes, node: SE2Tree, *args, **kwargs):
    h =None
    for child in node.children:
        h = ax.plot([node.pose.x, child.pose.x], [node.pose.y, child.pose.y], *args, **kwargs)
        draw_se2_tree(ax,child,*args,**kwargs)
    return h