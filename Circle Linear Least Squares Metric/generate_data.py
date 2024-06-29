import math
import random

def generate_circle_coordinates(num_robots, radius):
    coordinates = []
    
    for i in range(num_robots):
        r_thresh = 0.5 * radius
        theta = random.random() * 2 * math.pi
        curr_r = random.uniform(radius - r_thresh, radius + r_thresh)
        coordinates.append([curr_r * math.cos(theta), curr_r * math.sin(theta)])

    return coordinates