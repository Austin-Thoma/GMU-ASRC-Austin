from matplotlib import pyplot as plt
from matplotlib import patches
from circle_fit import taubinSVD
import generate_data
import math

if __name__ == '__main__':
    #   Generate dataset of x and y points
    num_robots = 10
    radius = 10
    point_coordinates = generate_data.generate_circle_coordinates(num_robots, radius)

    for coordinate in point_coordinates:
        print(coordinate)

    #   Fit a circle to the data
    xc, yc, r, sigma = taubinSVD(point_coordinates)
    print("xc, yc, r, sigma:", xc, yc, r, sigma)

    # Plotting
    x_coordinates = [coordinate[0] for coordinate in point_coordinates]
    y_coordinates = [coordinate[1] for coordinate in point_coordinates]

    fig, ax = plt.subplots()

    ax.scatter(x_coordinates, y_coordinates)

    ax.set_xlim(-15,15)
    ax.set_ylim(-15,15)

    ax.grid()
    ax.set_title('Robots')

    ax.plot(xc, yc, 'ro')

    circle = patches.Circle((xc, yc), r, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')

    plt.show()

    #   calculate new metric based on the circle
    sigma = 0
    for i in range(num_robots):
        sigma += math.dist(point_coordinates[i], [num_robots * xc, num_robots * yc])
    
    sigma /= num_robots
    sigma -= r

    print(sigma)
