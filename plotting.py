import numpy as np
import matplotlib.pyplot as plt

class Plotting:
    @staticmethod
    def plot_polygon(polygon):
        plt.plot(polygon[:, 0], polygon[:, 1], 'r-')
        plt.plot(polygon[[-1, 0], 0], polygon[[-1, 0], 1], 'r-')

    @staticmethod
    def plot_decision_boundary(weights, bias, xmin, xmax, label):
        x_vals = np.linspace(xmin, xmax, 100)
        y_vals = (-weights[0] / weights[1]) * x_vals - bias / weights[1]
        plt.plot(x_vals, y_vals, label=label)

    @staticmethod
    def plot_points(points_in, points_out):
        points_in = np.array(points_in)
        points_out = np.array(points_out)
        plt.scatter(points_in[:, 0], points_in[:, 1], c='green')
        plt.scatter(points_out[:, 0], points_out[:, 1], c='blue')

    @staticmethod
    def show_plot():
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Результаты работы персептрона')
        plt.show()