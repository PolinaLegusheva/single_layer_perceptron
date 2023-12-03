import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_output = np.random.rand(hidden_size, output_size, )
        self.bias_output = np.random.rand(output_size)

    def relu(self, x):
        return np.maximum(0,x)

    def point_in_polygon(self, point, polygon):
        pass

    def train(self, points, polygon, learning_rate=0.001, epochs=100):
        for epoch in range(epochs):
            for point in points:
                hidden_layer_input = np.dot(point, self.weights_hidden) + self.bias_hidden
                hidden_layer_output = self.relu(hidden_layer_input)
                output = np.dot(hidden_layer_output, self.weights_output) + self.bias_output

                target = 1 if self.point_in_polygon(point, polygon) else -1
                error = target - output

                self.weights_output += learning_rate * error * hidden_layer_output.reshape(-1,1)
                self.bias_output += learning_rate * error

                hidden_layer_grad = np.dot(self.weights_output, error)
                hidden_layer_grad[hidden_layer_input <= 0] = 0

                self.weights_hidden += learning_rate * np.outer(point, hidden_layer_grad)
                self.bias_hidden += learning_rate * hidden_layer_grad

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
    def show_plot():
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Результаты работы персептрона')
        plt.show()