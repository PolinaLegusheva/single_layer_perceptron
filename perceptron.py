import numpy as np

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_output = np.random.rand(hidden_size, output_size, )
        self.bias_output = np.random.rand(output_size)

    def relu(self, x):
        return np.maximum(0,x)

    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if ((y1 <= y < y2) or (y2 <= y < y1)) and (x < (x2 - x1) * (y - y1)/(y2 - y1) + x1):
                inside = not inside
        return inside

    def train(self, points, polygon, learning_rate=0.001, epochs=100):
        for epoch in range(epochs):
            for point in points:
                hidden_layer_input = (point @ self.weights_hidden) + self.bias_hidden
                hidden_layer_output = self.relu(hidden_layer_input)
                output = (hidden_layer_output @ self.weights_output) + self.bias_output

                target = 1 if self.point_in_polygon(point, polygon) else -1
                error = target - output

                self.weights_output += learning_rate * error * hidden_layer_output.reshape(-1,1)
                self.bias_output += learning_rate * error

                hidden_layer_grad = self.weights_output @ error
                hidden_layer_grad[hidden_layer_input <= 0] = 0

                self.weights_hidden += learning_rate * np.outer(point, hidden_layer_grad)
                self.bias_hidden += learning_rate * hidden_layer_grad
