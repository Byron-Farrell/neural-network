from apex_matrix.matrix import Matrix
from numpy.matrixlib.defmatrix import matrix
from numpy.random import default_rng
from numpy import exp, clip
SEED = 4815162342
random_number_generator = default_rng(SEED)

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers
        self.number_of_layers = len(layers)
        self.weights = self.randomize_weights()
        self.biases = self.randomize_biases()

    def randomize_weights(self):
        # Creates a list matrices of randomized weights of CurrentLayerNeuron x PreviousLayerNeuron

        list_of_weights = []

        for layer in range(1, self.number_of_layers):
            weights_matrix = Matrix.zero((self.layers[layer], self.layers[layer - 1]))
            for row in range(weights_matrix.rows):
                for col in range(weights_matrix.columns):
                    weights_matrix[row][col] = random_number_generator.random()
            list_of_weights.append(weights_matrix)

        return list_of_weights

    def randomize_biases(self):
        # returns a list of matrices of randomized biases per layer

        list_of_biases = []

        for layer_size in self.layers[1:]:
            bias_matrix = Matrix.zero((layer_size, 1))

            for row in range(bias_matrix.rows):
                bias_matrix[row][0] = random_number_generator.random()

            list_of_biases.append(bias_matrix)

        return list_of_biases


    def feed_forward(self, inputs):
        weighted_sums = []
        activations = []
        activation = inputs

        activations.append(inputs)
        for layer in range(len(self.weights)):
            weighted_sum = (self.weights[layer] * activation) + self.biases[layer]
            weighted_sums.append(weighted_sum)
            activation = self.matrix_sigmoid(weighted_sum)
            activations.append(activation)

        return weighted_sums, activations

    def matrix_sigmoid(self, z):
        z1 = Matrix.zero((z.rows, z.columns))
        for row in range(z.rows):
            for column in range(z.columns):
                z1[row][column] = self.sigmoid(z[row][column])
        return z1

    def sigmoid(self, z):
        return float(1 / (1 + 2.718281828**-z))


    def SGD(self, inputs, outputs, learning_rate, epoch, batch_size=None):
        if not batch_size:
            batch_size = len(inputs)

        input_batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        output_batches = [outputs[i:i + batch_size] for i in range(0, len(outputs), batch_size)]

        for _ in range(epoch):
            for input_batch, output_batch in zip(input_batches, output_batches):
                self.mini_batch(input_batch, output_batch, learning_rate)


    def matrix_sigmoid_derivative(self, z):
        for row in range(z.rows):
            for column in range(z.columns):
                z[row][column] = self.sigmoid_derivative(z[row][column])
        return z

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cost_derivate(self, output, expected_output):
        return 2 * (output - expected_output)

    def mini_batch(self, inputs, expected_outputs, learning_rate):
        w_gradients_sum = [Matrix.zero((weight.rows,weight.columns)) for weight in self.weights]
        b_gradients_sum = [Matrix.zero((bias.rows,bias.columns)) for bias in self.biases]
        n = len(inputs)

        for x, y in zip(inputs, expected_outputs):
            w_gradients, b_gradients = self.back_prop(x, y)

            for i in range(len(w_gradients)):
                w_gradients_sum[i] += w_gradients[i]
                b_gradients_sum[i] += b_gradients[i]

        for i in range(len(w_gradients_sum)):
            w_gradients_sum[i] *= 1/n
            b_gradients_sum[i] *= 1/n
            self.weights[i] -= learning_rate * w_gradients_sum[i]
            self.biases[i] -= learning_rate * b_gradients_sum[i]



    def back_prop(self, x, y):
        weighted_sums, activations = self.feed_forward(x)
        w_gradients = []
        b_gradients = []

        delta = self.cost_derivate(activations[-1], y)
        sigmoid_derivative = self.matrix_sigmoid_derivative(weighted_sums[-1])

        delta = delta.element_wise_product(sigmoid_derivative)

        w_gradient = delta * activations[-2].transpose()
        w_gradients.append(w_gradient)
        b_gradient = delta
        b_gradients.append(b_gradient)

        for layer in range(2, len(activations)):
            sigmoid_derivative = self.matrix_sigmoid_derivative(weighted_sums[-layer])

            delta =  self.weights[-layer+1].transpose() * delta
            delta = sigmoid_derivative.element_wise_product(delta)

            w_gradient = delta * activations[-layer-1].transpose()
            w_gradients.append(w_gradient)
            b_gradient = delta
            b_gradients.append(b_gradient)


        w_gradients.reverse()
        b_gradients.reverse()

        return w_gradients, b_gradients



if __name__ == '__main__':
    neural_network = NeuralNetwork([2, 2, 1])

    inputs = [
        Matrix([[0],[0]]),
        Matrix([[1], [0]]),
        Matrix([[0], [1]]),
        Matrix([[1], [1]])
    ]

    outputs = [
        Matrix([[0]]),
        Matrix([[1]]),
        Matrix([[1]]),
        Matrix([[0]])
    ]

    neural_network.SGD(inputs, outputs, 5, 10000)

    w, a = neural_network.feed_forward(inputs[0])
    print(f'activation = {a[-1]}')
    w, a = neural_network.feed_forward(inputs[1])
    print(f'activation = {a[-1]}')
    w, a = neural_network.feed_forward(inputs[2])
    print(f'activation = {a[-1]}')
    w, a = neural_network.feed_forward(inputs[3])
    print(f'activation = {a[-1]}')
