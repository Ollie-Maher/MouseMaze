# File for deep Q-network RL algorithm
import numpy as np

class deep_q():

    def __init__(self, action_space, input_size, layers, sizes, gamma = 0.95, epsilon = 0.1):
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_space = action_space
        self.layers = layers
        self.sizes = sizes

        self.set_shape(input_size, layers, sizes)

        self.weights = dict()
        self.biases = dict()
        self.activity_values = dict()
        self.dCdz = dict()
        
        self.activation = np.vectorize(self.ReLu)

    #set shape function
    def set_shape(self, input_size, layers, sizes):
        self.weights.clear()
        self.biases.clear()
        self.activity_values.clear()
        self.dCdz.clear()

        self.weights['layer1'] = np.empty([sizes[0], input_size], dtype=float)
        self.biases['layer1'] = np.empty(sizes[0], dtype=float)
        self.activity_values['layer1'] = np.empty([sizes[0]], dtype=float)
        self.dCdz['layer1'] = np.empty([sizes[0]], dtype=float)

        for i in range(layers-1):
            self.weights[f'layer{i+2}'] = np.empty([sizes[i+1], sizes[i]], dtype=float)
            self.biases[f'layer{i+2}'] = np.empty(sizes[i+2], dtype=float)
            self.activity_values[f'layer{i+2}'] = np.empty([sizes[i+1]], dtype=float)
            self.dCdz[f'layer{i+2}'] = np.empty([sizes[i+1]], dtype=float)
            

        '''take layer number, array-like of layer sizes
        for layer num: create 2d array of layer size * previous layer size => weights
        ?? for layer num: create 2d array of layer size => biases
        save matrices !! standard type for this?
        '''

    #set target network weights
    def target_set(self):
        self.target_weight = self.weights
        self.target_bias = self.biases

        '''
        set weights of target network to network
        '''

    #activation function
    def ReLu(a):
        return np.where(a > 0, a, 0)
        
    def ReLu_derivative(self, a):
        return np.where(a > 0, 1, 0)

    #run network
    def run_step(self, state, greedy = False, get_vals = False):
        if not greedy and np.random.rand() >= self.epsilon:
            action = np.random.randint(self.action_space)
            q_vals = np.empty(self.action_space)
        else:
            l1_input = np.add(np.matmul(self.weights['layer1'],state),self.biases['layer1'])
            if get_vals: self.activity_values['layer1'] = l1_input
            layer_ouput = self.activation(l1_input)
            for i in range(self.layers - 1):
                layer_input = np.add(np.matmul(self.weights[f'layer{i+2}'],layer_ouput),self.biases[f'layer{i+2}'])
                layer_ouput = self.activation(layer_input)
                if get_vals: self.activity_values[f'layer{i+2}'] = layer_input
            
        action = np.argmax(layer_ouput)
        q_vals = layer_ouput

        return action, q_vals

        '''take state, greedy bool
        if not greedy, epsilon chance for random actions
        output action, empty q vals
        else calculate activities L0 -> output
        output action, q vals array
        '''

    #run target network
    def run_target(self, state):
        l1_input = np.add(np.matmul(self.target_weight['layer1'],state),self.target_bias['layer1'])
        layer_ouput = self.activation(l1_input)
        for i in range(self.layers - 1):
                layer_input = np.add(np.matmul(self.target_weight[f'layer{i+2}'],layer_ouput),self.target_bias[f'layer{i+2}'])
                layer_ouput = self.activation(layer_input)
                
            
        return np.max(layer_ouput)
    
        '''take state
        calculate target activities L0 -> output
        output max q-val
        '''

    #learn function
    def update(self, state, action, reward, next_state, terminal):
        if terminal:
            target_q = reward
        
        else:
            target_q = reward + self.gamma * self.run_target(state)

        _, q_vals = self.run_step(state, greedy=True, get_vals=True)
        predicted_q = q_vals[action]

        loss_gradient = 2*(target_q - predicted_q)

        self.backprop(loss_gradient, action)

        '''take state, action, reward, next state

        if next state terminates y set to reward

        else y = r + discount * run target network

        action value = q val of action taken from greedy network
        backprop w/r to y, action value (action needed for output layer)
        '''

    #backprop function
    def backprop(self, loss_gradient, action):
        # Calculate dC/dz for the output layer
        # Only the neuron corresponding to the action taken will be affected in output layer
        self.dCdz[f'layer{self.layers}'] = np.zeros_like(self.activity_values[f'layer{self.layers}'])
        self.dCdz[f'layer{self.layers}'][action] = loss_gradient * self.ReLu_derivative(self.activity_values[f'layer{self.layers}'][action])
        
        # Backpropagate through each layer
        for layer in range(self.layers, 0, -1):
            # Gradient for weights and biases
            layer_output = self.activity_values[f'layer{layer}']
            prev_layer_output = self.activity_values[f'layer{layer-1}'] if layer > 1 else state
            
            dCdw = np.outer(self.dCdz[f'layer{layer}'], prev_layer_output)
            dCdb = self.dCdz[f'layer{layer}']
            
            # Update weights and biases for current layer
            self.weights[f'layer{layer}'] -= self.epsilon * dCdw
            self.biases[f'layer{layer}'] -= self.epsilon * dCdb
            
            # Calculate dC/dz for previous layer
            if layer > 1:
                weight_transposed = self.weights[f'layer{layer}'].T
                self.dCdz[f'layer{layer-1}'] = np.dot(weight_transposed, self.dCdz[f'layer{layer}']) * self.ReLu_derivative(prev_layer_output)



        '''takes y, action value, action

        from output layer to input
        calculates derivative of cost w/r to weights & biases
        updates weights & biases
        
        '''