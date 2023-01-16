class node:
    
    def __init__(self, node_label, is_output_node=False):
        '''
        Inputs:
            - node_label (str): label for the node, used for reference
            - is_output_node: TRUE when node is output node
        '''
        self.node_label = node_label
        self.weight_arr = []
        self.input_arr = []
        self.is_output_node = is_output_node
        
        self.output_value = 0
        
        self.error_term = 0
        
    def initializeWeights(self, weight_init_range, number_of_inputs):
        '''
        Creates array of weights which include:
        w_0, w_1, ... w_[number_of_inputs]
        '''
        self.weight_arr = np.random.uniform(low=weight_init_range[0], high=weight_init_range[1], size=number_of_inputs+1)
        
    def getNodeWeights(self):
        return self.weight_arr
    
    def setNodeOutput(self, x_arr):
        '''
        x_arr: values for inputs of node
        '''
        self.input_arr = x_arr
        y = np.dot(self.weight_arr, x_arr)
        z = 1/ (1 + np.exp(-y))
        self.output_value = z
    
    def getNodeOutput(self):
        return self.output_value

    def __str__(self):
        return f'Weights for {self.node_label}: {self.weight_arr}' 
    
    def __repr__(self):
        return f"<Node: {self.node_label} Weight: {self.weight_arr}>"

class layer:
    
    def __init__(self, layer_label):
            '''
            Inputs: 
                - layer_label (str): name of the label
                - node_dict (dictionary): nodes in the layer
            '''
            self.layer_label = layer_label
            self.is_output_layer = False
            self.node_dict = {}
            
            self.layer_inputs = []
            self.layer_outputs = []
            self.layer_errors = []    

    def initializeLayerNodes(self, number_of_nodes, number_of_inputs, weight_init_range=[-0.3, 0.3]):
        '''
        Initializes a new layer with a given number of nodes, and randomizes the
        weights of the nodes based on range.
        Inputs:
            - number_of_nodes (int): Number of nodes to be included (does not include bias node)
            - number_of_inputs (int): Number of inputs to each of the nodes
            - weight_init_range (array, float): Range for the randomized weights for nodes
        '''

        # Create all of the other nodes for layer
        for ind in range(0, number_of_nodes):
            node_label = f'node_{ind}'
            cur_node = node(node_label)
            cur_node.initializeWeights(weight_init_range, number_of_inputs)
            self.node_dict[node_label] = cur_node
            
    def setLayerInputs(self, input_arr):
        '''
        input_arr: does not contain bias value
        sets layer_inputs to include inputs wiht bias
        '''
        num_inputs_with_bias = len(input_arr) + 1
        new_input = np.ones(num_inputs_with_bias)
        for ind in range(1, num_inputs_with_bias):
            new_input[ind] = input_arr[ind-1]
        self.layer_inputs = new_input

    def calculateOutputAtLayerNodes(self):
        x_arr = self.layer_inputs
        layer_outputs=[]
        layer_errors = []
        for n in self.node_dict:
            cur_node = self.node_dict[n]
            cur_node.setNodeOutput(x_arr)
            node_output = cur_node.getNodeOutput()
            layer_outputs.append(node_output)
            layer_errors.append(0)
        self.layer_outputs = layer_outputs
        self.layer_errors = layer_errors
        
    def calculateOutputLayerError(self, t):
        z = self.layer_outputs[0]
        delta = z * (1-z) * (t - z)
        self.layer_errors[0] = delta
        
    def getLayerOutputs(self):
        return self.layer_outputs
    
    
class feedforwardNN:
    
    def __init__(self):
        self.layers = {}
        self.training_speed = 0.3
        self.number_of_layers = 0
        self.output_layer_label = ''
        
    def initializeNN(self, feature_arr, number_of_nodes_for_layers, weight_init_range):
        
        number_of_layers = len(number_of_nodes_for_layers)
        self.number_of_layers = number_of_layers
        
        num_inputs_at_layer = len(feature_arr)
        x_arr = feature_arr
        
        for ind in range(0, number_of_layers):
            layer_label = f'layer_{ind}'
            number_of_nodes = number_of_nodes_for_layers[ind]
            # Create the layer
            l = layer(layer_label)
            l.initializeLayerNodes(number_of_nodes, num_inputs_at_layer, weight_init_range)
            l.setLayerInputs(x_arr)
            l.calculateOutputAtLayerNodes()
            
            if ind == number_of_layers-1:
                l.is_output_layer == True
            
            self.layers[layer_label] = l

            # Reset the values for the next layer
            x_arr = l.getLayerOutputs()

            num_inputs_at_layer = len(x_arr)
        
        self.output_layer_label = layer_label
        
        
            
    def updateNNOutputs(self, feature_arr):
        num_inputs_at_layer = len(feature_arr)
        x_arr = feature_arr
        
        for ind in range(0, self.number_of_layers):
            layer_label = f'layer_{ind}'
            l = self.layers[layer_label]
            l.setLayerInputs(x_arr)
            l.calculateOutputAtLayerNodes()
            
            # Reset the values for the next layer
            x_arr = l.getLayerOutputs()
            num_inputs_at_layer = len(x_arr)
            
            
    def performBackprop(self, feature_arr, truth_value):
        
        #Calculate all of the outputs for the given inputs. 
        self.updateNNOutputs(feature_arr)
        
        # Start with error calculation for the last layer
        layer_list = list(self.layers.keys())
        layer_list.reverse()
        
        for label in layer_list: 
            l = self.layers[label]
            if label == self.output_layer_label:
                # Calculate error terms for output layer
                l.calculateOutputLayerError(truth_value)
                error_term = l.layer_errors
                l.node_dict['node_0'].error_term = error_term[0]
                
            else:
                # Calculate error terms for hidden layer
                prev_layer_errors = prev_l.layer_errors
                cur_layer_nodes = l.node_dict.keys()
                layer_errors = []
                for n_label in cur_layer_nodes:
                    cur_node = l.node_dict[n_label]
                    w_arr = cur_node.weight_arr
                    error_sum = 0
                    for p in prev_layer_errors:
                        for w_val in w_arr:
                            error_sum += p*w_val
                    z = cur_node.output_value
                    delta_cur_node = z * (1-z) * error_sum
                    cur_node.error_term = delta_cur_node
                    layer_errors.append(delta_cur_node)
                l.layer_errors = layer_errors
            prev_label = label
            prev_l = l
        
        # Update all of the network weights
        layer_list.reverse()
        eta = self.training_speed
        for l_label in layer_list:
            cur_layer = self.layers[l_label]
            for node_label in cur_layer.node_dict:
                cur_node = cur_layer.node_dict[node_label]
                delta = cur_node.error_term
                for ind in range(0, len(cur_node.weight_arr)):
                    w = cur_node.weight_arr[ind]
                    x = cur_node.input_arr[ind]
                    new_weight = w + (eta * delta * x)
                    cur_node.weight_arr[ind] = new_weight
                    
    def trainNetwork(self, train_df, col_list):
        # Iterate through each of the training datapoints
        error_val_arr = []
        num_correct = 0
        num_total = 0
        for ind in range(0, len(train_df)):
            feature_arr = []
            for c in col_list:
                input_val = train_df.iloc[ind][c]
                feature_arr.append(input_val)
            
            # Get truth value
            truth_val = train_df.iloc[ind]['Class']
            
            # Perform backprop
            self.performBackprop(feature_arr, truth_val)
            
            # Get output value of network, compare it to input
            predicted_val = self.layers[self.output_layer_label].layer_outputs[0]
            
            if predicted_val >0.5:
                predicted_val = 1
            else:
                predicted_val = 0
            
            num_total+=1 
            if truth_val == predicted_val:
                num_correct+=1
            

            cur_err = num_correct/num_total
            error_val_arr.append(cur_err)
        return error_val_arr
            
    
    def getPredictedValue(self):
        predicted_val = self.layers[self.output_layer_label].layer_outputs[0]
        if predicted_val > 0.5:
            p = 1
        else:
            p = 0   
        return p
                
    def getClassificationError(self, test_df, col_list):
        # Iterate through each of the training datapoints
        num_total = len(test_df)
        num_classified_correctly = 0
        num_misclassified = 0
        for ind in range(0, len(test_df)):
            feature_arr = []
            for c in col_list:
                input_val = test_df.iloc[ind][c]
                feature_arr.append(input_val)
                
            #Calculate all of the outputs for the given inputs. 
            self.updateNNOutputs(feature_arr)
            
            truth_val = test_df.iloc[ind]['Class']
            predicted_val = self.getPredictedValue()
            
            if predicted_val >0.5:
                predicted_val = 1
            else:
                predicted_val = 0
            
            if truth_val == predicted_val:
                num_classified_correctly+=1
            else:
                num_misclassified+=1
        
        error_rate = (num_misclassified/num_total)
        return error_rate
            
