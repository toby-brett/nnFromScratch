import numpy as np 
from Data import *

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))  # squishes numbers from -infinity to +infinity between 0 and 1

def sigmoidDerivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig) # the gradient function of sigmoid

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return np.where(x > 0, 1, 0)

class model():
    def __init__(self, params=0):
        
        self.m = 32 # batchsize
        self.learnrate = 0.001

        if params != 0:
            self.m = 1
            self.weightLayer1 = np.array(params[0])
            self.weightLayer2 = np.array(params[1])
            self.weightLayer3 = np.array(params[2])
            self.biasLayer1 = np.array(params[3])
            self.biasLayer2 = np.array(params[4])          
            self.biasLayer3 = np.array(params[5])
        
        else:
            self.weightLayer1 = np.random.uniform(low=-1, high=1, size=(32, 30)) # 32 in first layer, takes 30 paramter inputs, this initialises an array with random values between -1 and 1
            self.biasLayer1 = np.zeros(shape=(32, self.m)) # 32 weights for the first layer 
            self.weightLayer2 = np.random.uniform(low=-1, high=1, size=(16, 32)) # 16 outputs 32 inputs
            self.biasLayer2 = np.zeros(shape=(16, self.m))
            self.weightLayer3 = np.random.uniform(low=-1, high=1, size=(2, 16)) # 1 outputs (binary classification)
            self.biasLayer3 = np.zeros(shape=(2, self.m))

    def forwardProp(self, x):

         # loaded data in wrong dims
        self.inputs = x.transpose()
        self.logits1 = (self.weightLayer1 @ self.inputs) + self.biasLayer1
        self.activation1 = sigmoid(self.logits1)
        self.logits2 = (self.weightLayer2 @ self.activation1) + self.biasLayer2
        self.activation2 = sigmoid(self.logits2)
        self.logits3 = (self.weightLayer3 @ self.activation2) + self.biasLayer3
        self.activation3 = sigmoid(self.logits3)
        
        return self.activation3
    
    def backProp(self, preds, target):

        # d is delta, use upper case for variabales, and number after variable is layer.
        # e.g dA3dZ3 = the derivative of the third layer activations, with respect to the third layer logits 

        ### layer 3 ###
        
        # derivative of loss with respect to W3 and B3   

        self.dLdA3 = np.array([preds[0] - target[0], preds[1] - target[1]]) 
        self.dA3dZ3 = sigmoidDerivative(self.logits3) # output being logits (z), pre sigmoid (always positive)
        self.dZ3dW3 = self.activation2.transpose() # z = W * a + b, so dZ/dW = a.t
        self.dZ3dB3 = np.ones(shape=(self.m, 1))

        self.dLdW3 = (self.m**-1) * ((self.dLdA3 * self.dA3dZ3) @ self.dZ3dW3)
        self.dLdB3 = (self.m**-1) * ((self.dLdA3 * self.dA3dZ3) @ self.dZ3dB3)

        ### layer 2 ###

        # derivative of loss with respect to W2 and B2
        self.dZ3dA2 = self.weightLayer3.transpose()
        self.dA2dZ2 = sigmoidDerivative(self.logits2) 
        self.dZ2dW2 = self.activation1.transpose()
        self.dZ2dB2 = np.ones(shape=(self.m, 1))

        self.dLdW2 = (self.m**-1) * ((self.dZ3dA2 @ (self.dLdA3 * self.dA3dZ3) * self.dA2dZ2) @ self.dZ2dW2)
        self.dLdB2 = (self.m**-1) * ((self.dZ3dA2 @ (self.dLdA3 * self.dA3dZ3) * self.dA2dZ2) @ self.dZ2dB2)

        ### layer 1 ###

        self.dZ2dA1 = self.weightLayer2.transpose() # matrix 32x32
        self.dA1dZ1 = sigmoidDerivative(self.logits1)
        self.dZ1dW1 = self.inputs.transpose()
        self.dZ1dB1 = np.ones(shape=(self.m, 1)) # 1x1 mat

        # derivative of loss with respect to W1 and B1
                        
        self.dLdW1 = (self.m**-1) * ((self.dZ2dA1 @ (self.dZ3dA2 @ (self.dLdA3 * self.dA3dZ3) * self.dA2dZ2) * self.dA1dZ1) @ self.dZ1dW1)
        self.dLdB1 = (self.m**-1) * ((self.dZ2dA1 @ (self.dZ3dA2 @ (self.dLdA3 * self.dA3dZ3) * self.dA2dZ2) * self.dA1dZ1) @ self.dZ1dB1)

        self.step((self.dLdW3, self.dLdB3, self.dLdW2, self.dLdB2, self.dLdW1, self.dLdB1), (self.weightLayer3, self.biasLayer3, self.weightLayer2, self.biasLayer2, self.weightLayer1, self.biasLayer1))

    def step(self, derivatives, parameters):
        for derivative, parameter in zip(derivatives, parameters):
            parameter -= self.learnrate * derivative

       

train_data = train_batches

nn = model()
losses = []

def train(epochs):
    global losses
    for epoch in range(epochs):
        for batch in train_data:
            preds = nn.forwardProp(batch[0]) 
            target_list = [[],[]]
            for target_output in batch[1]:
                
                if target_output == 1:
                    target_list[0].append(1)
                    target_list[1].append(0)

                elif target_output == 0:
                    target_list[0].append(0)
                    target_list[1].append(1)
                    
            target = np.array(target_list)

            nn.backProp(preds, target)
            
            total_loss = 0
            total_loss += np.sum(0.5 * (preds[0] - target[0]) ** 2)
        
        total_loss /= len(train_data)
        losses.append(total_loss)

def get_parameters():

    saved_weights1 = nn.weightLayer1
    saved_weights2 = nn.weightLayer2
    saved_weights3 = nn.weightLayer3
    saved_bias1 = nn.biasLayer1.transpose()[0]
    saved_bias2 = nn.biasLayer2.transpose()[0]
    saved_bias3 = nn.biasLayer3.transpose()[0]
    
    return(saved_weights1, saved_weights2, saved_weights3,saved_bias1, saved_bias2, saved_bias3)

        
import matplotlib.pyplot as plt

train(10000)
params = get_parameters()
test_nn = model(params)

plt.plot(losses)

# Add title and labels
plt.title("Graph of losses")
plt.xlabel("time")
plt.ylabel("loss")

# Display the plot
plt.show()

def test():
    win = 0
    for i in range(100):
        #print('X_TEST', X_test[i])
        preds = test_nn.forwardProp(X_test[i])
        target = y_test[i]
        if preds[0] > preds[1]:
            pred = 1
        else:
            pred = 0

        if pred == target:
            win += 1

    print(f'percent correct {win/100}')

test()