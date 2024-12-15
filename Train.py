from Main import *

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

start_time = time.time()
train(10000)
end_time = time.time()

print('RUNTIME = ', end_time - start_time)
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
    for i in range(113):
        #print('X_TEST', X_test[i])
        preds = test_nn.forwardProp(X_test[i])
        target = y_test[i]
        if preds[0] > preds[1]:
            pred = 1
        else:
            pred = 0

        if pred == target:
            win += 1

    print(f'percent correct {win/113}')

test()
