# Neural Network from Scratch

A feedforward neural network made only using NumPy - without ML libraries.
Tested on the sklearn breast cancer dataset (binary classification)
Gets to 96 percent accuracy in 10,000 (random so accuracy is variable)
Trained on the [Wisconsin Breast Cancer dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) (binary classification). Achieves **~96% test accuracy** after 10,000 epochs.

---

## Why

I wanted to understand how AI worked (due to the 2020 ChatGPT craze), so I read up on how they worked, the basic linear algebra I needed (gradient descent, etc.)

---

## Architecture 
- 3 fcl: **30 -> 32 -> 16 -> 2**
- sigmoid activations throughout
- stochastic gradient descent (32 batch size)
- MSE loss, learning rate = 0.001

---

- `Main.py` - the network: `forwardProp`, `backProp`, `step`
- `Train.py` - training loop, loss tracking, accuracy test
- `Data.py` - loads breast cancer dataset, creates batches

---

# Setup 

```bash
git clone git@github.com:toby-brett/nnFromScratch.git
cd nnFromScratch
pip install -r requirements.txt
python Train.py
```
Trains for 10,000 epochs, plots loss over epoch, and prints accuracy on test set. 

### What I would change 
- no dropout, or batch normalisaiton, so training performance is poor
- would experiment with different loss functions
