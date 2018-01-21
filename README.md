# Text-Generation-RNN
In this project, we use a Recursive Neural Network with LSTM cells to generate text. We use a character-based dictionary to predict the next character at each time step.
During training time, the input of the network is a window of 40 characters, in a one-hot vector with the same length of our dictionary, and stride 3. And during test time, we concatenate the predicted output to the input and continue our generation.
