import numpy as np

# Set up data
X = np.array([0.5, 0.6, 0.7]) # Example feature with 3 neurons
y = np.array([1]) # Target label; this is what we EXPECT the output to be

# Initial weights and bias
weights = np.array([0.3, 0.2, 0.4]) # Initial weights are an arbitrary value
bias = 0.0 # Starting with 0.0 for bias; this will adjust during training

# Sigmoid function for activation
# This takes in a number as input and maps it to a number between 0 and 1
#  sigmoid(x) = 1 / (1 + exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
# When sigmoid is used as the activation function (as it is in this code),
# the derivative is used to calculate the gradient.  The derivative measures how much the
# output of the sigmoid function would change with a small change in input.
#  sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Hyperparameters
epochs = 100  # We'll make 10 passes through the data
learning_rate = 0.1 # Determines the size of "steps" to make as we adjust weights

# Training process
for epoch in range(epochs):
    # Forward pass
    weighted_sum = np.dot(X, weights)  # Multiply the nodes by weights and then sum them all up  (dot product)
    prediction = sigmoid(weighted_sum) # Run the activation function (sigmoid) to get a number between 0 and 1
    
    # Compute loss using the Mean Squared Error function
    #  MSE = (1/n) * Σ(y - prediction)^2
    # where n is the number of samples, y is the value returned by the model, and prediction is the predicted value
    loss = np.mean((y - prediction) ** 2)

    # Backpropagation
    d_loss_prediction = -2 * (y - prediction) # Derivative of loss w.r.t prediction
    d_prediction_d_weighted_sum = sigmoid_derivative(weighted_sum) # Derivative of prediction w.r.t weighted sum
    d_weighted_sum_d_weights = X # Derivative of weighted sum w.r.t weights
    d_weighted_sum_d_bias = 1 # Derivative of weighted sum w.r.t bias

    # Gradient for weights and bias
    gradients = d_loss_prediction * d_prediction_d_weighted_sum * d_weighted_sum_d_weights
    gradient_bias = d_loss_prediction * d_prediction_d_weighted_sum * d_weighted_sum_d_bias

    # Update weights and bias (set at the top of the code)
    weights -= learning_rate * gradients
    bias -= learning_rate * gradient_bias

    # Print epoch number and average loss for the epoch
    # Loss should improve (get smaller) with each epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# Final weights and bias after training
print("\nFinal weights after training:", weights)
print("Final bias after training:", bias)