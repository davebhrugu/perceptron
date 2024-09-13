# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from scipy.sparse import csr_matrix  # For working with sparse matrices (unused in this code)

# Load the dataset
df = pd.read_csv("C:\\Users\\daveb\\Downloads\\heart.csv")  # Read the CSV file containing heart disease data
df.head(14)  # Display the first 14 rows of the dataset

# Separate the label (target) from the features
label = df['target']  # 'target' is the label we are trying to predict (1: Disease, 0: No Disease)
df = df.drop(['target'], axis=1)  # Drop the 'target' column from the dataset as it will be used as labels

# Define the perceptron class
class ptron:
    # Constructor to initialize the perceptron
    def __init__(self, threshold, no_inputs, learning_rate):
        # Initialize weights randomly with size equal to the number of input features
        self.weights = np.random.randn(no_inputs - 1)  
        self.threshold = threshold  # Number of iterations (epochs) to train
        self.learning_rate = learning_rate  # Learning rate for weight updates
        print(self.weights)  # Print initial weights for reference
    
    # Prediction function using the sign of the dot product
    def prd(self, inputs):
        z = np.dot(inputs, self.weights)  # Compute the dot product of inputs and weights
        
        if z > 0:  # If the result is positive, predict 1
            return 1
        else:  # If the result is negative or zero, predict -1
            return -1
    
    # Training function using the perceptron learning rule
    def trn(self, inputs, labels):
        # Loop over the training process for a given number of epochs (threshold)
        for j in range(self.threshold):
            # Iterate through all samples in the dataset
            for i in range(len(labels)):
                # Calculate the weight update based on the prediction error and input
                delta_w = self.learning_rate * (labels[i] - self.prd(inputs[i])) * inputs[i]
                # Update the weights
                self.weights = self.weights + delta_w
        print('Training Done')  # Print when training is complete

# Create an instance of the perceptron with 1000 epochs, 14 inputs, and a learning rate of 0.05
c = ptron(1000, 14, 0.05)

# Train the perceptron using the dataset (features) and the corresponding labels (target)
c.trn(df.values, label)
