import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
studentPerformanceData = pd.read_csv("StudentPerformanceFactors.csv")

# Check the columns
print(studentPerformanceData.columns)

# Define target prediction (y)
y = studentPerformanceData["Exam_Score"]

# Define features (X) by removing the target column
X = studentPerformanceData.drop(columns=["Exam_Score"])

# Check for missing values
print("Missing values:\n", X.isnull().sum())

# Fill any missing values with the mode of the column
X.fillna(X.mode().iloc[0], inplace=True)

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the model
studentPerformanceModel = tf.keras.Sequential([
    layers.Input(shape=(train_X.shape[1],)),  # Taking all features as inputs
    layers.Dense(1024),
    layers.Dropout(0.05),
    layers.BatchNormalization(),
    layers.Dense(512),
    layers.Dropout(0.05),
    layers.BatchNormalization(),
    layers.Dense(256),
    layers.Dropout(0.05),
    layers.BatchNormalization(),
    layers.Dense(128),
    #layers.Dropout(0.05),
    layers.BatchNormalization(),
    layers.Dense(64),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model with Adam optimizer and mean absolute error as the loss function
studentPerformanceModel.compile(
    optimizer="adam",
    loss="mae"
)

# Early stopping to prevent overfitting, restoring the best weights based on validation loss
early_stopping = tf.keras.callbacks.EarlyStopping(
    min_delta=0.0001, 
    patience=70, 
    restore_best_weights=True
)

# Train the model
history = studentPerformanceModel.fit( 
    train_X, train_y, 
    validation_data=(val_X, val_y), 
    batch_size=256, 
    epochs=1000, 
    callbacks=[early_stopping], 
    verbose=1 
)

# Plot the training history for loss and validation loss
history_df = pd.DataFrame(history.history)
history_df.loc[:, ["loss", "val_loss"]].plot()
plt.title("Learning Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

studentPerformanceModel.save('trainedStudentPreformanceModel.keras')

# Make predictions on the validation set
val_predictions = studentPerformanceModel.predict(val_X)

# Print minimum validation loss
print("Min validation loss: {}".format(history_df["val_loss"].min()))

# Print the predictions for the validation set
print("Predictions for validation set: ", val_predictions)

