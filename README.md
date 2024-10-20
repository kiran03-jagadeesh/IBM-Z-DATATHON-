# Real-Time Adaptive Beam Prediction in V2V Communication Using Machine Learning

## Objective
To perform adaptive beam predictions in Vehicle-to-Vehicle (V2V) Communication using machine learning techniques.

## Introduction
Vehicle-to-vehicle (V2V) communication is essential for enhancing road safety, traffic efficiency, and the overall driving experience by enabling vehicles to exchange critical information. Beam management can be made more efficient, and beam training overhead reduced by using environmental and user equipment (UE) sensing data such as GPS receivers, LiDAR, RGB cameras, and RADAR. This helps vehicles identify optimal beam orientations and improves beam steering for V2V communication.

## Dataset
We used the **DeepSense 6G dataset**, which consists of real-world multi-modal data collected from multiple locations in Arizona, USA. It includes more than 1 million data points and covers over 40 deployment scenarios. The dataset spans various use cases such as vehicle-to-infrastructure, vehicle-to-vehicle, pedestrian, drone communication, fixed wireless, and indoor communication.

### Example Data Format from Scenario 36
```plaintext
abs_index  timestamp          unit1_radar1          unit1_radar2          unit1_radar3          unit1_radar4          unit1_overall-beam
2674       11-46-31.214536    data_9354.mat         data_9273.mat         data_9187.mat         data_9102.mat         162
2675       11-46-31.314452    data_9355.mat         data_9274.mat         data_9188.mat         data_9103.mat         161
```
## Beam Prediction Model

We implemented two machine learning models:
1. **LSTM (Long Short-Term Memory)** - used to predict the optimal beam index by leveraging temporal sequences of vehicle movement.
2. **KNN (K-Nearest Neighbors)** - used to classify beam indices based on proximity in the feature space (e.g., vehicle position, velocity).

### Model Performance:
| Model                | Top-1 Accuracy | Top-5 Accuracy |
| -------------------- | -------------- | -------------- |
| **K-Nearest Neighbors** | 33.73%         | 59.95%         |
| **LSTM Model**        | 35.73%         | 68.95%         |

## Python Code:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Data Loading and Preprocessing
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Extract relevant features
    X = df[['abs_index']].values # Only include numeric 'abs_index' initially
    
    # Add radar data columns with handling for non-string values
    for i in range(1, 5):
        X = np.column_stack((X, df[f'unit1_radar{i}'].apply(lambda x: int(x.split('/')[-1].split('_')[1]) if isinstance(x, str) else 0)))
    
    # Convert timestamp to numeric representation (e.g., total seconds)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H-%M-%S.%f').dt.time
    df['timestamp'] = df['timestamp'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)  # Convert to total seconds

    X = np.column_stack((X, df['timestamp'].values)) # Add the numeric timestamp
    
    y = df['unit1_overall-beam'].values
    return X, y

# Load the data
X, y = load_data('scenario37.csv')


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# KNN Model
def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Train KNN model
knn_model = train_knn(X_train_scaled, y_train)

# Evaluate KNN model
y_pred_knn = knn_model.predict(X_test_scaled)
print("KNN Model Performance:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# LSTM Model
def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(input_shape[1], 1)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Reshape data for LSTM (samples, time steps, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# One-hot encode the target variable
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Data Loading and Preprocessing
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Extract relevant features
    X = df[['abs_index']].values # Only include numeric 'abs_index' initially
    
    # Add radar data columns with handling for non-string values
    for i in range(1, 5):
        X = np.column_stack((X, df[f'unit1_radar{i}'].apply(lambda x: int(x.split('/')[-1].split('_')[1]) if isinstance(x, str) else 0)))
    
    # Convert timestamp to numeric representation (e.g., total seconds)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H-%M-%S.%f').dt.time
    df['timestamp'] = df['timestamp'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)  # Convert to total seconds

    X = np.column_stack((X, df['timestamp'].values)) # Add the numeric timestamp
    
    y = df['unit1_overall-beam'].values
    return X, y

# Load the data
X, y = load_data('scenario37.csv')


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# KNN Model
def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Train KNN model
knn_model = train_knn(X_train_scaled, y_train)

# Evaluate KNN model
y_pred_knn = knn_model.predict(X_test_scaled)
print("KNN Model Performance:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# LSTM Model
def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(input_shape[1], 1)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Reshape data for LSTM (samples, time steps, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


# One-hot encode the target variable
num_classes = int(y.max() + 1)  # Convert num_classes to an integer using int()
y_train_cat = to_categorical(y_train, num_classes=num_classes)  # Specify num_classes in to_categorical
y_test_cat = to_categorical(y_test, num_classes=num_classes)  # Specify num_classes in to_categorical


# Create and train LSTM model
lstm_model = create_lstm_model(X_train_lstm.shape, num_classes)
history = lstm_model.fit(X_train_lstm, y_train_cat, epochs=10, batch_size=32, validation_split=0.2)
# Evaluate LSTM model
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

print("\nLSTM")
y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

print("\nLSTM Model Performance:")
print(confusion_matrix(y_test_classes, y_pred_lstm_classes))
print(classification_report(y_test_classes, y_pred_lstm_classes))

# Visualize training history (if needed)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
```

## Results and Discussion

- **KNN Model**: Classified beam indices based on proximity in the feature space. A scatter plot was generated to compare actual and predicted beam indices, and decision boundaries were displayed to show how the model separates different beam index classes.
- **LSTM Model**: Demonstrated success in predicting beam indices using temporal sequences from radar data.

## Scope of Work
This project uses 6G technology to enhance communication efficiency and reliability in autonomous vehicle environments. The challenge focuses on improving V2V communication by predicting optimal beams using a combination of data from GPS receivers, LiDAR, RGB cameras, RADAR, and traditional wireless communication data.

## Workflow
1. Concept Development and Literature Review.
2. Processing data from RADAR.
3. Processing data from RGB camera images.
4. Creating a combined model using GPS, LiDAR, RGB cameras, and RADAR data.

## References
1. A. Alkhateeb et al., "DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset," IEEE Communications Magazine, 2023.
2. M. Alrabeiah et al., "Millimeter Wave Base Stations with Cameras: Vision-Aided Beam and Blockage Prediction," IEEE VTC 2020.
3. J. Morais et al., "DeepSense-V2V: A Vehicle-to-Vehicle Multi-Modal Sensing, Localization, and Communications Dataset," IEEE Vehicular Technology Conference, 2024.
4. M. Noor-A-Rahim et al., "6G for Vehicle-to-Everything (V2X) Communications: Enabling Technologies, Challenges, and Opportunities," IEEE, 2022.

## Thank You
