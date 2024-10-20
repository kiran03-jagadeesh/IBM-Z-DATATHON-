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
