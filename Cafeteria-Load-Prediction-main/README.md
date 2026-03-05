# 🍽️ Cafeteria Lunch Load Prediction using Machine Learning

## 📌 Project Overview
This project demonstrates a **machine learning–based approach to predict cafeteria lunch crowd load** based on environmental and contextual factors such as temperature, rainfall, weekday/weekend, and humidity.

A **Linear Regression model** is trained on synthetically generated data to estimate how these factors influence cafeteria demand during lunchtime. The project is simple, interpretable, and ideal for understanding regression-based prediction workflows.

---

## 🎯 Objectives
- Simulate realistic cafeteria crowd behavior using environmental variables  
- Train and evaluate a regression model for load prediction  
- Visualize the relationship between temperature and lunch load  
- Provide a reusable prediction function for real-time inputs  

---

## 🧠 Features Used
| Feature | Description |
|------|-----------|
| `temperature_c` | Ambient temperature in Celsius |
| `is_rainy` | Rain indicator (0 = No, 1 = Yes) |
| `is_weekday` | Day type (0 = Weekend, 1 = Weekday) |
| `humidity` | Relative humidity (%) |

---

## 📊 Target Variable
- **`lunch_load`** → Estimated number of people in the cafeteria during lunch hours

---

## ⚙️ Tech Stack
- **Python**
- **NumPy** – Data generation  
- **Pandas** – Data handling  
- **Matplotlib** – Data visualization  
- **Scikit-learn** – Machine learning model  

---

## 🧪 Methodology
1. Generate synthetic environmental data  
2. Define cafeteria load logic with realistic assumptions  
3. Perform exploratory visualization  
4. Split data into training and testing sets  
5. Train a Linear Regression model  
6. Evaluate performance using **R² score**  
7. Predict lunch load for new inputs  

---

## 📈 Sample Visualization
The project includes a scatter plot showing:

**Temperature vs Cafeteria Lunch Load**

This helps visualize how increasing temperature affects cafeteria crowd size.

---

## 🔍 Model Performance
- **Evaluation Metric:** R² Score  
- The model achieves a strong R² value, indicating good predictive capability on unseen data.

---

## 🔮 Prediction Function
You can predict cafeteria load using:

```python
predict_load(temperature, is_rainy, is_weekday, humidity)
