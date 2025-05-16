
# Churn Prediction for Hotel Reservations

## 1. Problem Definition
The objective of this project was to predict booking cancellations (churn) in a hotel reservation system. Accurately identifying at-risk bookings enables hotel managers to take proactive measures—such as offering discounts, sending reminders, or adjusting pricing strategies. By reducing cancellations, the hotel can improve customer retention, optimize resources, and increase overall revenue.

## 2. Data Exploration
The dataset used for this project consists of 36,275 records, each with 19 features. These features include both numerical and categorical data:
- **Numerical Features**: Number of adults, children, weekend nights, week nights, special requests, lead time, previous bookings, and more.
- **Categorical Features**: Market segment, meal plan, room type, and booking status (canceled or not).

### Data Cleaning and Preprocessing:
- **Duplicates**: 10,275 duplicate records were removed.
- **Missing Values**: No missing values remained after preprocessing.
- **Feature Encoding**: Categorical features were encoded using Label Encoding and One-Hot Encoding to prepare the data for machine learning models.
- **Feature Engineering**: New features such as `total_people`, `stay_length`, and `total_price` were created based on domain knowledge to better capture guest behavior and preferences.

### Key Insights from EDA:
- **Imbalanced Data**: The dataset exhibited a class imbalance, with more non-canceled bookings. This imbalance could lead to biased model predictions.
- **Feature Transformation**: Some features, such as `no_of_previous_cancellations` and `repeated_guest`, showed skewness. Log and square root transformations were applied to normalize the data and ensure that models performed optimally.

### Correlation and Seasonal Analysis:
- **Cancellation Patterns**: Cancellations were highly correlated with features like `lead_time`, `no_of_previous_cancellations`, and `no_of_previous_bookings_not_canceled`.
- **Seasonality**: Cancellations were more frequent during certain months, indicating seasonal booking patterns. This trend can inform better planning for high-risk cancellation periods.

## 3. Model Development
A variety of machine learning models were evaluated to predict cancellations:
- **Model Selection**: Logistic Regression, Random Forest, Extra Trees, Decision Tree, XGBoost, K-Nearest Neighbors, Gradient Boosting, AdaBoost, and Gaussian Naive Bayes.
- **Performance Metrics**: Models were assessed based on accuracy, F1-score, ROC AUC, and training time.

### Data Preprocessing:
- **Feature Scaling**: Features were standardized using MinMaxScaler to ensure proper scaling, especially for models like Logistic Regression and KNN.
- **Handling Data Imbalance**: **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to balance the dataset and avoid biased predictions toward the majority class.

### Feature Engineering:
- Derived features such as `total_people`, `stay_length`, and `total_price` provided the models with additional meaningful information, enhancing their predictive power.

## 4. Model Evaluation and Results
The following table summarizes the performance of the different models:

| **Model**                    | **Train Accuracy** | **Test Accuracy** | **Train F1** | **Test F1** | **Train ROC AUC** | **Test ROC AUC** | **Time (seconds)** |
|------------------------------|--------------------|-------------------|--------------|-------------|-------------------|------------------|-------------------|
| **RandomForestClassifier**    | 0.994151           | 0.973398          | 0.994155     | 0.979990    | 0.999520          | 0.971079         | 22.35             |
| **ExtraTreesClassifier**      | 0.994157           | 0.971744          | 0.994152     | 0.978657    | 0.999909          | 0.971277         | 23.05             |
| **DecisionTreeClassifier**    | 0.994157           | 0.968573          | 0.994152     | 0.976304    | 0.999909          | 0.966840         | 1.02              |
| **BaggingClassifier**         | 0.987040           | 0.963611          | 0.987002     | 0.972466    | 0.999015          | 0.963011         | 5.70              |
| **XGBClassifier**             | 0.904994           | 0.890283          | 0.906239     | 0.916997    | 0.972122          | 0.880493         | 1.37              |
| **KNeighborsClassifier**      | 0.900670           | 0.883253          | 0.898486     | 0.908856    | 0.971865          | 0.887870         | 22.91             |
| **GradientBoostingClassifier**| 0.830745           | 0.825913          | 0.833194     | 0.863974    | 0.916698          | 0.823836         | 19.83             |
| **AdaBoostClassifier**        | 0.775568           | 0.766919          | 0.775153     | 0.814115    | 0.862117          | 0.767203         | 5.52              |
| **LogisticRegression**        | 0.770795           | 0.755755          | 0.766196     | 0.801301    | 0.854640          | 0.763862         | 52.79             |
| **GaussianNB**                | 0.551622           | 0.410338          | 0.229609     | 0.229467    | 0.775323          | 0.548787         | 0.27              |

### Best Models:
- **RandomForestClassifier** and **ExtraTreesClassifier** delivered the best performance across both training and test data, excelling in accuracy, F1-score, and ROC AUC.
- **XGBClassifier** also performed well but was less efficient compared to Random Forest and Extra Trees models.
- **GaussianNB** had the lowest performance across all metrics, indicating it was not well-suited for this particular dataset.

## 5. Model Deployment
The **RandomForestClassifier** was chosen as the final model for deployment due to its outstanding performance. The model was saved using **joblib** for future use in production environments.

- **Model Saving**: The trained RandomForest model was saved as `'random_forest_model.pkl'`.
- **Deployment**: The model is deployed using **Streamlit**, providing a simple and interactive web-based application for real-time prediction of booking cancellations. Streamlit allows hotel managers to input customer details and instantly receive predictions on the likelihood of cancellation, enabling quick and informed decision-making.

## 6. Business Implications
- **Revenue Management**: By predicting cancellations, the hotel can adjust pricing strategies, offer targeted discounts, or provide incentives to customers at risk of canceling, thus improving revenue.
- **Resource Optimization**: By knowing which bookings are likely to be canceled, the hotel can optimize room availability and staffing levels, reducing the risk of overbooking or underutilization of resources.
- **Customer Retention**: The model helps identify at-risk customers, enabling hotel managers to engage them proactively through tailored communication and personalized offers.

## 7. Key Insights and Challenges
### Insights:
- **Lead Time**: Longer lead times correlate with a higher likelihood of cancellation, suggesting that guests with early bookings are more likely to change their plans as the arrival date approaches.
- **Previous Cancellations**: Customers with a history of cancellations are more likely to cancel again, making this a critical feature for predicting future cancellations.
- **Meal Plans and Parking**: Customers who select specific meal plans or request car parking are less likely to cancel, offering a potential area to target these customers for retention efforts.

### Challenges:
- **Data Imbalance**: The dataset’s class imbalance, with more non-canceled bookings, presented a challenge for training. SMOTE was applied to balance the dataset and mitigate this issue.
- **Feature Engineering**: Some features exhibited skewness, which was addressed using log and square root transformations to improve model performance.

## 8. Business Recommendations

### 1. Optimize Seasonal Pricing & Policies:
- **Summer**:
  - Reduce the free cancellation window to minimize last-minute cancellations.
  - Promote non-refundable bookings to ensure revenue stability.
  - Slight overbooking can help offset cancellations, especially during peak seasons.
- **Winter**:
  - Slightly increase prices due to steady demand.
  - Allow more flexibility with cancellation policies to attract more customers during this slower period.

### 2. Targeted Offers:
- Encourage early bookings by offering discounted but non-refundable options to lock in revenue.
- Market to couples (honeymoon packages, romantic getaways) as they are a major segment of the customer base.
- Incentivize repeat customers with loyalty benefits to foster long-term relationships and reduce cancellations.

### 3. Operational Adjustments:
- Prepare for higher occupancy in Q3 & Q4 by optimizing staff and resource allocation.
- Strengthen customer relationship management (CRM) for online and corporate segments, which have stable cancellation rates.
- Monitor weekend bookings closely, as they tend to have a higher chance of cancellation compared to weekday bookings.

## 9. Conclusion
This churn prediction model offers actionable insights to help hotel managers proactively manage cancellations, optimize revenue, and improve customer retention. The **RandomForestClassifier** and **ExtraTreesClassifier** showed the best performance and are recommended for deployment. By integrating this model into the hotel’s booking system via **Streamlit**, managers can make informed decisions to improve customer experience and optimize operational efficiency.

The next steps include continuously monitoring the model's performance in production, updating it with new data, and fine-tuning it to further enhance its predictive power.

---

# Hotel Booking Status Predictor

[Visit the Hotel Booking Status Predictor](https://georgtawadrous-hotelbookingstatuspredictor.hf.space/)

This is a link to a web application hosted on Hugging Face Spaces, created by Georg Tawardrous. The application is designed to predict hotel booking statuses, likely using machine learning or data analysis techniques. For more details, you can explore the application directly via the provided link.

# Hotel Booking Status Predictor - GitHub Repository

[Visit the Hotel Booking Status Predictor GitHub Repository](https://github.com/the3miaphysite3engineer3/HotelBookingStatusPredictor)

This is a link to the GitHub repository for the Hotel Booking Status Predictor, maintained by the3miaphysite3engineer3. The repository likely contains the source code, documentation, and related files for the hotel booking status prediction application. For more details, you can explore the repository directly via the provided link.

### **Project Contributors:**
This project was developed by the following team members:
- **Mohammed Ahmed Mohammed Abdul-Aziz**  
  Email: [mohammed@example.com](mailto:mohammed@example.com)
  
- **Nada Nasser Ragab**  
  Email: [nadanasssrnasser309@gmail.com](mailto:nadanasssrnasser309@gmail.com)

- **Naira Mohamed Abdelbasset**  
  Email: [eng.naira2311@gmail.com](mailto:eng.naira2311@gmail.com)

- **Abdallah Ahmed Mostafa**  
  Email: [bedo.ahmedusa2001@gmail.com](mailto:bedo.ahmedusa2001@gmail.com)

- **George Joseph Basilious Tawadrous**  
  Email: [georgejoseph5000@gmail.com](mailto:georgejoseph5000@gmail.com)

- **Saif El-Din Mohammad Moheb**  
  Email: [seifmoh495@gmail.com](mailto:seifmoh495@gmail.com)

---

