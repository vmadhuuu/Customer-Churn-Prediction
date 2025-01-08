# **Customer Churn Prediction with Machine Learning and Flask**

This project implements an end-to-end **Customer Churn Prediction System** using machine learning techniques, with a focus on **Logistic Regression** and **XGBoost** models. The solution is deployed via a **Flask web application**, enabling real-time predictions and actionable insights to improve customer retention strategies.

---

## **Features**
- **Machine Learning Models**: Logistic Regression and XGBoost for churn prediction.
- **Data Preprocessing**: EDA, data cleaning, feature engineering, and handling imbalanced datasets using **SMOTE**.
- **Web Application**: Built with Flask, featuring REST API endpoints for real-time churn prediction.
- **Performance Metrics**: 5-fold cross-validation for robust model evaluation and recall improvement by 15%.
- **Automated Pipelines**: Includes preprocessing pipelines for seamless integration.

---

## **Tech Stack**
- **Programming Language**: Python
- **Libraries**: 
  - Data Processing: pandas, NumPy, scikit-learn
  - Machine Learning: XGBoost, scikit-learn
  - Imbalanced Data Handling: imbalanced-learn (SMOTE)
  - Web Framework: Flask
- **Deployment**: REST API endpoints for real-time prediction

---

## **How to Run**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Web Application**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

## **Data Pipeline**
1. **Data Extraction and Cleaning**:
   - Performed EDA to identify and clean missing or inconsistent data.
   - Applied feature engineering to derive useful predictors for churn analysis.

2. **Model Training**:
   - Trained Logistic Regression and XGBoost models with hyperparameter tuning.
   - Improved recall by 15% using SMOTE to balance the dataset.

3. **Web Application**:
   - Integrated the trained models into a Flask web application for real-time predictions.

---

## **Usage**
- Upload customer data through the web application.
- View predictions for customer churn and gain insights into customer retention strategies.

---

## **Screenshots**
_Add screenshots of your Flask application interface, data visualizations, or API results here._

---

## **Results**
- Achieved a **15% increase in recall** through data preprocessing and SMOTE.
- Successfully deployed a user-friendly web interface for **real-time churn prediction**.

---

## **Future Work**
- Add additional machine learning models for comparison.
- Implement advanced visualizations for retention analysis.
- Enhance the Flask interface with more insights and dashboards.

---

## **Contributors**
- [Madhumitha Venkatesan](https://github.com/vmadhuuu)

---

## **License**
This project is licensed under the MIT License.
