# Mini-Project
# 🎓 Predicting Student Dropout and Academic Success Using Machine Learning

## 📘 Project Overview
This project focuses on predicting student dropout risk and academic success using machine learning techniques. By analyzing demographic, academic, and economic factors, we aim to help educational institutions identify at-risk students early and take appropriate interventions to improve student outcomes.

---

## 🎯 Objectives
- Predict the likelihood of student dropout
- Classify students based on academic success (High/Medium/Low)
- Identify the most important features influencing dropout and success
- Provide data-driven insights for academic support strategies

---

## 📂 Dataset Description
The dataset contains 35 features across several categories:

- **Demographic**: `Age`, `Gender`, `Marital Status`
- **Academic**: `Application Mode`, `Course`, `Attendance`, `Grades`
- **Economic**: `Parent’s Occupation`, `Unemployment Rate`, `GDP`, `Inflation Rate`
- **Target Variables**:
  - `Dropout`: Yes / No
  - `Academic Success`: High / Medium / Low

---

## ⚙️ Methodology

### 🔧 Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Scaling numerical features

### 📊 Model Development
Applied multiple machine learning models:
- Logistic Regression
- Decision Tree
- Random Forest ✅ *(Best for dropout prediction – 88% accuracy)*
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM) ✅ *(Best for success classification – 82% accuracy)*

### 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🧠 Key Findings
- **Top dropout predictors**: Attendance, Previous Grades, Application Mode
- **Top success predictors**: Course, Economic background, Parent’s Occupation
- Random Forest and SVM outperformed other models in respective tasks

---

## ✅ Recommendations
- Build real-time early warning systems for at-risk students
- Track attendance and performance continuously
- Provide additional support based on economic and academic backgrounds

---

## 🔮 Future Work
- Include psychological and behavioral data
- Apply deep learning for more complex modeling
- Deploy a full web-based prediction dashboard

---

## 📄 License
This project is for academic and educational purposes.

---

## 🤝 Acknowledgments
Thanks to [Your University/Instructor Name] for guidance and support in completing this research.

