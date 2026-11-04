# 🏙️ NYC Airbnb Room Type Predictor

An end-to-end Machine Learning project that predicts the room type of Airbnb listings in New York City using listing attributes such as location, price, availability, and review activity.

The project combines **Machine Learning, Scikit-learn, FastAPI, and Streamlit** to deliver an interactive room-type prediction application.

---
## 🚀 Live Demo

- **NYC Airbnb Room Type Predictor – [Try it Live](https://nyc-airbnb-room-type-predictor-ndtrjzhihpqpuj6omhz9zt.streamlit.app/)**
- **Backend API (Render) - [Open API](https://nyc-airbnb-room-type-predictor-qalv.onrender.com/docs)**
---

## 📌 Project Overview

The NYC Airbnb Room Type Predictor is a multi-class classification application designed to predict which type of accommodation an Airbnb listing offers.

Given information about a listing, the trained Random Forest model predicts one of three categories:

- 🏠 **Entire home/apt** – The guest gets the entire property.
- 🛏️ **Private room** – The guest gets a private room within a shared home.
- 🛋️ **Shared room** – The guest shares the sleeping space with others.

The project follows an end-to-end Machine Learning workflow, from data exploration and preprocessing to model selection, hyperparameter tuning, evaluation, and deployment.

## 🎯 Objectives

- Analyze New York City Airbnb listing data.
- Explore patterns and relationships between listing attributes and room types.
- Build and compare multiple classification algorithms.
- Handle class imbalance during model training.
- Optimize the best-performing model using hyperparameter tuning.
- Deploy the trained model through a FastAPI backend.
- Build an interactive Streamlit frontend for real-time predictions.

## 📊 Dataset

**Dataset:** [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

The dataset contains Airbnb listings across five New York City boroughs.

| Attribute | Description |
|---|---|
| Records | 48,895 |
| Original Features | 16 |
| Target Variable | `room_type` |
| Boroughs | 5 |
| Classification Type | Multi-class |

### Target Classes

- Entire home/apt
- Private room
- Shared room

### Input Features

The model uses 10 features:

| Feature | Description |
|---|---|
| latitude | Geographic latitude |
| longitude | Geographic longitude |
| price | Price per night |
| minimum_nights | Minimum nights required |
| number_of_reviews | Total number of reviews |
| reviews_per_month | Monthly review activity |
| calculated_host_listings_count | Number of listings managed by the host |
| availability_365 | Availability throughout the year |
| neighbourhood_group | NYC borough |
| neighbourhood | Specific neighbourhood |

## 🔍 Exploratory Data Analysis (EDA)

The notebook covers:

- Missing value analysis.
- Target variable distribution.
- Univariate analysis of numerical features.
- Categorical feature distribution.
- Price distribution across room types.
- Correlation analysis.
- Geographic distribution of Airbnb listings.
- Outlier inspection.

**Key observation:** The dataset contains imbalanced target classes, with Shared Room representing a small minority. Therefore, accuracy alone is not sufficient for evaluating model performance.

## 🧹 Data Preprocessing & Feature Engineering

The following preprocessing steps were implemented:

1. **Column Removal:** Removed `id`, `name`, `host_id`, `host_name`, and `last_review`.
2. **Missing Value Handling:** Replaced missing `reviews_per_month` values with `0`.
3. **Outlier Treatment:** Applied 99th-percentile clipping to `price` and `minimum_nights`.
4. **Feature-Target Separation:** Separated input features (`X`) and target (`y`).
5. **Train-Test Split:** Used stratified splitting with a 33% test set and `random_state=42`.

### Preprocessing Pipeline

Implemented using Scikit-learn's `ColumnTransformer` and `Pipeline`.

| Feature Type | Preprocessing |
|---|---|
| Numerical | Median Imputation + Standard Scaling |
| Categorical | Most Frequent Imputation + One-Hot Encoding |

`OneHotEncoder(handle_unknown='ignore')` ensures that previously unseen categorical values do not cause encoding errors during prediction.

## 🤖 Model Development & Comparison

Four classification algorithms were evaluated using 3-fold stratified cross-validation.

| Model | CV Accuracy | CV Macro F1 |
|---|---:|---:|
| Logistic Regression | 65.9% | 0.522 |
| Decision Tree | 78.2% | 0.648 |
| Random Forest | 84.9% | 0.734 |
| Gradient Boosting | 85.0% | 0.705 |

**Selected Model: Random Forest Classifier**

Although Gradient Boosting achieved slightly higher accuracy, Random Forest performed better in terms of Macro F1-score, making it a better choice for the imbalanced classification problem.

Class weighting (`class_weight='balanced'`) was used for Logistic Regression, Decision Tree, and Random Forest to give more importance to minority classes.

## ⚙️ Hyperparameter Tuning

RandomizedSearchCV was used to optimize the Random Forest classifier.

**Search Configuration:**
- Cross-validation: 3-fold
- Iterations: 10
- Scoring: Macro F1
- Random State: 42

**Best Hyperparameters:**

| Parameter | Selected Value |
|---|---|
| n_estimators | 100 |
| max_depth | None |
| min_samples_split | 2 |

Best Cross-Validation Macro F1: **0.7321**

## 📈 Final Model Evaluation

The tuned Random Forest pipeline was evaluated on the held-out test dataset.

<details>
<summary><strong>Final Performance Metrics</strong></summary>

| Metric | Score |
|---|---:|
| Test Accuracy | 85.47% |
| Test Macro F1-score | 0.7375 |

</details>

The model achieved approximately **85.47% accuracy** and a **0.7375 Macro F1-score** on unseen test data.

The complete preprocessing and classification pipeline was saved using Joblib as `Model_Pipeline.pkl`.

## 🖥️ Application Features

The Streamlit application provides an interactive interface with the following capabilities:

- **Custom Listing Inputs:** Enter location, price, minimum nights, availability, and review information.
- **Neighbourhood Selection:** Choose from five NYC boroughs and their neighbourhoods.
- **Interactive Map:** Visualize the selected listing location.
- **Real-Time Prediction:** Predict the Airbnb room type with confidence probabilities.
- **Probability Visualization:** Compare the predicted probabilities of all three classes.
- **Example Presets:** Try predefined listing examples from Manhattan, Brooklyn, and Queens.
- **Price Sensitivity Analysis:** Explore how changing nightly prices affects prediction probabilities.
- **Prediction History:** View previous predictions and download the history as a CSV file.
- **API Connectivity:** Switch between FastAPI backend predictions and local model predictions.

## 🏗️ System Architecture

The project follows a decoupled frontend-backend architecture.

```text
             User
              |
              v
      Streamlit Frontend
           (app.py)
              |
              | HTTP POST /predict
              v
        FastAPI Backend
           (main.py)
              |
              v
      Model_Pipeline.pkl
              |
              v
      Random Forest Model
              |
              v
       Prediction Results
              |
              v
     Streamlit Result Display
```

The Streamlit frontend communicates with the FastAPI backend through HTTP requests. The backend loads the serialized ML pipeline and returns prediction probabilities.

A local model fallback is also available if the API becomes unreachable and the model artifact is present.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | NumPy, Pandas |
| Data Visualization | Matplotlib, Seaborn, Altair |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Backend | FastAPI |
| Frontend | Streamlit |
| API Communication | REST API, Requests |
| Interactive Mapping | PyDeck |
| Deployment | Render, Streamlit Community Cloud |
| Dataset Download | KaggleHub |

## 📂 Project Structure

```text
NYC-Airbnb-Room-Type-Predictor/
│
├── app.py
├── main.py
├── nyc_data.py
│
├── Model_Pipeline.pkl
│
├── nyc_airbnb_room_type_classification.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚡ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/NYC-Airbnb-Room-Type-Predictor.git

cd NYC-Airbnb-Room-Type-Predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Backend

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the Streamlit Frontend

Open another terminal and run:

```bash
streamlit run app.py
```

The application will open in your browser.

**Note:** To use local model prediction without starting FastAPI, select `Local model file` in the Streamlit sidebar. Make sure `Model_Pipeline.pkl` is present alongside `app.py`.

## 🔮 Future Enhancements

- Improve minority-class prediction performance.
- Explore additional ensemble learning techniques.
- Add more advanced feature engineering.
- Integrate additional location-based insights.
- Improve model explainability using feature importance and SHAP.
- Add automated model monitoring and retraining.

## 👩‍💻 Author

**Mousumi Badyakar**