#  House Price Prediction

This project predicts house prices using the Ames Housing dataset. I built it as an end-to-end machine learning project to strengthen my understanding of the complete ML workflow—from exploring raw data to deploying a model with Streamlit.

The main goal of this project wasn't just to build an accurate model, but also to organize the code in a clean and reusable way, similar to how real-world machine learning projects are structured.

---

## Project Overview

The project follows a complete machine learning pipeline:

- Understanding and exploring the dataset
- Cleaning and preprocessing the data
- Handling missing values
- Creating meaningful features
- Training multiple machine learning models
- Selecting the best-performing model
- Building a prediction pipeline
- Creating an interactive web application using Streamlit

---

## Dataset

The project uses the **Ames Housing Dataset**, which contains information about nearly **3,000 houses** and more than **80 different features** describing each property.

The target variable is:

**SalePrice**

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

---

## Project Structure

```text
house_price_prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── predict.py
│   └── __init__.py
│
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Machine Learning Workflow

The project follows the following workflow:

```
Raw Dataset
      ↓
Exploratory Data Analysis
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Save Trained Model
      ↓
Prediction Module
      ↓
Streamlit Web Application
```

---

## Model Used

I experimented with multiple regression algorithms during development and selected **Gradient Boosting Regressor** because it provided the best balance between accuracy and generalization on this dataset.

The model is trained using a Scikit-learn Pipeline, which combines preprocessing and prediction into a single reusable object.

---

## Features Included

Some of the important steps implemented in this project are:

- Handling missing values
- Feature engineering
- Feature selection
- Data preprocessing pipeline
- Model training
- Model evaluation
- Saving the trained model
- Predicting prices using new inputs
- Interactive Streamlit application

---

## Running the Project

Clone the repository

```bash
git clone https://github.com/<your-username>/house_price_prediction.git
```

Move into the project directory

```bash
cd house_price_prediction
```

Install the required packages

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Run the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

---

## Sample Application

*(Add screenshots of your Streamlit application here after deployment.)*

Example:

```
images/
    home_page.png
    prediction.png
```

---

## What I Learned

While working on this project, I gained practical experience with:

- Building an end-to-end machine learning pipeline
- Organizing ML projects into reusable modules
- Feature engineering techniques
- Creating prediction pipelines with Scikit-learn
- Deploying machine learning models using Streamlit
- Writing cleaner and more maintainable Python code

---

## Future Improvements

Some improvements I'd like to add in the future are:

- Hyperparameter tuning
- Feature importance visualization
- Model comparison dashboard
- XGBoost and LightGBM models
- Cloud deployment
- Docker support

---

## About Me

I'm currently building projects to strengthen my skills in **Machine Learning**, **Data Science**, and **Python**.

If you'd like to connect, feel free to reach out.

- GitHub: https://github.com/<your-username>
- LinkedIn: https://linkedin.com/in/<your-profile>

---

Thank you for checking out this project!