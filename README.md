# 🏦 Loan Default Prediction & Dashboard

This project analyzes and predicts **loan default risk** using machine learning, and presents results through an **interactive dashboard built with Dash & Plotly**.

---

## 📂 Project Structure
```
loan-default-prediction/
│── loan_prediction.ipynb   # Jupyter Notebook for EDA, preprocessing, ML
│── trial.py                # Dash app for interactive analysis
│── LoanDefault_Cleaned.csv # Dataset (cleaned)
│── LoanDefault_Updated.csv # Dataset (updated, if you use it)
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
│── .gitignore              # Ignore unnecessary files
│
├── models/                 # (optional) saved models (e.g., loan_model.pkl, scaler.pkl)
├── assets/                 # (optional) custom CSS/images for Dash
```

> **Note on dataset path**: In `trial.py` you currently load the dataset using an absolute Windows path.  
> For portability, change it to a **relative path**, for example:
> ```python
> file_path = "LoanDefault_Cleaned.csv"
> data = pd.read_csv(file_path)
> ```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-default-prediction.git
   cd loan-default-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Mac/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Usage

### Run Jupyter Notebook
```bash
jupyter notebook loan_prediction.ipynb
```
This opens the notebook for EDA, preprocessing, and model training.

### Run Dash Dashboard
```bash
python trial.py
```
Then open **http://127.0.0.1:8051/** in your browser.

---

## 📈 Features
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Logistic Regression for default prediction
- Performance metrics: Confusion Matrix, ROC-AUC, Classification Report
- Statistical Tests: T-Test, Chi-Square
- Interactive Dashboard with filters:
  - Education Level, Employment Type, Marital Status
  - Mortgage & Dependents status
  - Loan Purpose, Co-Signer
  - Age & Credit Score ranges

---

## 👨‍💻 Team
- **Ankit Tadla**
- **Shilpa**
- **Vivek**

---

## 📜 License
This project is open-source for learning and research use.
