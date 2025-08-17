
# 🏦 Loan Default Analysis Dashboard

![Dashboard](https://img.shields.io/badge/Dashboard-Interactive-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Dash](https://img.shields.io/badge/Dash-2.14+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Overview

An interactive web-based dashboard for comprehensive loan default analysis using machine learning and statistical methods. This project provides real-time insights into loan default patterns through dynamic visualizations and predictive modeling.

## ✨ Key Features

### 📊 Interactive Visualizations
- Income distribution analysis by default status
- Loan amount patterns and correlations
- Credit score vs loan amount relationships
- Multi-dimensional scatter plots with filtering

### 🔍 Advanced Filtering System
- Education level filtering
- Employment type selection
- Marital status and demographic filters
- Age and credit score range sliders
- Real-time data updates

### 📈 Statistical Analysis
- T-Test analysis for income comparisons
- Chi-Square tests for categorical relationships
- Comprehensive statistical reporting

### 🤖 Machine Learning Model
- Logistic Regression implementation
- ROC curve analysis with AUC scoring
- Confusion matrix visualization
- Classification performance metrics

## 🛠️ Technologies Used

- **Frontend**: Dash, Plotly, HTML/CSS
- **Backend**: Python, Pandas, SQLite
- **Machine Learning**: Scikit-learn, SciPy
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly Express, Plotly Graph Objects

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/loan-default-dashboard.git
   cd loan-default-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place your `LoanDefault_Cleaned.csv` file in the project directory
   - Or update the file path in the code

4. **Run the application**
   
   **Option 1: Jupyter Notebook**
   ```bash
   jupyter notebook Dashboard_Final.ipynb
   ```
   
   **Option 2: Python Script**
   ```bash
   python dashboard.py
   ```

5. **Access the dashboard**
   - Open your browser
   - Navigate to: `http://127.0.0.1:8051`

## 📊 Dataset Requirements

The dashboard expects a CSV file with the following columns:
- `Income` - Borrower's annual income
- `LoanAmount` - Requested loan amount
- `CreditScore` - Credit score (300-850)
- `DTIRatio` - Debt-to-income ratio
- `Age` - Borrower's age
- `Education` - Education level
- `EmploymentType` - Type of employment
- `MaritalStatus` - Marital status
- `HasMortgage` - Mortgage status (Yes/No)
- `HasDependents` - Dependents status (Yes/No)
- `LoanPurpose` - Purpose of the loan
- `HasCoSigner` - Co-signer status (Yes/No)
- `Default` - Target variable (0/1)

## 📸 Screenshots

### Dashboard Overview
![Dashboard Overview](screenshots/dashboard_overview.png)

### Interactive Filters
![Filter Section](screenshots/filter_section.png)

### Statistical Analysis
![Charts Section](screenshots/charts_section.png)

### Machine Learning Results
![ML Results](screenshots/ml_results.png)

## 🎯 Usage Guide

### Filtering Data
1. Use the dropdown menus to filter by categorical variables
2. Adjust age and credit score ranges using the sliders
3. Charts update automatically based on your selections

### Interpreting Results
- **Box plots** show distribution differences between defaulters and non-defaulters
- **Scatter plots** reveal correlations between numerical variables
- **Statistical tests** provide significance levels for observed differences
- **ML metrics** show model performance and prediction accuracy

## 🔧 Configuration

### Changing Port
If port 8051 is busy, modify the last line in the code:
```python
app.run(debug=True, port=8052)  # Use different port
```

### Data Path
Update the file path in your code:
```python
file_path = "path/to/your/LoanDefault_Cleaned.csv"
```

## 🧪 Model Performance

The logistic regression model provides:
- **Feature Analysis**: Income, Loan Amount, Credit Score, DTI Ratio
- **Performance Metrics**: Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: ROC Curves, Confusion Matrix
- **Statistical Validation**: Cross-validation and significance testing

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use Error:**
```bash
OSError: Address 'http://127.0.0.1:8051' already in use
```
**Solution**: Change the port number in the code or restart your system.

**File Not Found Error:**
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'LoanDefault_Cleaned.csv'
```
**Solution**: Check your file path and ensure the CSV file exists.

**Missing Package Error:**
```bash
ModuleNotFoundError: No module named 'dash'
```
**Solution**: Install missing packages using `pip install -r requirements.txt`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing tools
- Dash and Plotly teams for the visualization framework
- Scikit-learn contributors for machine learning capabilities

## 📞 Contact

For questions or support, please reach out to any team member:
- **Project Repository**: [GitHub Link]
- **Issues**: [GitHub Issues Page]

---

**⭐ If you found this project helpful, please give it a star!**

---

*Built with ❤️ using Python, Dash, and Plotly*
