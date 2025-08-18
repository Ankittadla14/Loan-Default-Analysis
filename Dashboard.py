# Loading all necessary libraries
import sqlite3
import pandas as pd
from dash import dcc, html, Input, Output, dash_table
import dash
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# Loading the dataset
file_path = "LoanDefault_Cleaned.csv"
data = pd.read_csv(file_path)

# Preparing and training the logistic regression model
features = ['Income', 'LoanAmount', 'CreditScore', 'DTIRatio']
X = data[features].dropna()
y = data['Default'].dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

# Metrics
classification_rep = classification_report(y_test, y_pred, output_dict=False)
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Custom CSS styles
external_stylesheets = ['https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap']

# Initializing the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Loan Default Analysis Dashboard"

# Color scheme
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'background': '#ffffff',
    'card_bg': '#ffffff',
    'border': '#e9ecef'
}

# Custom styles
card_style = {
    'backgroundColor': colors['card_bg'],
    'padding': '20px',
    'borderRadius': '12px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'border': f'1px solid {colors["border"]}',
    'margin': '10px'
}

header_style = {
    'backgroundColor': colors['primary'],
    'padding': '30px',
    'marginBottom': '30px',
    'borderRadius': '0 0 20px 20px',
    'boxShadow': '0 4px 20px rgba(31, 119, 180, 0.3)'
}

filter_style = {
    'backgroundColor': colors['light'],
    'padding': '25px',
    'borderRadius': '12px',
    'margin': '10px 0',
    'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.05)',
    'border': f'1px solid {colors["border"]}'
}

app.layout = html.Div([
    # Header Section
    html.Div([
        html.H1("🏦 Loan Default Analysis Dashboard", 
                style={
                    'textAlign': 'center',
                    'color': 'white',
                    'fontSize': '2.5rem',
                    'fontWeight': '700',
                    'margin': '0',
                    'fontFamily': 'Inter, sans-serif'
                }),
        html.P("Team: Ankit, Shilpa, Vivek | Advanced Analytics & Risk Assessment",
               style={
                   'textAlign': 'center',
                   'color': 'rgba(255, 255, 255, 0.9)',
                   'fontSize': '1.1rem',
                   'margin': '10px 0 0 0',
                   'fontFamily': 'Inter, sans-serif'
               })
    ], style=header_style),

    # Filters Section
    html.Div([
        html.H3("🎛️ Filter Controls", 
                style={
                    'color': colors['dark'],
                    'fontSize': '1.5rem',
                    'fontWeight': '600',
                    'marginBottom': '20px',
                    'fontFamily': 'Inter, sans-serif'
                }),
        
        html.Div([
            # Left Column Filters
            html.Div([
                html.Div([
                    html.Label("📚 Education Level:", 
                              style={'fontWeight': '600', 'color': colors['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='education-dropdown',
                        options=[{'label': i, 'value': i} for i in data['Education'].dropna().unique()],
                        placeholder="Select Education Level",
                        style={'marginBottom': '15px'}
                    )
                ]),
                
                html.Div([
                    html.Label("💼 Employment Type:", 
                              style={'fontWeight': '600', 'color': colors['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='employment-dropdown',
                        options=[{'label': i, 'value': i} for i in data['EmploymentType'].dropna().unique()],
                        placeholder="Select Employment Type",
                        style={'marginBottom': '15px'}
                    )
                ]),
                
                html.Div([
                    html.Label("💑 Marital Status:", 
                              style={'fontWeight': '600', 'color': colors['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='marital-dropdown',
                        options=[{'label': i, 'value': i} for i in data['MaritalStatus'].dropna().unique()],
                        placeholder="Select Marital Status",
                        style={'marginBottom': '15px'}
                    )
                ]),
                
                html.Div([
                    html.Label("🏠 Has Mortgage:", 
                              style={'fontWeight': '600', 'color': colors['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='mortgage-dropdown',
                        options=[{'label': i, 'value': i} for i in data['HasMortgage'].dropna().unique()],
                        placeholder="Select Mortgage Status",
                        style={'marginBottom': '15px'}
                    )
                ]),
                
                html.Div([
                    html.Label("👨‍👩‍👧‍👦 Has Dependents:", 
                              style={'fontWeight': '600', 'color': colors['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='dependents-dropdown',
                        options=[{'label': i, 'value': i} for i in data['HasDependents'].dropna().unique()],
                        placeholder="Select Dependents Status",
                        style={'marginBottom': '15px'}
                    )
                ])
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),

            # Right Column Filters
            html.Div([
                html.Div([
                    html.Label("🎯 Loan Purpose:", 
                              style={'fontWeight': '600', 'color': colors['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='loanpurpose-dropdown',
                        options=[{'label': i, 'value': i} for i in data['LoanPurpose'].dropna().unique()],
                        placeholder="Select Loan Purpose",
                        style={'marginBottom': '15px'}
                    )
                ]),
                
                html.Div([
                    html.Label("🤝 Has Co-Signer:", 
                              style={'fontWeight': '600', 'color': colors['dark'], 'marginBottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='cosigner-dropdown',
                        options=[{'label': i, 'value': i} for i in data['HasCoSigner'].dropna().unique()],
                        placeholder="Select Co-Signer Status",
                        style={'marginBottom': '20px'}
                    )
                ]),
                
                html.Div([
                    html.Label("🎂 Age Range:", 
                              style={'fontWeight': '600', 'color': colors['dark'], 'marginBottom': '15px', 'display': 'block'}),
                    dcc.RangeSlider(
                        id='age-slider',
                        min=data['Age'].min(),
                        max=data['Age'].max(),
                        step=1,
                        value=[data['Age'].min(), data['Age'].max()],
                        marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(int(data['Age'].min()), int(data['Age'].max()) + 1, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': '25px'}),
                
                html.Div([
                    html.Label("📊 Credit Score Range:", 
                              style={'fontWeight': '600', 'color': colors['dark'], 'marginBottom': '15px', 'display': 'block'}),
                    dcc.RangeSlider(
                        id='credit-slider',
                        min=data['CreditScore'].min(),
                        max=data['CreditScore'].max(),
                        step=10,
                        value=[data['CreditScore'].min(), data['CreditScore'].max()],
                        marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(int(data['CreditScore'].min()), int(data['CreditScore'].max()) + 1, 100)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
        ])
    ], style=filter_style),

    # Charts Section
    html.Div([
        html.H3("📈 Exploratory Data Analysis", 
                style={
                    'color': colors['dark'],
                    'fontSize': '1.5rem',
                    'fontWeight': '600',
                    'marginBottom': '20px',
                    'fontFamily': 'Inter, sans-serif'
                }),
        
        html.Div([
            html.Div([
                dcc.Graph(id='income-default-graph')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='loanamount-default-graph')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.Div([
            html.Div([
                dcc.Graph(id='income-loan-scatter')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='creditscore-loan-scatter')
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ], style=card_style),

    # Statistical Tests Section
    html.Div([
        html.H3("🧮 Statistical Analysis", 
                style={
                    'color': colors['dark'],
                    'fontSize': '1.5rem',
                    'fontWeight': '600',
                    'marginBottom': '20px',
                    'fontFamily': 'Inter, sans-serif'
                }),
        
        html.Div([
            html.Div([
                html.Div(id='t-test-results', style={'padding': '15px', 'backgroundColor': colors['light'], 'borderRadius': '8px', 'marginBottom': '10px'})
            ], style={'width': '50%', 'display': 'inline-block', 'paddingRight': '1%'}),
            
            html.Div([
                html.Div(id='chi-square-results', style={'padding': '15px', 'backgroundColor': colors['light'], 'borderRadius': '8px', 'marginBottom': '10px'})
            ], style={'width': '50%', 'display': 'inline-block', 'paddingLeft': '1%'})
        ])
    ], style=card_style),

    # Machine Learning Section
    html.Div([
        html.H3("🤖 Machine Learning Analysis", 
                style={
                    'color': colors['dark'],
                    'fontSize': '1.5rem',
                    'fontWeight': '600',
                    'marginBottom': '20px',
                    'fontFamily': 'Inter, sans-serif'
                }),
        
        html.Div(id="classification-report"),
        
        html.Div([
            html.Div([
                dcc.Graph(id='roc-curve')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='confusion-matrix')
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ], style=card_style),

    # Footer
    html.Div([
        html.P("© 2024 Loan Default Analysis Dashboard | Built with Dash & Plotly",
               style={
                   'textAlign': 'center',
                   'color': colors['dark'],
                   'fontSize': '0.9rem',
                   'margin': '20px 0',
                   'opacity': '0.7'
               })
    ])
], style={
    'fontFamily': 'Inter, sans-serif',
    'backgroundColor': colors['light'],
    'minHeight': '100vh',
    'padding': '0'
})

@app.callback(
    [Output('income-default-graph', 'figure'),
     Output('loanamount-default-graph', 'figure'),
     Output("income-loan-scatter", "figure"),
     Output("creditscore-loan-scatter", "figure"),
     Output('t-test-results', 'children'),
     Output('chi-square-results', 'children'),
     Output('classification-report', 'children'),
     Output('roc-curve', 'figure'),
     Output('confusion-matrix', 'figure')],
    [Input('education-dropdown', 'value'),
     Input('employment-dropdown', 'value'),
     Input('marital-dropdown', 'value'),
     Input('mortgage-dropdown', 'value'),
     Input('dependents-dropdown', 'value'),
     Input('loanpurpose-dropdown', 'value'),
     Input('cosigner-dropdown', 'value'),
     Input('age-slider', 'value'),
     Input('credit-slider', 'value')]
)
def update_graphs(education, employment, marital, mortgage, dependents, loanpurpose, cosigner, age_range, credit_range):
    # SQL query building logic (same as original)
    query = '''
    SELECT Income, LoanAmount, CreditScore, Age, "Default", LoanPurpose
    FROM LoanData
    WHERE 1=1
    '''
    filters = []
    if education:
        education = education.replace("'", "''")
        filters.append(f"Education = '{education}'")
    if employment:
        employment = employment.replace("'", "''")
        filters.append(f"EmploymentType = '{employment}'")
    if marital:
        marital = marital.replace("'", "''")
        filters.append(f"MaritalStatus = '{marital}'")
    if mortgage:
        mortgage = mortgage.replace("'", "''")
        filters.append(f"HasMortgage = '{mortgage}'")
    if dependents:
        dependents = dependents.replace("'", "''")
        filters.append(f"HasDependents = '{dependents}'")
    if loanpurpose:
        loanpurpose = loanpurpose.replace("'", "''")
        filters.append(f"LoanPurpose = '{loanpurpose}'")
    if cosigner:
        cosigner = cosigner.replace("'", "''")
        filters.append(f"HasCoSigner = '{cosigner}'")
    if age_range:
        filters.append(f"Age BETWEEN {age_range[0]} AND {age_range[1]}")
    if credit_range:
        filters.append(f"CreditScore BETWEEN {credit_range[0]} AND {credit_range[1]}")
    
    if filters:
        query += " AND " + " AND ".join(filters)

    # Database connection and query execution
    conn = sqlite3.connect(":memory:")
    data.to_sql("LoanData", conn, index=False, if_exists="replace")

    try:
        filtered_data = pd.read_sql_query(query, conn)
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, str(e), str(e), str(e), dash.no_update, dash.no_update
    
    if filtered_data.empty:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "No data", "No data", "No data", dash.no_update, dash.no_update

    # Custom theme for plots
    template = "plotly_white"
    color_palette = [colors['primary'], colors['secondary'], colors['success'], colors['danger']]

    # Income Distribution Box Plot
    income_fig = px.box(filtered_data, x="Default", y="Income", color="Default", 
                       title="💰 Income Distribution by Default Status",
                       color_discrete_sequence=color_palette)
    income_fig.update_layout(
        template=template,
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'family': "Inter", 'weight': 'bold'}},
        font={'family': 'Inter'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Loan Amount Distribution Box Plot
    loanamount_fig = px.box(filtered_data, x="Default", y="LoanAmount", color="Default", 
                           title="💳 Loan Amount Distribution by Default Status",
                           color_discrete_sequence=color_palette)
    loanamount_fig.update_layout(
        template=template,
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'family': "Inter", 'weight': 'bold'}},
        font={'family': 'Inter'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Income vs Loan Amount Scatter Plot
    income_loan_fig = px.scatter(filtered_data, x="Income", y="LoanAmount", color="Default", 
                                title="💰 Income vs Loan Amount Relationship",
                                color_discrete_sequence=color_palette)
    income_loan_fig.update_layout(
        template=template,
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'family': "Inter", 'weight': 'bold'}},
        font={'family': 'Inter'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Credit Score vs Loan Amount Scatter Plot
    creditscore_loan_fig = px.scatter(filtered_data, x="CreditScore", y="LoanAmount", color="Default", 
                                     title="📊 Credit Score vs Loan Amount Analysis",
                                     color_discrete_sequence=color_palette)
    creditscore_loan_fig.update_layout(
        template=template,
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'family': "Inter", 'weight': 'bold'}},
        font={'family': 'Inter'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Statistical Tests
    if not filtered_data.empty:
        group1 = filtered_data[filtered_data['Default'] == 0]['Income']
        group2 = filtered_data[filtered_data['Default'] == 1]['Income']
        t_stat, p_val_ttest = ttest_ind(group1, group2, nan_policy='omit')
        
        t_test_result = html.Div([
            html.H5("📊 T-Test Results", style={'color': colors['primary'], 'fontWeight': '600', 'marginBottom': '10px'}),
            html.P([
                html.Strong("T-Statistic: ", style={'color': colors['dark']}), 
                html.Span(f"{t_stat:.3f}", style={'color': colors['info'], 'fontWeight': '500'}),
                html.Br(),
                html.Strong("P-Value: ", style={'color': colors['dark']}), 
                html.Span(f"{p_val_ttest:.2e}", style={'color': colors['info'], 'fontWeight': '500'})
            ], style={'margin': '0', 'fontSize': '14px'})
        ])

        contingency = pd.crosstab(filtered_data['LoanPurpose'], filtered_data['Default'])
        chi2, p_val_chi2, _, _ = chi2_contingency(contingency)
        
        chi_square_result = html.Div([
            html.H5("🔍 Chi-Square Test", style={'color': colors['primary'], 'fontWeight': '600', 'marginBottom': '10px'}),
            html.P([
                html.Strong("Chi-Square: ", style={'color': colors['dark']}), 
                html.Span(f"{chi2:.3f}", style={'color': colors['info'], 'fontWeight': '500'}),
                html.Br(),
                html.Strong("P-Value: ", style={'color': colors['dark']}), 
                html.Span(f"{p_val_chi2:.2e}", style={'color': colors['info'], 'fontWeight': '500'})
            ], style={'margin': '0', 'fontSize': '14px'})
        ])
    else:
        t_test_result = html.Div("No data available for T-Test", style={'color': colors['danger']})
        chi_square_result = html.Div("No data available for Chi-Square Test", style={'color': colors['danger']})
    
    # Classification Report
    classification_rep_df = pd.DataFrame.from_dict(classification_report(y_test, y_pred, output_dict=True)).T
    classification_table = dash_table.DataTable(
        data=classification_rep_df.round(3).to_dict('records'),
        columns=[{"name": i, "id": i} for i in classification_rep_df.columns],
        style_table={'overflowX': 'auto', 'margin': '20px 0'},
        style_cell={
            'textAlign': 'center',
            'padding': '12px',
            'fontFamily': 'Inter',
            'fontSize': '14px',
            'border': '1px solid #e9ecef'
        },
        style_header={
            'backgroundColor': colors['primary'],
            'color': 'white',
            'fontWeight': '600',
            'border': '1px solid #e9ecef'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': colors['light']
            }
        ]
    )

    classification_report_output = html.Div([
        html.H4("📋 Model Performance Metrics", 
               style={'color': colors['dark'], 'fontWeight': '600', 'marginBottom': '15px'}),
        classification_table,
        html.P(f"🎯 Overall ROC AUC Score: {roc_auc:.3f}", 
              style={'textAlign': 'center', 'fontSize': '16px', 'fontWeight': '600', 
                    'color': colors['success'], 'marginTop': '15px'})
    ])

    # ROC Curve
    roc_curve_fig = go.Figure()
    roc_curve_fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines', 
        name=f"ROC Curve (AUC = {roc_auc:.3f})",
        line=dict(color=colors['primary'], width=3)
    ))
    roc_curve_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', 
        line=dict(dash='dash', color=colors['danger'], width=2), 
        name='Random Classifier'
    ))
    roc_curve_fig.update_layout(
        template=template,
        title={'text': "📈 ROC Curve Analysis", 'x': 0.5, 'xanchor': 'center', 
               'font': {'size': 16, 'family': "Inter", 'weight': 'bold'}},
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        font={'family': 'Inter'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=0.6, y=0.1)
    )
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted: No Default', 'Predicted: Default'],
        y=['Actual: No Default', 'Actual: Default'],
        colorscale='RdYlBu_r',
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size": 16, "family": "Inter"},
        hoverongaps=False
    ))
    conf_matrix_fig.update_layout(
        template=template,
        title={'text': "🎯 Confusion Matrix", 'x': 0.5, 'xanchor': 'center', 
               'font': {'size': 16, 'family': "Inter", 'weight': 'bold'}},
        font={'family': 'Inter'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return (income_fig, loanamount_fig, income_loan_fig, creditscore_loan_fig, 
            t_test_result, chi_square_result, classification_report_output, 
            roc_curve_fig, conf_matrix_fig)

# Running the app
if __name__ == "__main__":
    app.run(debug=True, port=8051)