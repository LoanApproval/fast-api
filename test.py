import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the data
data = pd.read_csv('loan_data.csv')

# One-hot encode categorical features
data_encoded = pd.get_dummies(data, columns=['person_gender', 'person_education', 'previous_loan_defaults_on_file'], drop_first=True)

# Feature engineering: create new features
data_encoded['loan_to_income'] = data_encoded['loan_amnt'] / data_encoded['person_income']
data_encoded['loan_interest_to_income'] = data_encoded['loan_int_rate'] / data_encoded['person_income']
data_encoded['credit_score_impact'] = data_encoded['credit_score'] / data_encoded['loan_amnt']

# Binning age
bins = [20, 25, 30, 35, 40, 45, 50, 60, 100]
labels = ['20-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-60', '60+']
data_encoded['age_group'] = pd.cut(data_encoded['person_age'], bins=bins, labels=labels, right=False)

# Handle missing values
data_encoded['cb_person_cred_hist_length'].fillna(data_encoded['cb_person_cred_hist_length'].median(), inplace=True)

# Scale numeric features
scaler = StandardScaler()
numeric_columns = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'credit_score']
data_encoded[numeric_columns] = scaler.fit_transform(data_encoded[numeric_columns])

# Split the data into features (X) and target (y)
X = data_encoded.drop('loan_status', axis=1)
y = data_encoded['loan_status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
