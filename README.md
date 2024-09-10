# Real-Estate-Investment-Trust
Real Estate Investment Trust
1. Project Title
Descriptive Title: House Sales Analysis in King County, USA

Description: This project involves analyzing house sales data from King County to understand the factors affecting house prices and develop predictive models using linear regression and Ridge regression.

2. Dataset Overview
Description: This section provides an overview of the dataset used in the project. The dataset includes information about house sales, such as house features, prices, and other relevant attributes.

3. Importing Data
Description: Import the necessary libraries and load the dataset into a pandas DataFrame.

python
Copiar código
import pandas as pd

# Load the dataset
df = pd.read_csv('house_sales.csv')
4. Data Wrangling
4.1 Display Data Types of Each Column
Description: Display the data types of each column using the dtypes attribute and capture a screenshot of this output.

python
Copiar código
# Display the data types of each column
print(df.dtypes)
4.2 Remove Unnecessary Columns
Description: Remove the columns "id" and "Unnamed: 0" from the DataFrame and generate a statistical summary of the data.

python
Copiar código
# Remove unnecessary columns
df.drop(columns=["id", "Unnamed: 0"], inplace=True)

# Get a statistical summary
print(df.describe())
4.3 Count Unique Values of "landvalue"
Description: Count the number of houses with unique land values and convert the result into a DataFrame.

python
Copiar código
# Count unique values of "landvalue"
landvalue_counts = df['landvalue'].value_counts().to_frame()
print(landvalue_counts)
5. Exploratory Data Analysis
5.1 Boxplot to Visualize Outliers
Description: Use Seaborn’s boxplot function to visualize whether houses with or without waterfront views have more outliers in prices.

python
Copiar código
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot
sns.boxplot(x='waterfront', y='price', data=df)
plt.title('Boxplot of Price by Waterfront View')
plt.show()
5.2 Regplot for Correlation with 'sqft_above'
Description: Use Seaborn’s regplot function to examine if 'sqft_above' is positively or negatively correlated with the price.

python
Copiar código
# Regplot
sns.regplot(x='sqft_above', y='price', data=df)
plt.title('Correlation between sqft_above and Price')
plt.show()
6. Model Development
6.1 Linear Regression Model with 'sqft_living'
Description: Fit a linear regression model to predict price using 'sqft_living' and calculate the R².

python
Copiar código
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[['sqft_living']]
Y = df['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lm = LinearRegression()
lm.fit(X_train, Y_train)
R_squared = lm.score(X_test, Y_test)
print(f'R²: {R_squared}')
6.2 Linear Regression Model with Multiple Features
Description: Fit a linear regression model to predict price using multiple features and calculate the R².

python
Copiar código
features = ['floors', 'waterfront', 'lat', 'bedrooms', 'sqft_basement', 'view', 'bathrooms', 'sqft_living15', 'sqft_above', 'grade', 'sqft_living']
X_multi = df[features]

X_train_multi, X_test_multi, Y_train_multi, Y_test_multi = train_test_split(X_multi, Y, test_size=0.2, random_state=42)
lm_multi = LinearRegression()
lm_multi.fit(X_train_multi, Y_train_multi)
R_squared_multi = lm_multi.score(X_test_multi, Y_test_multi)
print(f'R²: {R_squared_multi}')
6.3 Polynomial Transformation Pipeline
Description: Create a pipeline that scales the data, performs polynomial transformation, and fits a linear regression model.

python
Copiar código
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])

pipeline.fit(X_train_multi, Y_train_multi)
R_squared_pipeline = pipeline.score(X_test_multi, Y_test_multi)
print(f'R²: {R_squared_pipeline}')
6.4 Ridge Regression Model
Description: Create and fit a Ridge regression model with a regularization parameter of 0.1 and calculate the R².

python
Copiar código
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train_multi, Y_train_multi)
R_squared_ridge = ridge_model.score(X_test_multi, Y_test_multi)
print(f'R²: {R_squared_ridge}')
6.5 Polynomial Transformation of Second Order
Description: Perform a second-order polynomial transformation on both training and test data, then fit a Ridge regression model and calculate the R².

python
Copiar código
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_multi)
X_test_poly = poly.transform(X_test_multi)

ridge_poly_model = Ridge(alpha=0.1)
ridge_poly_model.fit(X_train_poly, Y_train_multi)
R_squared_poly = ridge_poly_model.score(X_test_poly, Y_test_multi)
print(f'R²: {R_squared_poly}')
7. Conclusion
Description: Summarize the key findings from your analysis, discussing model performance and the importance of selected features.

8. References
Description: Include any sources or references used throughout your project.
