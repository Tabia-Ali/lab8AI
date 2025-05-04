from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

# Load dataset
df = pd.read_csv('social_network_ads.csv')

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

@app.route('/')
def index():
    # Data Inspection
    info = str(df.info())
    head = df.head().to_html(classes='table table-striped')
    
    # Descriptive Statistics
    stats = df.describe().to_html(classes='table table-bordered')

    # Visualizations
    plt.figure(figsize=(5, 4))
    sns.countplot(x='Purchased', data=df)
    plt.title('Distribution of Purchased')
    plt.savefig('static/purchased_hist.png')
    plt.clf()

    plt.figure(figsize=(5, 4))
    sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=df)
    plt.title('Age vs Estimated Salary')
    plt.savefig('static/age_salary_scatter.png')
    plt.clf()

    return render_template('index.html', head=head, stats=stats)

@app.route('/predict')
def predict():
    # Preprocessing
    X = df[['Age', 'EstimatedSalary']]
    y = df['Purchased']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn5 = KNeighborsClassifier(n_neighbors=5)

    knn3.fit(X_train, y_train)
    knn5.fit(X_train, y_train)

    pred3 = knn3.predict(X_test)
    pred5 = knn5.predict(X_test)

    acc3 = accuracy_score(y_test, pred3)
    acc5 = accuracy_score(y_test, pred5)

    report3 = classification_report(y_test, pred3, output_dict=True)
    report5 = classification_report(y_test, pred5, output_dict=True)

    return render_template('prediction.html',
                           acc3=round(acc3*100, 2),
                           acc5=round(acc5*100, 2),
                           matrix=confusion_matrix(y_test, pred5))

if __name__ == '__main__':
    app.run(debug=True)
