

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc
import numpy as np
import seaborn as sns
import itertools

# Load Dataset
st.title("Decision Tree Classifier for Customer Purchase Prediction")

st.markdown("""
### Objective:
This application uses a Decision Tree Classifier to predict whether a customer will purchase a product or service based on demographic and behavioral data.
""")

@st.cache_data
def load_data():
    return pd.read_csv("bank-additional.csv", sep=';')

data = load_data()
st.write("### Sample Data", data.head())

# Train Model
st.subheader("Training Decision Tree Model")

features = data.drop(columns=['y'])  # Assuming 'y' is the target
labels = data['y']

# Convert categorical features to numeric
features = pd.get_dummies(features)
labels = labels.map({'yes': 1, 'no': 0})

# Train Classifier
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(features, labels)

# Plot Decision Tree
st.subheader("Decision Tree Visualization")
st.write("This visualization represents the decision-making process of the model. Each node splits based on the most influential feature, and the leaf nodes represent the final prediction (Yes or No for purchase).")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(clf, filled=True, feature_names=features.columns, class_names=['No', 'Yes'])
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
st.write("The feature importance graph highlights the most influential factors in predicting customer purchases. Features with higher importance scores have a greater impact on the decision tree's outcome.")
importance = pd.Series(clf.feature_importances_, index=features.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x=importance.values, y=importance.index, ax=ax)
ax.set_title("Feature Importance in Decision Tree")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
st.pyplot(fig)

# Additional Visualizations
st.subheader("Exploratory Data Analysis (EDA)")

# Pairplot
st.write("### Pairplot of Features")
st.write("This pairplot helps visualize relationships between numerical features. The diagonal plots show distributions, while the scatter plots reveal correlations between different variables.")
fig = sns.pairplot(data, hue='y', diag_kind='kde')
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
st.write("The ROC curve illustrates the performance of the Decision Tree classifier in distinguishing between customers who will and will not purchase. The area under the curve (AUC) provides a measure of accuracy—closer to 1 means better performance.")
y_scores = clf.predict_proba(features)[:, 1]
fpr, tpr, _ = roc_curve(labels, y_scores)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# Histogram of Features
st.write("### Histogram of Features")
st.write("The histogram visualizes the distribution of numerical features, showing how values are spread out. This helps in identifying trends and patterns in customer behavior.")
fig, ax = plt.subplots(figsize=(10, 10))
data.hist(ax=ax, color='#00FFFF')
st.pyplot(fig)

# Correlation Heatmap
st.write("### Correlation Heatmap")
st.write("The correlation heatmap provides insights into relationships between numerical features. A strong correlation suggests that two features have a significant impact on each other and may influence the model’s performance.")
numeric_df = data.select_dtypes(include=['number'])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='Set3', linewidths=0.2, ax=ax)
st.pyplot(fig)

# Model Insights and Conclusion
st.subheader("Insights and Conclusion")
st.write("""
### Key Insights:
- The Decision Tree model effectively classifies whether a customer will purchase a product based on demographic and behavioral factors.
- Feature importance analysis reveals which variables significantly impact decision-making. For instance, age, job type, and previous interactions may have a major influence.
- The correlation heatmap highlights relationships between different features, helping in feature selection and engineering.
- The ROC curve provides a measure of model performance. An AUC closer to 1 suggests better classification ability.
- The histogram and pairplot further help in understanding the distribution of data and possible patterns.

### Conclusion:
- The Decision Tree Classifier is an intuitive and interpretable model for predicting customer purchases.
- Businesses can use the feature importance insights to focus on the most influential attributes when designing marketing strategies.
- Understanding correlations and feature distributions enables better data preprocessing and model optimization.
- While the current model performs well, improvements can be made by fine-tuning hyperparameters, using ensemble models like Random Forest, or incorporating additional behavioral data.
- The insights gained from this model can help companies refine their sales strategies, target the right customers, and enhance overall business performance.

This study provides a structured approach to analyzing customer purchasing behavior, offering valuable takeaways for business decision-making.
""")

st.success("Model training, visualization, and analysis completed!")
