# Customer Segmentation Analysis

## 📌 Project Overview

This project performs customer segmentation using behavioral and demographic data.

The objective is to identify distinct customer groups based on purchasing behavior, spending patterns, engagement, and customer characteristics, and then develop actionable business strategies for each segment.

## 🎯 Objectives

- Analyze customer demographics and purchasing behavior
- Identify meaningful customer segments
- Apply K-Means clustering using Python
- Determine the optimal number of clusters using the Elbow Method
- Visualize customer segments
- Generate actionable business recommendations

## 🛠️ Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- K-Means Clustering

## 📊 Dataset

The dataset contains 500 customer records with 15 attributes covering:

- Demographics
- Annual income
- Customer tenure
- Number of orders
- Average order value
- Total spending
- Recency
- Website visits
- App usage
- Discount usage
- Satisfaction score

## 🔍 Methodology

### 1. Data Preparation

The dataset was loaded using Pandas and checked for:

- Missing values
- Duplicate records
- Data types
- Basic statistical characteristics

### 2. Exploratory Data Analysis

Customer behavior was explored using:

- Distribution charts
- Correlation analysis
- Spending analysis
- Income vs. spending analysis
- Orders vs. spending analysis

### 3. Feature Selection

The following behavioral variables were selected for clustering:

- Annual Income
- Orders
- Average Order Value
- Total Spend
- Recency
- Website Visits
- App Usage
- Discount Usage

### 4. Feature Scaling

StandardScaler was used to standardize the clustering variables before applying K-Means.

### 5. Customer Segmentation

K-Means clustering was used to divide customers into four behavioral segments.

The Elbow Method was used to evaluate the appropriate number of clusters.

## 👥 Customer Segments

### 💎 Premium High-Value Customers

Customers with the highest income, order frequency, and total spending.

**Strategy:**
- VIP rewards
- Premium products
- Personalized recommendations
- Exclusive offers

### ⭐ Active Loyal Customers

Customers with strong purchasing activity and relatively recent engagement.

**Strategy:**
- Loyalty programs
- Cross-selling
- Personalized recommendations
- Repeat-purchase incentives

### 🔄 High-Value At-Risk Customers

Customers with relatively strong spending but longer periods since their last activity.

**Strategy:**
- Re-engagement campaigns
- Personalized offers
- Limited-time incentives
- Reminder campaigns

### 🛍️ Low-Value Customers

Customers with comparatively lower average order values and total spending.

**Strategy:**
- Product bundles
- Targeted promotions
- Upselling
- Personalized recommendations

## 💡 Key Business Insights

The analysis identifies four distinct customer groups with different spending and engagement behaviors.

Premium customers represent an important high-value segment, while active loyal customers show strong recent engagement.

The high-value at-risk segment presents an opportunity for customer retention and re-engagement.

The low-value segment represents an opportunity to increase customer value through targeted promotions and upselling.

## 📈 Visualizations

The project includes visualizations covering:

- Numerical distributions
- Customer spending
- Correlation analysis
- Income vs. spending
- Orders vs. spending
- Elbow Method
- Customer segment distribution
- Segment spending
- Segment behavioral comparison

## 📁 Project Structure

customer-segmentation-analysis/

├── data/

│   ├── customer_segmentation_dataset.csv

│   ├── customer_segments.csv

│   ├── customer_segment_profiles.csv

│   └── customer_segments_final.csv

├── notebooks/

│   └── customer_segmentation.py

├── visuals/

│   ├── numerical_distributions.png

│   ├── correlation_matrix.png

│   ├── customer_spending_distribution.png

│   ├── orders_vs_spend.png

│   ├── income_vs_spend.png

│   ├── elbow_method.png

│   ├── customer_segments.png

│   ├── segment_distribution.png

│   ├── segment_spending.png

│   └── segment_behavior_comparison.png

├── README.md

└── requirements.txt

## 🚀 Outcome

This project demonstrates the application of customer analytics and unsupervised machine learning to identify customer groups and translate analytical findings into business strategies.