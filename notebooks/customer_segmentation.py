# Customer Segmentation Analysis
# Thiranex - Data Analytics Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("../data/customer_segmentation_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# -----------------------------
# 2. Basic Information
# -----------------------------

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# 3. Missing Values
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 4. Duplicate Records
# -----------------------------

print("\nDuplicate Rows:", df.duplicated().sum())

# -----------------------------
# 5. Data Types
# -----------------------------

print("\nData Types:")
print(df.dtypes)

# -----------------------------
# 6. Categorical Analysis
# -----------------------------

print("\nGender Distribution:")
print(df["Gender"].value_counts())

print("\nCity Distribution:")
print(df["City"].value_counts())

print("\nEducation Distribution:")
print(df["Education"].value_counts())

# -----------------------------
# 7. Numerical Distributions
# -----------------------------

numerical_columns = [
    "Age",
    "AnnualIncome_INR",
    "Tenure_Months",
    "Orders",
    "AverageOrderValue_INR",
    "TotalSpend_INR",
    "Recency_Days",
    "WebsiteVisits",
    "AppUsage",
    "DiscountUsage",
    "SatisfactionScore"
]

df[numerical_columns].hist(
    figsize=(15, 12),
    bins=20
)

plt.tight_layout()
plt.savefig("../visuals/numerical_distributions.png")
plt.show()

# -----------------------------
# 8. Correlation Analysis
# -----------------------------

plt.figure(figsize=(12, 8))

correlation = df[numerical_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="Blues"
)

plt.title("Customer Behavior Correlation Matrix")
plt.tight_layout()

plt.savefig("../visuals/correlation_matrix.png")
plt.show()

# -----------------------------
# 9. Spending Analysis
# -----------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["TotalSpend_INR"],
    bins=30,
    kde=True
)

plt.title("Distribution of Customer Spending")
plt.xlabel("Total Spend (INR)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("../visuals/customer_spending_distribution.png")
plt.show()

# -----------------------------
# 10. Orders vs Total Spend
# -----------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Orders",
    y="TotalSpend_INR"
)

plt.title("Orders vs Total Customer Spend")
plt.xlabel("Number of Orders")
plt.ylabel("Total Spend (INR)")

plt.tight_layout()
plt.savefig("../visuals/orders_vs_spend.png")
plt.show()

# -----------------------------
# 11. Income vs Spending
# -----------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="AnnualIncome_INR",
    y="TotalSpend_INR"
)

plt.title("Annual Income vs Total Spend")
plt.xlabel("Annual Income (INR)")
plt.ylabel("Total Spend (INR)")

plt.tight_layout()
plt.savefig("../visuals/income_vs_spend.png")
plt.show()

print("\nExploratory Data Analysis completed successfully.")


# -----------------------------
# 12. Customer Segmentation
# -----------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Features selected for customer segmentation
segmentation_features = [
    "AnnualIncome_INR",
    "Orders",
    "AverageOrderValue_INR",
    "TotalSpend_INR",
    "Recency_Days",
    "WebsiteVisits",
    "AppUsage",
    "DiscountUsage"
]

X = df[segmentation_features].copy()

# -----------------------------
# 13. Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures selected for clustering:")
print(segmentation_features)

print("\nScaled data shape:", X_scaled.shape)

# -----------------------------
# 14. Elbow Method
# -----------------------------

inertia = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.xticks(range(2, 11))
plt.grid(True)

plt.tight_layout()

plt.savefig("../visuals/elbow_method.png")
plt.show()

# -----------------------------
# 15. Apply K-Means Clustering
# -----------------------------

optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCustomer Cluster Distribution:")
print(df["Cluster"].value_counts().sort_index())

# -----------------------------
# 16. Cluster Profiles
# -----------------------------

cluster_profile = df.groupby("Cluster")[segmentation_features].mean().round(2)

print("\nCluster Profiles:")
print(cluster_profile)

# -----------------------------
# 17. Cluster Visualization
# -----------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Orders",
    y="TotalSpend_INR",
    hue="Cluster",
    palette="viridis",
    s=80
)

plt.title("Customer Segments: Orders vs Total Spend")
plt.xlabel("Number of Orders")
plt.ylabel("Total Spend (INR)")
plt.legend(title="Cluster")

plt.tight_layout()

plt.savefig("../visuals/customer_segments.png")
plt.show()

# -----------------------------
# 18. Save Segmented Dataset
# -----------------------------

df.to_csv(
    "../data/customer_segments.csv",
    index=False
)

print("\nSegmented dataset saved successfully.")

# -----------------------------
# 19. Analyze Cluster Profiles
# -----------------------------

print("\nDetailed Cluster Profiles:")
print(cluster_profile)

# Add customer count
cluster_profile["CustomerCount"] = df["Cluster"].value_counts().sort_index()

print("\nCluster Profiles with Customer Count:")
print(cluster_profile)

# -----------------------------
# 20. Cluster Summary
# -----------------------------

for cluster in cluster_profile.index:
    print(f"\nCluster {cluster}")
    print("-" * 40)
    print(f"Customers: {cluster_profile.loc[cluster, 'CustomerCount']}")
    print(f"Average Income: ₹{cluster_profile.loc[cluster, 'AnnualIncome_INR']:,.0f}")
    print(f"Average Orders: {cluster_profile.loc[cluster, 'Orders']:.1f}")
    print(f"Average Order Value: ₹{cluster_profile.loc[cluster, 'AverageOrderValue_INR']:,.0f}")
    print(f"Average Total Spend: ₹{cluster_profile.loc[cluster, 'TotalSpend_INR']:,.0f}")
    print(f"Average Recency: {cluster_profile.loc[cluster, 'Recency_Days']:.1f} days")
    print(f"Website Visits: {cluster_profile.loc[cluster, 'WebsiteVisits']:.1f}")
    print(f"App Usage: {cluster_profile.loc[cluster, 'AppUsage']:.1f}")
    print(f"Discount Usage: {cluster_profile.loc[cluster, 'DiscountUsage']:.1f}")

# -----------------------------
# 21. Customer Segment Analysis
# -----------------------------

# Calculate average metrics for each cluster
segment_profile = df.groupby("Cluster")[segmentation_features].mean()

# Add customer count
segment_profile["CustomerCount"] = df["Cluster"].value_counts()

# Calculate percentage of customers
segment_profile["CustomerPercentage"] = (
    segment_profile["CustomerCount"] / len(df) * 100
).round(1)

print("\n" + "=" * 70)
print("FINAL CUSTOMER SEGMENT PROFILE")
print("=" * 70)

print(segment_profile.round(2))

# -----------------------------
# 22. Identify Segment Characteristics
# -----------------------------

for cluster in segment_profile.index:

    profile = segment_profile.loc[cluster]

    print("\n" + "=" * 50)
    print(f"CLUSTER {cluster}")
    print("=" * 50)

    print(f"Customers       : {int(profile['CustomerCount'])}")
    print(f"Percentage      : {profile['CustomerPercentage']}%")
    print(f"Income          : ₹{profile['AnnualIncome_INR']:,.0f}")
    print(f"Orders          : {profile['Orders']:.1f}")
    print(f"Avg Order Value : ₹{profile['AverageOrderValue_INR']:,.0f}")
    print(f"Total Spend     : ₹{profile['TotalSpend_INR']:,.0f}")
    print(f"Recency         : {profile['Recency_Days']:.1f} days")
    print(f"Website Visits  : {profile['WebsiteVisits']:.1f}")
    print(f"App Usage       : {profile['AppUsage']:.1f}")
    print(f"Discount Usage  : {profile['DiscountUsage']:.1f}")

# -----------------------------
# 23. Save Segment Profile
# -----------------------------

segment_profile.round(2).to_csv(
    "../data/customer_segment_profiles.csv"
)

print("\nSegment profile saved successfully.")

# -----------------------------
# 24. Segment Size Visualization
# -----------------------------

plt.figure(figsize=(9, 6))

cluster_counts = df["Cluster"].value_counts().sort_index()

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title("Customer Distribution by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("../visuals/segment_distribution.png")
plt.show()

# -----------------------------
# 26. Business Insights
# -----------------------------

print("\n" + "=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)

highest_spend_cluster = segment_profile["TotalSpend_INR"].idxmax()
highest_orders_cluster = segment_profile["Orders"].idxmax()
lowest_spend_cluster = segment_profile["TotalSpend_INR"].idxmin()
most_recent_cluster = segment_profile["Recency_Days"].idxmin()
least_recent_cluster = segment_profile["Recency_Days"].idxmax()

print(
    f"\n1. Cluster {highest_spend_cluster} has the highest "
    f"average customer spending."
)

print(
    f"2. Cluster {highest_orders_cluster} has the highest "
    f"average number of orders."
)

print(
    f"3. Cluster {lowest_spend_cluster} has the lowest "
    f"average customer spending and may require targeted offers."
)

print(
    f"4. Cluster {most_recent_cluster} has the lowest recency, "
    f"indicating the most recently active customers."
)

print(
    f"5. Cluster {least_recent_cluster} has the highest recency, "
    f"indicating customers who may need re-engagement."
)

# -----------------------------
# 27. Segment Performance Comparison
# -----------------------------

metrics = [
    "Orders",
    "AverageOrderValue_INR",
    "TotalSpend_INR",
    "WebsiteVisits",
    "AppUsage"
]

# Normalize metrics so they can be compared
from sklearn.preprocessing import MinMaxScaler

comparison_data = segment_profile[metrics].copy()

comparison_scaler = MinMaxScaler()

comparison_scaled = pd.DataFrame(
    comparison_scaler.fit_transform(comparison_data),
    columns=metrics,
    index=comparison_data.index
)

plt.figure(figsize=(12, 7))

for cluster in comparison_scaled.index:
    plt.plot(
        comparison_scaled.columns,
        comparison_scaled.loc[cluster],
        marker="o",
        label=f"Cluster {cluster}"
    )

plt.title("Customer Segment Behavioral Comparison")
plt.xlabel("Customer Behavior Metrics")
plt.ylabel("Normalized Score")
plt.legend()

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("../visuals/segment_behavior_comparison.png")
plt.show()

# -----------------------------
# 29. Business Recommendations
# -----------------------------

recommendations = {
    "Premium High-Value Customers":
        "Focus on VIP rewards, premium products, personalized recommendations, and exclusive offers.",

    "Active Loyal Customers":
        "Encourage repeat purchases through loyalty rewards, cross-selling, and personalized product recommendations.",

    "High-Value At-Risk":
        "Launch re-engagement campaigns using personalized offers, reminders, and limited-time incentives.",

    "Low-Value Customers":
        "Use targeted promotions, product bundles, and personalized recommendations to increase average order value."
}

print("\n" + "=" * 70)
print("BUSINESS RECOMMENDATIONS")
print("=" * 70)

for segment, recommendation in recommendations.items():
    print(f"\n{segment}:")
    print(recommendation)