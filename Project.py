import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. LOAD DATA
file_path = r"C:\Users\Asus\Downloads\spotify_user_behavior_realistic_50000_rows.csv"
df = pd.read_csv(file_path)

# 2. CLEAN DATA
df = df.drop_duplicates().dropna()


# 3. VISUALIZATIONS 
plt.style.use("seaborn-v0_8")

# 1. Device Usage 
df["primary_device"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.title("Device Usage Distribution")
plt.ylabel("")
plt.show()

# 2. Top Genres (Bar)
df["favorite_genre"].value_counts().head(10).plot(
    kind="bar",
    color="#8172B2"
)
plt.title("Top 10 Music Genres")
plt.xlabel("Genre")
plt.ylabel("Users")
plt.xticks(rotation=45)
plt.show()

# 3. Age vs Listening (Scatter)
sample = df.sample(500)
plt.scatter(
    sample["age"],
    sample["avg_listening_hours_per_week"],
    color="#DD8452",
    alpha=0.7
)
plt.title("Age vs Listening Hours")
plt.xlabel("Age")
plt.ylabel("Hours")
plt.show()

# 4. Subscription Status (Bar)
df["subscription_status"].value_counts().plot(
    kind="bar",
    color="#64B5CD"
)
plt.title("Subscription Status")
plt.xlabel("Type")
plt.ylabel("Users")
plt.show()

# 5. Avg Listening by Device (Bar)
df.groupby("primary_device")["avg_listening_hours_per_week"].mean().plot(
    kind="bar",
    color="#55A868"
)
plt.title("Avg Listening Hours by Device")
plt.xlabel("Device")
plt.ylabel("Hours")
plt.show()

# 6. Genre vs Listening (Horizontal Bar)
top_genres = df["favorite_genre"].value_counts().head(5).index
filtered = df[df["favorite_genre"].isin(top_genres)]

filtered.groupby("favorite_genre")["avg_listening_hours_per_week"].mean().plot(
    kind="barh",
    color="#C44E52"
)
plt.title("Top Genres vs Avg Listening Hours")
plt.xlabel("Hours")
plt.ylabel("Genre")
plt.show()

# 7. Age Distribution 
plt.hist(
    df["age"],
    bins=20,
    color="#7FB3D5",
    edgecolor="black",   
    linewidth=1          
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 4. LINEAR REGRESSION 

X = df[["age"]]
y = df["avg_listening_hours_per_week"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Default matplotlib colors
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)

plt.title("Linear Regression: Age vs Listening Hours")
plt.xlabel("Age")
plt.ylabel("Listening Hours")
plt.show()

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)