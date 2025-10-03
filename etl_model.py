import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

df = pd.read_csv('data/flights.csv')
df['duration_mins'].fillna(df['duration_mins'].median(), inplace=True)

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]

df['route'] = df['source'] + '-' + df['destination']

avg_price_route = df.groupby('route')['price'].mean().sort_values(ascending=False).head(5)

plt.figure(figsize=(8,5))
sns.barplot(x=avg_price_route.values, y=avg_price_route.index)
plt.title('Top 5 Routes by Avg Price')
plt.xlabel('Average Price')
plt.ylabel('Route')
plt.tight_layout()
plt.savefig('top_routes.png')
plt.show()

X = df[['days_to_departure','duration_mins','total_stops']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print(' Model saved to models/model.pkl')
