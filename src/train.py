import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/otodom.csv')
df = df.dropna()
X = df[['area']]
Y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))