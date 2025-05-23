from sklearn.neighbors import KNeighborsRegressor
x = [[0],[1],[2],[3]]
y = [0.1,0.2,0.3,0.4]
model = KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
print(model.predict([[0.5]]))