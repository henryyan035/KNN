from sklearn.neighbors import KNeighborsClassifier

x = [[0], [1], [3], [4], [2]]
y = [0,0,1,1,1]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)
print(model.predict([[5]]))