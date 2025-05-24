import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# 获取数据集
from sklearn.datasets import load_iris
# 数据集划分
from sklearn.model_selection import train_test_split
# 标准化
from sklearn.preprocessing import StandardScaler
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 加载数据集
iris = load_iris()
# print(iris)
# print(iris.target)
# 展示数据集
# 把数据转换成dataframe格式，设置data,colmnus等属性
# iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(iris_df)
# iris_df['lable'] = iris.target
# print(iris_df)
# col1 = 'sepal length (cm)'
# col2 = 'petal width (cm)'
# sns.lmplot显示
# sns.lmplot(x=col1, y=col2, data=iris_df,hue='lable', palette='Set1')
# plt.xlabel(col1)
# plt.ylabel(col2)
# plt.title('iris')
# plt.show()
# 特征工程（预处理-标准化）
# 数据集划分
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0, stratify=y)
# print(len(X_train),len(X))
#  标准化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# 模型训练
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
# 计算测试样本中model预测对了多少
score = model.score(X_test, y_test)
print(score)
# 模型预测
x = [[5.1,3.5,1.4,0.2]]
# 标准化
x = sc.transform(x)
y_pred = model.predict(x)
print(y_pred)
print(model.predict_proba(x))

