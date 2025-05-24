# 0.导入工具包
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# from sklearn.model_selection


# 加载数据集
iris = load_iris()
# 特征工程（预处理-标准化）
# 数据集划分
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0, stratify=y)
#  标准化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# 模型实例化+交叉验证+网格搜索
model = KNeighborsClassifier(n_neighbors=1)
# print(model)
paras_grid = {'n_neighbors':[3,4,5,7,9]}
estimator =GridSearchCV(estimator=model,param_grid=paras_grid,cv=4)
estimator.fit(X_train,y_train)
# print(estimator.best_score_)
# print(estimator.best_estimator_)
# print(estimator.cv_results_)
print("best param: ",estimator.best_params_)
# model = estimator.best_estimator_
# score = model.score(X_test,y_test)
# print(score)




# 计算测试样本中model预测对了多少
# score = model.score(X_test, y_test)
# print(score)

# 4.模型实例化+交叉验证+网格搜索

# estimator =GridSearchCV(estimator=model,param_grid=paras_grid,cv=4)
# estimator.fit(x_train,y_train)

