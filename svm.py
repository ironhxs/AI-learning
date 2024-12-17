import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# 1.加载数据集 加载 CSV 文件
data = pd.read_csv('data.csv')

# 查看前几行数据
# print(df.head())

# 2.数据预处理
# 提取特征和标签
X = data.iloc[:, :-1]  # 前30列特征
y = data.iloc[:, -1]   # 第31列标签

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 查看标准化后的前五行
# print(X_scaled[:5])

# 3.数据划分
# 前500行作为训练集
X_train = X.iloc[:500]
y_train = y.iloc[:500]

# 后69行作为测试集/预测集
X_test = X.iloc[500:]
y_test = y.iloc[500:]  # 如果有标签的话，使用它进行评估

# 4.选择模型
def line_svm(c):
    # 创建线性SVM模型
    svm_model = SVC(kernel='linear', C=c)
    svm_model.fit(X_train, y_train)
    # 预测测试集
    y_pred = svm_model.predict(X_test)

    # 计算准确率
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"linear模型(C:{c})的准确率: {accuracy:.2f}")
def Poly_svm(d, co , c):
    # 创建线性SVM模型
    svm_model = SVC(kernel='poly', degree=d, coef0=co, C=c)
    svm_model.fit(X_train, y_train)
    # 预测测试集
    y_pred = svm_model.predict(X_test)

    # 计算准确率
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Poly模型(degree:{d},coef0:{co})的准确率: {accuracy:.2f}")

def Rdf_svm(d, co):
    # 创建线性SVM模型
    svm_model = SVC(kernel='poly', degree=d, coef0=co)
    svm_model.fit(X_train, y_train)
    # 预测测试集
    y_pred = svm_model.predict(X_test)

    # 计算准确率
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Poly模型(degree:{d},coef0:{co})的准确率: {accuracy:.2f}")

line_svm(1)
line_svm(10)
line_svm(100)
Poly_svm(3, 0)
Poly_svm(8, 0)
Poly_svm(2, 0)
Poly_svm(1, 0)
Poly_svm(3, 1)
Poly_svm(8, 1)
Poly_svm(2, 1)
Poly_svm(1, 1)