import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 保存实验结果的列表
line_results = []
poly_results = []
rbf_results = []

# 数据加载与预处理
def load_data():
    # 假设csv数据已经加载
    data = pd.read_csv('data.csv')  # 修改为你的数据文件路径
    x = data.iloc[:, :-1]  # 特征
    y = data.iloc[:, -1]  # 标签
    return x, y

def preprocess_data(x, y):
    # 标准化特征
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # 前500行作为训练集
    x_train = x[:500]
    y_train = y[:500]

    # 后69行作为测试集/预测集
    x_test = x[500:]
    y_test = y[500:]  # 如果有标签的话，使用它进行评估

    return x_train, x_test, y_train, y_test

# 线性SVM函数
def line_svm(X_train, X_test, y_train, y_test, c):
    svm_model = SVC(kernel='linear', C=c)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"linear模型(C:{c})的准确率: {accuracy:.2f}")
    # 保存结果
    line_results.append({ "C": c, "Accuracy": accuracy})

# 多项式核SVM函数
def Poly_svm(X_train, X_test, y_train, y_test, d, co, c):
    svm_model = SVC(kernel='poly', degree=d, coef0=co, C=c)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Poly模型(C:{c}, degree:{d}, coef0:{co})的准确率: {accuracy:.2f}")
    # 保存结果
    poly_results.append({"C": c, "Degree": d, "Coef0": co, "Accuracy": accuracy})

# RBF核SVM函数
def Rbf_svm(X_train, X_test, y_train, y_test, c, g):
    svm_model = SVC(kernel='rbf', C=c, gamma=g)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"RBF模型(C:{c}, gamma:{g})的准确率: {accuracy:.2f}")
    # 保存结果
    rbf_results.append({ "C": c, "Gamma": g, "Accuracy": accuracy})

# 调用示例
def main():
    # 加载和处理数据
    x, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(x, y)

    # 1. 线性SVM
    print("线性SVM实验：")
    line_svm(X_train, X_test, y_train, y_test, 1)
    line_svm(X_train, X_test, y_train, y_test, 10)
    line_svm(X_train, X_test, y_train, y_test, 100)
    line_svm(X_train, X_test, y_train, y_test, 0.1)
    line_svm(X_train, X_test, y_train, y_test, 50)
    line_svm(X_train, X_test, y_train, y_test, 0.01)
    line_svm(X_train, X_test, y_train, y_test, 200)
    line_svm(X_train, X_test, y_train, y_test, 0.5)
    line_svm(X_train, X_test, y_train, y_test, 5)
    line_svm(X_train, X_test, y_train, y_test, 0.001)
    line_svm(X_train, X_test, y_train, y_test, 1000)
    line_svm(X_train, X_test, y_train, y_test, 20)
    line_svm(X_train, X_test, y_train, y_test, 2)
    line_svm(X_train, X_test, y_train, y_test, 0.05)
    line_svm(X_train, X_test, y_train, y_test, 2000)
    line_svm(X_train, X_test, y_train, y_test, 0.005)
    line_svm(X_train, X_test, y_train, y_test, 500)
    line_svm(X_train, X_test, y_train, y_test, 0.2)
    line_svm(X_train, X_test, y_train, y_test, 0.25)

    # 2. 多项式核SVM
    print("\n多项式SVM实验：")
    Poly_svm(X_train, X_test, y_train, y_test, 2, 0, 1)
    Poly_svm(X_train, X_test, y_train, y_test, 3, 0, 1)
    Poly_svm(X_train, X_test, y_train, y_test, 2, 1, 10)
    Poly_svm(X_train, X_test, y_train, y_test, 5, 1, 0.1)
    Poly_svm(X_train, X_test, y_train, y_test, 3, 1, 100)
    Poly_svm(X_train, X_test, y_train, y_test, 2, 0, 100)
    Poly_svm(X_train, X_test, y_train, y_test, 3, 0, 10)
    Poly_svm(X_train, X_test, y_train, y_test, 4, 1, 1)
    Poly_svm(X_train, X_test, y_train, y_test, 6, 0, 0.5)
    Poly_svm(X_train, X_test, y_train, y_test, 3, 0, 0.5)
    Poly_svm(X_train, X_test, y_train, y_test, 2, 1, 1)
    Poly_svm(X_train, X_test, y_train, y_test, 4, 0, 50)
    Poly_svm(X_train, X_test, y_train, y_test, 3, 1, 1)
    Poly_svm(X_train, X_test, y_train, y_test, 5, 0, 10)
    Poly_svm(X_train, X_test, y_train, y_test, 2, 1, 0.5)
    Poly_svm(X_train, X_test, y_train, y_test, 5, 1, 1)
    Poly_svm(X_train, X_test, y_train, y_test, 6, 0, 10)
    Poly_svm(X_train, X_test, y_train, y_test, 4, 1, 0.1)
    Poly_svm(X_train, X_test, y_train, y_test, 3, 0, 0.1)
    Poly_svm(X_train, X_test, y_train, y_test, 8, 0, 1)
    Poly_svm(X_train, X_test, y_train, y_test, 1, 0, 1)
    Poly_svm(X_train, X_test, y_train, y_test, 8, 1, 1)
    Poly_svm(X_train, X_test, y_train, y_test, 1, 1, 1)

    # 3. RBF核SVM
    print("\nRBF SVM实验：")
    Rbf_svm(X_train, X_test, y_train, y_test, 1, 0.1)
    Rbf_svm(X_train, X_test, y_train, y_test, 10, 0.1)
    Rbf_svm(X_train, X_test, y_train, y_test, 100, 0.1)
    Rbf_svm(X_train, X_test, y_train, y_test, 1, 1)
    Rbf_svm(X_train, X_test, y_train, y_test, 10, 1)
    Rbf_svm(X_train, X_test, y_train, y_test, 0.1, 0.1)
    Rbf_svm(X_train, X_test, y_train, y_test, 50, 0.5)
    Rbf_svm(X_train, X_test, y_train, y_test, 0.1, 10)
    Rbf_svm(X_train, X_test, y_train, y_test, 1000, 1)
    Rbf_svm(X_train, X_test, y_train, y_test, 0.5, 0.01)

    # 将每个结果保存到不同的CSV文件
    pd.DataFrame(line_results).to_csv('line_results.csv', index=False)
    pd.DataFrame(poly_results).to_csv('poly_results.csv', index=False)
    pd.DataFrame(rbf_results).to_csv('rbf_results.csv', index=False)

if __name__ == "__main__":
    main()
