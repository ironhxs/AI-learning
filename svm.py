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
    return accuracy
# 多项式核SVM函数
def Poly_svm(X_train, X_test, y_train, y_test, d, co, c):
    svm_model = SVC(kernel='poly', degree=d, coef0=co, C=c)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Poly模型(C:{c}, degree:{d}, coef0:{co})的准确率: {accuracy:.2f}")
    # 保存结果
    poly_results.append({"C": c, "Degree": d, "Coef0": co, "Accuracy": accuracy})
    return accuracy
# RBF核SVM函数
def Rbf_svm(X_train, X_test, y_train, y_test, c, g):
    svm_model = SVC(kernel='rbf', C=c, gamma=g)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"RBF模型(C:{c}, gamma:{g})的准确率: {accuracy:.2f}")
    # 保存结果
    rbf_results.append({ "C": c, "Gamma": g, "Accuracy": accuracy})
    return accuracy
# 线性SVM实验
def linear_svm_experiment(X_train, X_test, y_train, y_test, C_values):
        print("线性SVM实验：")
        best_C = 0
        best_accuracy = 0
        for C in C_values:
            accuracy = line_svm(X_train, X_test, y_train, y_test, C)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_C = C
        print(f"最优C值：{best_C}, 准确率：{best_accuracy}")
        return best_C, best_accuracy

    # 多项式SVM实验
def poly_svm_experiment(X_train, X_test, y_train, y_test, degree_values, coef0_values, C_values):
        print("\n多项式SVM实验：")
        best_params = 0
        best_accuracy = 0
        for degree in degree_values:
            for coef0 in coef0_values:
                for C in C_values:
                    accuracy = Poly_svm(X_train, X_test, y_train, y_test, degree, coef0, C)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (degree, coef0, C)
        print(f"最优参数：degree={best_params[0]}, coef0={best_params[1]}, C={best_params[2]}, 准确率：{best_accuracy}")
        return best_params, best_accuracy

    # RBF SVM实验
def rbf_svm_experiment(X_train, X_test, y_train, y_test, C_values, gamma_values):
        print("\nRBF SVM实验：")
        best_params = 0
        best_accuracy = 0
        for C in C_values:
            for gamma in gamma_values:
                accuracy = Rbf_svm(X_train, X_test, y_train, y_test, C, gamma)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (C, gamma)
        print(f"最优参数：C={best_params[0]}, gamma={best_params[1]}, 准确率：{best_accuracy}")
        return best_params, best_accuracy

def main():
    # 加载和处理数据
    x, y = load_data()
    x_train, x_test, y_train, y_test = preprocess_data(x, y)
    C_values = [1, 10, 100, 0.1, 50, 0.01]
    degree_values = [2, 3, 4, 5]
    coef0_values = [0, 1, 10]
    gamma_values = [0.1, 1, 10, 100]

    # 假设已经加载了数据：X_train, X_test, y_train, y_test

    # 进行线性SVM实验
    best_C, best_accuracy_linear = linear_svm_experiment(x_train, x_test, y_train, y_test, C_values)

    # 进行多项式SVM实验
    best_params_poly, best_accuracy_poly = poly_svm_experiment(x_train, x_test, y_train, y_test, degree_values,
                                                               coef0_values, C_values)

    # 进行RBF SVM实验
    best_params_rbf, best_accuracy_rbf = rbf_svm_experiment(x_train, x_test, y_train, y_test, C_values, gamma_values)

    # 输出最终的最优参数和准确率
    print("\n最优参数和准确率：")
    print(f"线性SVM最优C值：{best_C}, 准确率：{best_accuracy_linear}")
    print(
        f"多项式SVM最优参数：degree={best_params_poly[0]}, coef0={best_params_poly[1]}, C={best_params_poly[2]}, 准确率：{best_accuracy_poly}")
    print(f"RBF SVM最优参数：C={best_params_rbf[0]}, gamma={best_params_rbf[1]}, 准确率：{best_accuracy_rbf}")

    # 将每个结果保存到不同的CSV文件
    pd.DataFrame(line_results).to_csv('line_results.csv', index=False)
    pd.DataFrame(poly_results).to_csv('poly_results.csv', index=False)
    pd.DataFrame(rbf_results).to_csv('rbf_results.csv', index=False)

if __name__ == "__main__":
    main()
