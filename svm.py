# 学习曲线绘制函数
import numpy as np
import pandas as pd  # 用于数据处理
import sklearn.metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler  # 用于标准化数据
from sklearn.svm import SVC  # 支持向量机（SVM）
from sklearn.tree import DecisionTreeClassifier  # 决策树
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为 SimHei，支持显示中文
rcParams['font.family'] = ['SimHei']  # SimHei 是黑体
rcParams['axes.unicode_minus'] = False  # 防止负号显示成方块

matplotlib.use('TkAgg')  # 更改 Matplotlib 后端

# 保存实验结果的列表
line_results = []
poly_results = []
rbf_results = []
decision_tree_results = []
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
    y_train_pred = svm_model.predict(X_train)
    y_pred = svm_model.predict(X_test)
    train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f"linear模型(C:{c})的准确率: {accuracy:.2f}")
    # 保存结果
    line_results.append({ "C": c, "Accuracy": accuracy})
    return train_accuracy, accuracy

# 多项式核SVM函数
def Poly_svm(X_train, X_test, y_train, y_test, d, co, c):
    svm_model = SVC(kernel='poly', degree=d, coef0=co, C=c)
    svm_model.fit(X_train, y_train)
    y_train_pred = svm_model.predict(X_train)
    y_pred = svm_model.predict(X_test)
    train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f"Poly模型(C:{c}, degree:{d}, coef0:{co})的准确率: {accuracy:.2f}")
    # 保存结果
    poly_results.append({"C": c, "Degree": d, "Coef0": co, "Accuracy": accuracy})
    return train_accuracy, accuracy

# RBF核SVM函数
def Rbf_svm(X_train, X_test, y_train, y_test, c, g):
    svm_model = SVC(kernel='rbf', C=c, gamma=g)
    svm_model.fit(X_train, y_train)
    y_train_pred = svm_model.predict(X_train)
    y_pred = svm_model.predict(X_test)
    train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f"RBF模型(C:{c}, gamma:{g})的准确率: {accuracy:.2f}")
    # 保存结果
    rbf_results.append({ "C": c, "Gamma": g, "Accuracy": accuracy})
    return train_accuracy, accuracy

#神经网络模型
def nn(x_train, x_test, y_train, y_test):
    # 构建神经网络模型
    model = Sequential()

    # 添加输入层和第一个隐藏层
    model.add(Input(shape=(x_train.shape[1],)))

    model.add(Dense(units=128, activation='relu'))

    # 添加第二个隐藏层
    model.add(Dense(units=32, activation='relu'))

    # 添加输出层
    model.add(Dense(units=1, activation='sigmoid'))  # 二分类任务，输出0或1

    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

    # 在测试集上评估模型
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)  # 将输出概率值转换为0或1
    y_train_pred = model.predict(x_train)
    y_train_pred =  (y_train_pred > 0.5)
    # 输出分类准确率
    train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f'神经网络模型的准确率: {accuracy:.4f}')
    return train_accuracy,  accuracy

# 决策树函数
def decision_tree(X_train, X_test, y_train, y_test, max_depth, min_samples_split, min_samples_leaf, criterion):
    # 创建决策树模型
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    # 计算准确率
    train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    return train_accuracy, accuracy

# 线性SVM实验 搜索找到最优超参数
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

# 多项式SVM实验 搜索找到最优超参数
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

# RBF SVM实验 搜索找到最优超参数
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

# 决策树实验 搜索找到最优超参数
def decision_tree_experiment(X_train, X_test, y_train, y_test, max_depth_values, min_samples_split_values,min_samples_leaf_values, criterion_values):
    print("\n决策树实验：")
    best_params = None
    best_accuracy = 0

    # 遍历所有超参数组合
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for min_samples_leaf in min_samples_leaf_values:
                for criterion in criterion_values:
                    # 进行模型训练和评估
                    accuracy = decision_tree(X_train, X_test, y_train, y_test, max_depth, min_samples_split,
                                             min_samples_leaf, criterion)

                    # 输出当前参数组合的结果
                    print(
                        f"参数组合：max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, criterion={criterion}, 准确率={accuracy:.4f}")

                    decision_tree_results.append({"max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "criterion": criterion,
                        "Accuracy": accuracy})
                    # 更新最优参数和准确率
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (max_depth, min_samples_split, min_samples_leaf, criterion)

    print(f"最优参数：max_depth={best_params[0]}, min_samples_split={best_params[1]}, min_samples_leaf={best_params[2]}, criterion={best_params[3]}, 准确率：{best_accuracy:.4f}")

    return best_params, best_accuracy

#画机器学习的学习曲线
def plot_learning_curve(model, X_train, y_train):
    # 获取模型的名称
    model_name = model.__class__.__name__
    # 如果是 SVM，获取其核函数
    if model_name == "SVC":
        kernel_type = model.kernel
        model_name += f" ({kernel_type} 核)"
    # 获取学习曲线的数据 StratifiedKFold是 sklearn.model_selection 中的一种交叉验证方法，与普通的 KFold 不同，它在每个折叠中都保持各个类别的比例一致。这对于处理类别不平衡的数据特别有用。
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)

    # 计算平均值和标准差
    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)

    # 绘制学习曲线
    plt.plot(train_sizes, train_mean, label='训练集')
    plt.plot(train_sizes, valid_mean, label='验证集')
    plt.xlabel('训练集大小')
    plt.ylabel('准确率')
    plt.title(f'学习曲线 - {model_name}')  # 在标题中显示模型名称
    plt.legend()
    plt.show()

#画神经网络的学习曲线
def plot_nn_learning_curve(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    :param model: Keras 神经网络模型
    :param X_train: 训练数据特征
    :param y_train: 训练数据标签
    :param X_test: 测试数据特征
    :param y_test: 测试数据标签
    :param epochs: 训练的 epoch 数
    :param batch_size: 批次大小
    """
    # 训练模型并获取历史记录，同时计算验证集（测试集）准确率
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test, y_test), verbose=0)

    # 获取训练过程中的准确率
    train_acc = history.history['accuracy']
    # 获取测试集的准确率
    val_acc = history.history['val_accuracy']

    # 绘制训练集和测试集的准确率
    plt.plot(range(1, epochs + 1), train_acc, label='训练集准确率')
    plt.plot(range(1, epochs + 1), val_acc, label='测试集准确率')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('神经网络学习曲线')
    plt.legend()
    plt.show()

# 评估所有模型的函数
def evaluate_models(X_train, X_test, y_train, y_test):
    # 学习曲线：绘制每个模型的学习曲线
    print("\n绘制学习曲线：")

    line_svm_model = SVC(kernel='linear', C=100)
    poly_svm_model = SVC(kernel='poly', degree=1, coef0=0, C=500)
    rbf_svm_model = SVC(kernel='rbf', C=50, gamma=0.01)

    nn_model = Sequential()
    # 添加输入层和第一个隐藏层
    nn_model.add(Input(shape=(X_train.shape[1],)))
    nn_model.add(Dense(units=128, activation='relu'))
    nn_model.add(Dense(units=32, activation='relu'))
    nn_model.add(Dense(units=1, activation='sigmoid'))
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    tree_model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=4, criterion='entropy')

    print("\n线性SVM学习曲线：")
    plot_learning_curve(line_svm_model, X_train, y_train)

    print("\n多项式SVM学习曲线：")
    plot_learning_curve(poly_svm_model, X_train, y_train)

    print("\nRBF SVM学习曲线：")
    plot_learning_curve(rbf_svm_model, X_train, y_train)

    print("\n神经网络学习曲线：")
    plot_nn_learning_curve(nn_model, X_train, y_train,X_test, y_test, epochs=50)

    print("\n决策树学习曲线：")
    plot_learning_curve(tree_model, X_train, y_train)

    # 线性SVM
    line_train_acc, line_test_acc = line_svm(X_train, X_test, y_train, y_test, 100)

    # 多项式SVM
    poly_train_acc, poly_test_acc = Poly_svm(X_train, X_test, y_train, y_test, 1, 0, 500)

    # RBF SVM
    rbf_train_acc, rbf_test_acc = Rbf_svm(X_train, X_test, y_train, y_test, 50, 0.01)

    # 神经网络
    nn_train_acc, nn_test_acc = nn(X_train, X_test, y_train, y_test)

    # 决策树
    tree_train_acc, tree_test_acc = decision_tree(X_train, X_test, y_train, y_test, max_depth=5, min_samples_split=2,min_samples_leaf=4, criterion='entropy')

    # 输出各模型的训练集和测试集准确率对比来直观检验是否过拟合
    print(f"\n模型的训练集和测试集准确率：")
    print(f"线性SVM - 训练集: {line_train_acc:.4f}, 测试集: {line_test_acc:.4f}")
    print(f"多项式SVM - 训练集: {poly_train_acc:.4f}, 测试集: {poly_test_acc:.4f}")
    print(f"RBF SVM - 训练集: {rbf_train_acc:.4f}, 测试集: {rbf_test_acc:.4f}")
    print(f"神经网络 - 训练集: {nn_train_acc:.4f}, 测试集: {nn_test_acc:.4f}")
    print(f"决策树 - 训练集: {tree_train_acc:.4f}, 测试集: {tree_test_acc:.4f}")

#测试各个svm的函数
def test():
    # 加载和处理数据
    x, y = load_data()
    x_train, x_test, y_train, y_test = preprocess_data(x, y)
    C_values = [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]
    degree_values = [1, 2, 3, 4, 5, 6]
    coef0_values = [0, 0.5, 1, 5, 10, 50]
    gamma_values = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]

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

#测试神经网路的函数
def test2():
    x, y = load_data()
    x_train, x_test, y_train, y_test = preprocess_data(x, y)
    nn(x_train, x_test, y_train, y_test)

#测试决策树的函数
def test3():
    x, y = load_data()
    x_train, x_test, y_train, y_test = preprocess_data(x, y)
    max_depth_values = [5, 10, 15, 20, None]  # None表示没有限制最大深度
    min_samples_split_values = [2, 5, 10]
    min_samples_leaf_values = [1, 2, 4]
    criterion_values = ['gini', 'entropy']
    best_params, best_accuracy = decision_tree_experiment(x_train, x_test, y_train, y_test, max_depth_values,min_samples_split_values, min_samples_leaf_values,criterion_values)
    pd.DataFrame(decision_tree_results).to_csv('_decision_tree_results.csv', index=False)

#评估各个模型的函数
def test4():
    x, y = load_data()
    x_train, x_test, y_train, y_test = preprocess_data(x, y)
    print(np.unique(y_train))
    print(np.bincount(y_train))
    evaluate_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    test4()
