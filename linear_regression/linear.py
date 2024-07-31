import os
from sklearn.linear_model import LinearRegression
import pickle

def linear_model_predict(X_train, y_train, X_test):
    """
     创建模型并保存，然后进行预测
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    # 指定模型保存路径
    model_path = 'model/linear_model.pkl'
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # 保存模型
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    print("Linear Regression Model coef,intercept: ", model.coef_,model.intercept_)
    y_pred = model.predict(X_test)
    return y_pred

def load_linear_model_predict(X_test):
    """
    加载模型并预测
    """
    model = LinearRegression()
    # 打开文件并加载模型
    with open('model/linear_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Linear Regression Model coef,intercept: ", model.coef_,model.intercept_)
    y_pred = model.predict(X_test)
    return y_pred