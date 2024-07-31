import unittest
import sys
import os

# 添加模块的路径
module_path = os.path.abspath('.')
if module_path not in sys.path:
    sys.path.append(module_path)

from linear_regression import linear

class TestLinearRegression(unittest.TestCase):
    def test_linear_regression(self):
    # 对简单房价特征数据集进行线性回归
    #        卧室数量      建筑面积(平方米)   车库数量    年龄(年) 位置评分    房价(万元)
    # 样本1       3              120         2       10         85        120
    # 样本2       2              85          1       15         75        80
    # 样本3       4              150         2       5          90        180
    # 样本4       3              110         1       8          80        110
    # 样本5       2              90          0       20         70        75
        X_train = [[3,120,2,10,85],
                   [2,85,1,15,75],
                   [4,150,2,5,90],
                   [3,110,1,8,80],
                   [2,90,0,20,70],
                   [2,90,2,20,90]
                   ]
        
        y_train = [120, 80, 180, 110,75,190]
        X_test = [[3,120,2,10,90]]
        # predictions = linear.linear_model_predict(X_train, y_train, X_test)
        predictions = linear.load_linear_model_predict( X_test)
        print(predictions)

if __name__ == '__main__':
    unittest.main()