import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def DecisionBoundaryDisplay(datasets : dict, classifiers : dict):
    """
    DecisionBoundaryDisplay(datasets : dict, classifiers : dict)
        根据所提供的数据集和分类器，可视化决策边界
    
    Parameters
    ----------
    datasets : dict
        用字典存储的数据集，`key`代表数据集名称，`value`代表数据集`(X,y)`
    classfiers : dict
        用字典存储的分类器方法，`key`代表分类器名称，`value`代表分类器实体
    """
    def plot_boundary(cls, xrange : list, yrange : list, ax : 'axes', cmap : 'colormap'):
        """
        plot_boundary(cls, xrange : list, yrange : list, ax, cmap)
            可视化决策边界
        
        Parameters
        ----------
        cls : Any
            训练好的分类器
        xrange : list
            x轴的范围
        yrange : list
            y轴的范围
        ax : axes
            matplotlib的axes实例
        cmap : colormap
            配置绘图颜色
        """
        # 创建网格，并拼接为 data
        X = np.linspace(xrange[0],xrange[1],num=20)
        Y = np.linspace(yrange[0],yrange[1],num=20)
        X, Y = np.meshgrid(X,Y)
        data = np.vstack([X.flatten(),Y.flatten()]).T
                
        # 对分类概率做出预测
        prob = cls.predict_proba(data)[:,1]
        # 改变形状
        prob = np.reshape(prob,(20,20))
        
        # 用等高线图绘制决策边界
        ax.contourf(X,Y,prob,cmap=cmap,alpha=0.7)
        
    # 数据集数量，分类器数量
    n, m = len(datasets), len(classifiers)
    # 初始化画布
    fig = plt.figure(figsize=(3*(m + 1),3*n),dpi=80)
    
    # 设置颜色为深红和深蓝
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    
    # 在不同的数据集上迭代
    for i, data_name in enumerate(datasets.keys()):
        # 获取当前数据集
        X, y = datasets[data_name]
        # 对特征做标准化处理
        X = StandardScaler().fit_transform(X)
        # 做一个平移变换，保证特征为正
        X[:,0] = X[:,0] - X[:,0].min()
        X[:,1] = X[:,1] - X[:,1].min()
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=0)
        
        # 确定坐标轴范围
        x1_min, x1_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
        x2_min, x2_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

        
        # 绘制数据集
        ax = plt.subplot(n,m+1,i*(m+1) + 1)
        ax.scatter(X_train[:,0], X_train[:,1], c=y_train, alpha=0.8, cmap = cm, label="train")
        ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap = cm_bright, alpha=0.5, edgecolors="k", label="test")
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_ylabel(data_name,fontsize=16)
        if i == 0:
            ax.set_title("Datasets",fontsize=16)
        
        # 在不同分类器上迭代
        for j, cls_name in enumerate(classifiers.keys()):
            # 获取当前分类器
            cls = classifiers[cls_name]
            # 选择画布位置
            ax = plt.subplot(n,m+1,i*(m+1) + 1 + j + 1)
            # 训练模型
            cls.fit(X_train, y_train)
            y_pred = cls.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # 绘制决策边界
            plot_boundary(cls, xrange=[x1_min,x1_max], yrange=[x2_min,x2_max], ax=ax, cmap=cm)
            
            # 绘制训练点和测试点
            ax.scatter(X_train[:,0], X_train[:,1], c=y_train, alpha=0.8, cmap = cm)
            ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap = cm_bright, alpha=0.5, edgecolors="k")
            ax.set_xlim(x1_min, x1_max)
            ax.set_ylim(x2_min, x2_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if i == 0:
                ax.set_title(cls_name,fontsize=16)
            # 展示正确率
            ax.text(
                x1_max - 0.3,
                x2_min + 0.3,
                ("%.2f" % acc).lstrip("0"),
                size=16,
                horizontalalignment="right",
            )
            
    plt.tight_layout()