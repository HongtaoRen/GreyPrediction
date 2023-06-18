import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class dgm1n(object):
    """
    定义DGM(1,N,c)模型
    rel_data:相关因素序列
    sys_data:系统行为序列
    predict_step:预测步长
    background_coff:背景值系数，默认值为0.5
    """

    def __init__(self, rel_data: pd.DataFrame, sys_data: pd.DataFrame, predict_step: int = 3,
                background_coff: float = 0.5):
        self.sys_data = sys_data
        self.rel_data = rel_data
        self.data_shape = self.sys_data.shape[0] - predict_step
        self.predict_step = predict_step
        self.bgc = background_coff
        self.coff = None
        self.sim_data = [self.sys_data[0]]
        self.pred_data = []
        self.sim_cdgm_data = [self.sys_data[0]]

    def __lsm(self):
        # 定义矩阵Y
        Y = np.cumsum(self.sys_data.iloc[:self.data_shape])[1:].values.reshape((self.data_shape - 1, 1))
        self.Y = Y
        # 计算背景值
        cum_sys_data = np.cumsum(self.sys_data)
        # 计算相关序列的累加
        rel_data_cum = np.cumsum(self.rel_data[:-self.predict_step], axis=0)
        rel_data_cum = rel_data_cum.iloc[1:self.data_shape, :].values
        Z = np.cumsum(self.sys_data.iloc[:self.data_shape])[:-1]
        one = np.ones(shape=[self.data_shape - 1, 1])

        # 得到矩阵B
        B = np.column_stack((Z, rel_data_cum, one))
        self.B = B
        # 使用最小二乘求解系数
        self.coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))

    def fit(self):
        self.__lsm()
        sys_data = self.sys_data.copy().iloc[:self.data_shape]
        sys_data_cum = np.cumsum(sys_data).values
        rel_data_cum = np.cumsum(self.rel_data, axis=0)
        rel_data_cum = rel_data_cum.values

        # 原始DGM(1,N)模型

        temp_lt = [sys_data_cum[0]]
        for k in range(1, self.data_shape + self.predict_step):
            beta1 = self.coff[0]
            x = beta1 ** k * temp_lt[0] + (1 - beta1 ** (k)) / (1 - beta1) * self.coff[-1]
            s = 0
            for r in range(1, k + 1):
                m = 0
                for i in range(rel_data_cum.shape[1]):
                    m += self.coff[i + 1] * rel_data_cum[r, i]
                s += beta1 ** (k - r) * m
            x = x + s
            temp_lt.append(x)
        self.temp_lt = temp_lt
        for i in range(len(temp_lt) - 1):
            x = temp_lt[i + 1] - temp_lt[i]
            self.sim_data.append(float(x))

        list_c = []

        for k in range(1, self.data_shape):
            beta1 = self.coff[0]
            x = beta1 * sys_data_cum[k-1]
            s = 0
            for i in range(rel_data_cum.shape[1]):
                s += self.coff[i + 1] * rel_data_cum[k, i]
            x += s

            c = sys_data_cum[k]-x
            list_c.append(float(c))
        self.list_c = list_c

        return self.sim_data[:-self.predict_step]

    def predict(self):
        return self.sim_data[-self.predict_step:]

    def all_data(self):
        return self.sim_data

    def plot(self):
        plt.style.use('seaborn')
        plt.plot(np.array(self.sim_data) * 7.11, 'r-o', label='DGM')
        plt.plot([i for i in range(self.data_shape + self.predict_step-self.predict_step,
                                   self.data_shape + self.predict_step )], np.array(self.sim_data[-self.predict_step:]) * 7.11, 'g-o', label='DGM predict')
        plt.plot(self.sys_data * 7.11, 'k-.o', label='Original')
        plt.plot(np.array(self.sim_cdgm_data)*7.11,'y-o',label='CDGM')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Predict  Values')
        plt.show()
    def get_c(self):
        return self.list_c

    def cdgm(self):
        sys_data = self.sys_data.copy().iloc[:self.data_shape]
        sys_data_cum = np.cumsum(sys_data).values
        rel_data_cum = np.cumsum(self.rel_data, axis=0)
        rel_data_cum = rel_data_cum.values

        # 离散的情况,核的计算结果
        # mean_c = np.mean(self.list_c)
        # min_c = np.min(self.list_c)
        max_c = np.max(self.list_c)
        temp_lt = [sys_data_cum[0]]
        for k in range(1, self.data_shape + self.predict_step):
            beta1 = self.coff[0]
            x = beta1 ** (k) * temp_lt[0] + (1 - beta1 ** k) / (1 - beta1) * max_c
            # x = beta1 ** (k) * temp_lt[0] + (1 - beta1 ** k) / (1 - beta1) * mean_c
            # x = beta1 ** k * temp_lt[0] + (1 - beta1 ** k) / (1 - beta1) * min_c
            s = 0
            for r in range(1, k + 1):
                m = 0
                for i in range(rel_data_cum.shape[1]):
                    m += self.coff[i + 1] * rel_data_cum[r, i]
                s += beta1 ** (k - r) * m
            x = x + s
            temp_lt.append(x)

        for i in range(len(temp_lt) - 1):
            x = temp_lt[i + 1] - temp_lt[i]
            self.sim_cdgm_data.append(float(x))
        return self.sim_cdgm_data



if __name__ == '__main__':
    data = pd.read_excel('data.xlsx', sheet_name='Sheet1', header=None)
    # data = pd.read_excel('Total1.xlsx', sheet_name='Sheet1', header=None)
    system_data = data.iloc[:, 0]
    relevent_data = data.iloc[:, 1:]

    model = dgm1n(relevent_data, system_data, predict_step=3,)
    fit_values = model.fit()
    predict_values = model.predict()
    sim_data = model.all_data()
    sim_cdgm_data = model.cdgm()
    model.plot()
    cc = model.get_c()

    print(predict_values)
    print(model.__dict__['coff'])
    print(model.__dict__["sim_data"])
    print(sim_cdgm_data)
    print(cc)
    cc_mean = np.mean(cc)
    print("Average value of cc:", cc_mean)
