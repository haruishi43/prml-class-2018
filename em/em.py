import numpy as np
import matplotlib.pyplot as plt

# 2次元のガウス分布を3つ作成
def Toydata2D(n_data):
    x1 = np.random.normal(0,1,2*n_data).reshape(-1,2)
    x2 = np.random.normal(0,1,2*n_data).reshape(-1,2)
    x3 = np.random.normal(0,1,2*n_data).reshape(-1,2)
    x1 += np.array([-3,3])
    x2 += np.array([3,3])
    x3 += np.array([0,-3])
    return np.concatenate([x1, x2, x3], axis=0)

class GMM(object):
    def __init__(self, K):
        self.K = K  # クラス数

    def fit(self, X, iteration):
        self.dim = X.shape[1]   # 次元
        self.lam = np.ones(self.K)/self.K    # 重み初期値1/3
        self.mu = np.random.uniform(X.min(), X.max(), (self.dim, self.K))   # 平均
        self.sig = np.repeat(10 * np.eye(self.dim), self.K).reshape(self.dim, self.dim, self.K) # 標準偏差

        for i in range(iteration):
            Q = self.Estep(X)   # Eステップ
            self.Mstep(X, Q)    # Mステップ

    # ガウス関数
    def gauss(self, X):
        diff = X[:,:,None] - self.mu
        precisions = np.linalg.inv(self.sig.T).T
        exponents = np.sum(np.einsum('nik,ijk->njk', diff, precisions) * diff, axis=1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.sig.T).T * (2 * np.pi) ** self.dim)

    def Estep(self, X):
        Q = self.lam * self.gauss(X)
        Q /= Q.sum(axis=-1, keepdims=True)
        return Q

    def Mstep(self, X, Q):
        Nk = np.sum(Q, axis=0)  # Qの和
        self.mu = X.T.dot(Q) / Nk   # 平均の更新
        self.lam = Nk / X.shape[0]  # 重みの更新
        diff = X[:, :, None] - self.mu
        # 標準偏差の更新
        self.sig = np.einsum('nik,njk->ijk', diff, diff * np.expand_dims(Q, 1)) / Nk

    def classify(self, X):   # 分類
        pk = self.lam * self.gauss(X)
        return np.argmax(pk, axis=1)

def main():
    # データ作成
    X = Toydata2D(100)
    n_iter = 100

    # モデル定義
    model = GMM(3)
    # フィット
    model.fit(X, n_iter)
    # 分類結果
    labels = model.classify(X)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # input
    ax[0].scatter(X[:,0], X[:,1])
    ax[0].set_xlim(-7, 7)
    ax[0].set_ylim(-7, 7)
    ax[0].set_title("input")
    # result
    colors = ["red", "blue", "green"]
    ax[1].scatter(X[:, 0], X[:, 1], c=[colors[int(label)] for label in labels])
    ax[1].set_xlim(-7, 7)
    ax[1].set_ylim(-7, 7)
    ax[1].set_title("result")

    filename = "output.png"
    fig.savefig(filename)

if __name__ == "__main__":
    main()