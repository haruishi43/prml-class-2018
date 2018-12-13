
import numpy as np
import matplotlib.pyplot as plt


def Toydata2D(n_data):
    '''Create 3 2D Gaussian Distribution'''
    x1 = np.random.normal(0,1,2*n_data).reshape(-1,2)
    x2 = np.random.normal(0,1,2*n_data).reshape(-1,2)
    x3 = np.random.normal(0,1,2*n_data).reshape(-1,2)
    x1 += np.array([-3,3])
    x2 += np.array([3,3])
    x3 += np.array([0,-3])
    return np.concatenate([x1, x2, x3], axis=0)


class GMM(object):
    def __init__(self, K):
        # number of class
        self.K = K

    def _Estep(self, X):
        Q = self.lam * self._gauss(X)
        Q /= Q.sum(axis=-1, keepdims=True)
        return Q

    def _Mstep(self, X, Q):

        # take the sum of Q 
        Nk = np.sum(Q, axis=0)
        self.mu = X.T.dot(Q) / Nk
        self.lam = Nk / X.shape[0]
        diff = X[:, :, None] - self.mu

        # update
        self.sig = np.einsum('nik,njk->ijk', diff, diff * np.expand_dims(Q, 1)) / Nk

    def fit(self, X, iteration):
        self.dim = X.shape[1]
        self.lam = np.ones(self.K)/self.K  # initial weight is 1/3

        # mean and variance
        self.mu = np.random.uniform(X.min(), X.max(), (self.dim, self.K))
        self.sig = np.repeat(10 * np.eye(self.dim), self.K).reshape(self.dim, self.dim, self.K)

        for i in range(iteration):
            # E and M step
            Q = self._Estep(X)
            self._Mstep(X, Q)

    def _gauss(self, X):
        '''Gauss function'''
        diff = X[:,:,None] - self.mu
        precisions = np.linalg.inv(self.sig.T).T
        exponents = np.sum(np.einsum('nik,ijk->njk', diff, precisions) * diff, axis=1)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.sig.T).T * (2 * np.pi) ** self.dim)

    def classify(self, X):
        pk = self.lam * self._gauss(X)
        return np.argmax(pk, axis=1)


if __name__ == "__main__":
    # create data
    X = Toydata2D(100)
    n_iter = 100

    # initialize model
    model = GMM(3)

    # fit the model using the data
    model.fit(X, n_iter)
    
    # classify
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
