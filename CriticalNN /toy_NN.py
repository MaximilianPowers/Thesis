import numpy as np
import matplotlib.pyplot as plt


class FC():
    def __init__(self, d, step_size) -> None:
        self.d = d
        self.a_1 = np.random.normal(0,.5/self.d)
        self.a_2 = np.random.normal(0,.5/self.d)
        self.bias = 0

        # self.tol = 1e-8
        # self.lambda_ = np.random.poisson(lam = 1) + 1 + self.tol
        self.lambda_ = 1.5
        self.step_size = step_size

    def gen_data(self, N):
        e = np.zeros((N, self.d))
        perm = np.random.randint(0, self.d, N)
        e[np.arange(0, N), perm] = 1

        y = 2*np.random.randint(0, 2, (N, 1)) - 1
        x = self.lambda_*(np.ones_like(e)*y*e) + np.random.normal(0, 1, (N, self.d))
        return x, y.squeeze()

    def f(self, x1, x2):
        return self.a_1 * x1 + self.a_2 * x2

    def loss(self, y_pred, y_true):
        return np.sum(logi(y_pred*y_true))


def logi(z):
    return np.log(1 + np.exp(-z)) 

def ReLU(x):
    return np.maximum(x, 0)


def dReLU(x):
    return np.where(x <= 0, 0, 1)

def train_SGD(N, d, step_size, run=""):

    n_epochs = int(10 // step_size)
    model = FC(d, step_size)
    xTr, yTr = model.gen_data(N)
    xTe, yTe = model.gen_data(N // 5)
    loss_profile = []
    store_bias = []
    store_a_1 = []
    store_a_2 = []
    store = []
    for i in range(n_epochs):
        indx = np.random.randint(N)
        xTr_indx, yTr_indx = xTr[indx], yTr[indx]
        x1 = np.sum(ReLU(-xTr_indx + model.bias))
        x2 = np.sum(ReLU(xTr_indx + model.bias))
        y_pred = model.f(x1, x2)
        loss = model.loss(y_pred, yTr_indx)
        tmp = np.exp(-yTr_indx*(model.a_1*x1 + model.a_2*x2))
        model.bias += step_size * np.sum(tmp * yTr_indx  * ( model.a_1*np.sum(dReLU(-xTr_indx + model.bias)) + model.a_2*np.sum(dReLU(xTr_indx + model.bias)))/(1 + tmp))
        model.a_1 += step_size * np.sum(tmp * yTr_indx * x1/(1 +tmp) )
        model.a_2 += step_size * np.sum(tmp * yTr_indx * x2/(1 +tmp) )        
        loss_profile.append(loss)
        store_a_1.append(model.a_1)
        store_a_2.append(model.a_2)
        store.append(d*(model.a_2 + model.a_1))
        store_bias.append(model.bias)
    x1 = np.sum(ReLU(-xTr + model.bias), axis=1)
    x2 = np.sum(ReLU(xTr + model.bias), axis=1)
    y_pred = np.sign(model.f(x1, x2)) 
    train_err = np.sum(np.where(yTr*y_pred == 1, 0, 1))/len(yTr)
    print(f"Training Error: {train_err:.4f}")
    x1 = np.sum(ReLU(-xTe + model.bias), axis=1)
    x2 = np.sum(ReLU(xTe + model.bias), axis=1)
    y_pred = np.sign(model.f(x1, x2)) 
    test_err = np.sum(np.where(yTe*y_pred == 1, 0, 1))/len(yTe)
    print(f"Testing Error: {test_err:.4f}")

    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(np.linspace(0,100, len(store)), loss_profile)
    ax[0,0].set_title("Loss")
    ax[0,1].plot(np.linspace(0,100,len(store)), store_a_1)
    ax[0,1].set_title("$a_1$")
    ax[1,0].plot(np.linspace(0,100,len(store)), store_a_1)
    ax[1,0].set_title("$a_2$")
    ax[1,1].plot(np.linspace(0,100,len(store)), store_bias)
    ax[1,1].set_title("$b$")
    fig.savefig("Figures/NNs/SGD_" + run + "_profile.png")
    plt.figure()
    plt.plot(np.linspace(0,100,len(store)), store)
    plt.title("$d \cdot (a_2 + a_1)$")
    plt.savefig("Figures/NNs/SGD_"+run+"_diff.png")

def train_single(N, d, step_size, run=""):
    n_epochs = int(10 // step_size)
    model = FC(d, step_size)
    xTr, yTr = model.gen_data(N)
    xTe, yTe = model.gen_data(N // 5)
    loss_profile = []
    store_bias = []
    store_a_1 = []
    store_a_2 = []
    store = []
    for i in range(n_epochs):
        indx = i % N
        xTr_indx, yTr_indx = xTr[indx], yTr[indx]
        x1 = np.sum(ReLU(-xTr_indx + model.bias))
        x2 = np.sum(ReLU(xTr_indx + model.bias))
        y_pred = model.f(x1, x2)
        loss = model.loss(y_pred, yTr_indx)
        tmp = np.exp(-yTr_indx*(model.a_1*x1 + model.a_2*x2))
        model.bias += step_size * np.sum(tmp * yTr_indx  * ( model.a_1*np.sum(dReLU(-xTr_indx + model.bias)) + model.a_2*np.sum(dReLU(xTr_indx + model.bias)))/(1 + tmp))
        model.a_1 += step_size * np.sum(tmp * yTr_indx * x1/(1 +tmp) )
        model.a_2 += step_size * np.sum(tmp * yTr_indx * x2/(1 +tmp) )        
        loss_profile.append(loss)
        store_a_1.append(model.a_1)
        store_a_2.append(model.a_2)
        store.append(d*(model.a_2 + model.a_1))
        store_bias.append(model.bias)

    x1 = np.sum(ReLU(-xTr + model.bias), axis=1)
    x2 = np.sum(ReLU(xTr + model.bias), axis=1)
    y_pred = np.sign(model.f(x1, x2)) 
    train_err = np.sum(np.where(yTr*y_pred == 1, 0, 1))/len(yTr)
    print(f"Training Error: {train_err:.4f}")
    x1 = np.sum(ReLU(-xTe + model.bias), axis=1)
    x2 = np.sum(ReLU(xTe + model.bias), axis=1)
    y_pred = np.sign(model.f(x1, x2)) 
    test_err = np.sum(np.where(yTe*y_pred == 1, 0, 1))/len(yTe)
    print(f"Testing Error: {test_err:.4f}")

    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(np.linspace(0,100, len(store)), loss_profile)
    ax[0,0].set_title("Loss")
    ax[0,1].plot(np.linspace(0,100,len(store)), store_a_1)
    ax[0,1].set_title("$a_1$")
    ax[1,0].plot(np.linspace(0,100,len(store)), store_a_1)
    ax[1,0].set_title("$a_2$")
    ax[1,1].plot(np.linspace(0,100,len(store)), store_bias)
    ax[1,1].set_title("$b$")
    fig.savefig("Figures/NNs/" + run + "_profile.png")

    plt.figure()
    plt.plot(np.linspace(0,100,len(store)), store)
    plt.title("$d \cdot (a_2 + a_1)$")
    plt.savefig("Figures/NNs/"+run+"_diff.png")
train_SGD(300, 200, 2.5*(10**(-5)), run="tiny")
train_SGD(300, 200, 2.5*(10**(-4)), run="medium")
train_SGD(300, 200, 2.5*(10**(-3)), run="big")