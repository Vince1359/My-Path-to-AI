# 逻辑回归与前馈神经网络

本篇通过将逻辑回归看做一个简单的神经网络，介绍参数初始化，前向传播，损失函数与代价函数，反向传播及参数更新等步骤，并扩展到多层的前馈神经网络。

## 内容：

- 符号约定
- 什么是逻辑回归。
- Python3代码实现逻辑回归。
- 什么是前馈神经网络。
- Python3代码实现前馈神经网络。

## 符号约定：

- $w$ 表示权重，$b$ 表示偏置。
- $x$ 表示输入特征，$y$表示输出标签。
- $z=wx+b$ 表示线性表换后的值，g表示激活函数，$a=g(z)$表示激活值。
- 小写字母表示向量，大写字母表示矩阵。
- 上标 $[l]$ 表示与第 $l$ 层相关的数值。
  - 例如：$a^{[L]}$ 表示第 $L$ 层的激活值， $W^{[L]}$ 和 $b^{[L]}$ 是第 $L$ 层的参数。
- 上标 $(i)$ 表示与第 $i$ 个样例相关的数值。
  - 例如: $x^{(i)}$ 是第 $i$ 个训练样例。
- 下标 $i$ 表示一个向量中的第 $i$ 维的数值。
  - 例如: $a^{[l]}_i$ 表示第 $l$ 层的激活值中的第 $i$ 维的数值。

## 什么是逻辑回归：

## 逻辑回归实现步骤：

- 建立完整的逻辑回归算法流程：
  - 确立模型的结构，如输入的维度。
  - 初始化模型的参数（权重W和偏置b）。
  - 训练循环：
    - 计算当前损失函数值（前向传播）。
    - 计算当前梯度（反向传播）。
    - 更新参数（梯度下降法）。
- 用函数实现上述算法流程中各部分的功能，并最后整合到一个model函数中。

**逻辑回归核心数学表达式**：

对于一个训练例子 $x^{(i)}$：

前向传播：
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$

计算损失函数：
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

计算代价函数：
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{4}$$

反向传播：
$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{5}$$
$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{6}$$

## Python代码实现逻辑回归

- 实现sigmoi函数：$sigmoid(Z) = \frac{1}{1 + e^{-(Z)}}$

```python
def sigmoid(z):
    """
    计算sigmoid(z)的值。
    :param z: 一个标量或任意numpy数组。
    :return: sigmoid(z)
    """
    return 1 / (1 + np.exp(-z))
```

- 实现逻辑回归的参数初始化

```python
def initialize_logistic(dim):
    """
    该函数创建一个形状为(dim, 1)的numpy向量，代表参数w，将偏置b初始化为0。
    :param dim: 输入训练样本的维度。
    :return w: 初始化后的参数w，w.shape = (dim, 1)
    :return b: 初始化后的偏置b，b为标量0。
    """
    w = np.zeros((dim, 1))
    b = 0
    return w, b
```

- 实现前向传播与反向传播

```python
def propagate_logistic(w, b, X, Y):
    """
    实现公式(1)~(6)
    :param w: 一个维度为(dim, 1)的numpy数组。
    :param b: 一个标量b，表示偏置。
    :param X: 一个维度为(dim, m)的训练样本矩阵。
    :param Y: 一个维度为(1, m)的训练样本的标签向量。
    :return grads: grads['dw']: 损失函数对w的梯度，与w的维度相同。
                   grads['db']: 损失函数对b的梯度，与b的维度相同。
    :return cost: 代价函数的值。
    """
    # 获得训练样本的总数
    m = X.shape[1]

    # 前向传播（求得代价函数）
    A = sigmoid(np.dot(w.T, X) + b)                                    # 公式(1)(2)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))    # 公式(4)

    # 反向传播（求得梯度）
    dw = 1 / m * np.dot(X, (A - Y).T)     # 公式(5)
    db = 1 / m * np.sum(A - Y)            # 公式(6)

    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}

    return grads, cost
```

- 实现参数的优化

```python
def optimize_logistic(w, b, X, Y, num_iterations, learning_rate, print_cost=True):
    """
    该函数使用梯度下降法优化参数w和偏置b的值。
    :param w: 一个维度为(dim, 1)的numpy数组。
    :param b: 一个标量b，表示偏置。
    :param X: 一个维度为(dim, m)的训练样本矩阵。
    :param Y: 一个维度为(1, m)的训练样本的标签向量。
    :param num_iterations: 优化算法要执行的循环数。
    :param learning_rate: 学习率。
    :param print_cost: 如果为True，则每个100个循环打印一次代价函数的值。
    :return parameter: parameter['w']: 优化完成后的参数w。
                       parameter['b']: 优化完成后的偏置b。
    :return grads: grads['dw']: 损失函数对w的梯度，与w的维度相同。
                   grads['db']: 损失函数对b的梯度，与b的维度相同。
    :return costs: 优化过程中计算得到的代价函数的值的列表，会被用来打印学习曲线。
    """
    costs = []
    for i in range(num_iterations):
        # 调用上面实现的propagate_logistic函数计算代价函数和梯度
        grads, cost = propagate_logistic(w, b, X, Y)

        # 获得梯度dw和db
        dw = grads["dw"]
        db = grads["db"]

        # 使用梯度下降法更新参数w和偏置b
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 每100个循环记录一次代价函数的值
        if i % 100 == 0:
            costs.append(cost)

        # 每100个循环打印一次代价函数的值
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs
```

- 实现将模型用于预测的predict函数

```python
def predict_logistic(w, b, X):
    """
    使用优化后的参数w和偏置b来预测输入的标签是多少。
    :param w: 训练后的参数w，维度为(dim, 1)的numpy数组。
    :param b: 训练后的偏置b，是一个标量。
    :param X: 待预测的样本矩阵，维度为(dim, 待测样本数量)。
    :return Y_prediction: 包含X中所有样本的标签预测值的维度为(1, 待测样本数量)的向量。
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # 将A[0, i]所表示的概率转化成实际的标签值p[0,i]（概率大于0.5时标签为1，否则标签为0）
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction
```

- 将上述各功能整合到一个model函数中

```python
def model_logistic(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5， print_cost=False):
    """
    通过调用上面实现的以_logistic结尾的函数，构建逻辑回归模型
    :param X_train: 训练样本，一个维度为(dim, m_train)的numpy数组。
    :param Y_train: 训练标签，一个维度为(1, m_train)的numpy数组。
    :param X_test: 测试样本，一个维度为(dim, m_test)的numpy数组。
    :param Y_test: 测试标签，一个维度为(1, m_test)的numpy数组。
    :param num_iterations: 优化算法要执行的循环数。
    :param learning_rate: 学习率。
    :param print_cost: 如果为True，则每个100个循环打印一次代价函数的值。
    :return d: 一个存储模型相关信息，并包含有以下字段的字典: 'costs', 'Y_prediction_test', 'Y_prediction_train', 'w', 'b', 'learning_rate', 'num_iterations'
    """
    # 初始化参数w和偏置b
    w, b = initialize_logistic(X_train.shape[0])

    # 梯度下降法更新模型参数
    parameters, grads, costs = optimize_logistic(w, b, X_train, Y_train, num_iterations, learning_rate)

    # 获得训练完的参数w和偏置b
    w = parameters["w"]
    b = parameters["b"]

    # 预测测试集合训练集的标签
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练集和测试集的准确率
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d
```

## 什么是前馈神经网络

## 前馈神经网络实现步骤：

- 建立完整的前馈神经网络算法流程：
  - 确立模型的结构，如神经网络隐藏层数量，每层的神经元数量等。
  - 初始化模型的参数（为一个$L$层神经网络的各层权重W和偏置b赋予初始值）。
  - 前向传播模块：
    - 完成前向传播中的线性部分（得到 $Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$）。
    - 实现激活函数relu和sigmoid。
    - 将上面两个步骤合并成一个 [线性变换->激活函数] 前向传播函数（得到 $A^{[l]} = g(Z^{[l]})$，g可以是任意激活函数）。
    - 重复堆叠 [线性变换->RELU] 前向传播函数 L-1 次（从第 1 层到第 L-1 层），并在最后添加 [线性变换->SIGMOID] 前向传播函数（第 $L$ 层）。
    - 将上述各步骤整合起来得到前向传播模块。
  - 计算交叉熵代价函数。
  - 反向传播模块：
    - 完成反向传播中的线性部分。
    - 实现激活函数relu和sigmoid的反向梯度计算。
    - 将上面两个步骤合并成一个 [线性变换->激活函数] 反向传播函数。
    - 重复堆叠 [线性变换->RELU] 反向传播函数 L-1 次，并在最后添加 [线性变换->SIGMOID] 反向传播函数。
  - 更新模型参数。
- 用函数实现上述算法流程中各部分的功能，并最后整合到一个model函数中。

**前馈神经网络核心数学表达式**：

对于一批训练例子，使用向量化的方式实现，输入为 $X$：

前向传播的线性部分：
$$A^{[0]} = X\tag{7}$$
$$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}\tag{8}$$

前向传播的线性部分与激活部分结合：
$$A^{[l]} = g^{[l]}(Z^{[l]}) = g^{[l]}(W^{[l]}A^{[l-1]} +b^{[l]})\tag{9}$$
其中 $g^{[l]}$ 是第$l$层的激活函数，可以是sigmoid，relu，tanh等激活函数中的任意一个。

计算损失函数：
$$\mathcal{L}(a^{[L](i)}, y^{(i)}) =  y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)\tag{10}$$

计算代价函数：
$$J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \mathcal{L}(a^{[L](i)}, y^{(i)})\tag{11}$$

反向传播的线性部分：
$$ dW^{[l]} = \frac{\partial \mathcal{L} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} \tag{12}$$
$$ db^{[l]} = \frac{\partial \mathcal{L} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}\tag{13}$$
$$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} \tag{14}$$

反向传播的激活部分：
$$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]}) \tag{15}$$

反向传播初始化部分（计算$dA^{[L]}$）：
$$dA^{[L]} = \frac{\partial \mathcal{L}}{\partial A^{[L]}} = -\frac{Y} {A^{[L]}} - \frac{1-Y}{1-{A^{[L]}}}\tag{16}$$

更新模型参数：
$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{17}$$
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{18}$$

其中 $\alpha$ 是学习率。

## Python代码实现前馈神经网络

- 实现两个激活函数sigmoid和relu及其反向传播函数sigmoid_backward与relu_backward

```python
def sigmoid(Z):
    """
    计算sigmoid(z)的值。
    :param Z: 线性变换的值Z，可以是numpy数组或者是标量。
    :return A: sigmoid(Z)的值。
    :return cache: 记录Z的值，在反向传播的时候使用。
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    """
    实现sigmoid函数的反向传播。
    :param dA: 代价函数对A的梯度。
    :param cache: 里面保存着前向传播是记录的Z值。
    :return dZ: 代价函数对Z的梯度。
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu(Z):
    """
    计算relu(Z)的值。
    :param Z: 线性变换的值Z，可以是numpy数组或者是标量。
    :return A: relu(Z)的值。
    :return cache: 记录Z的值，在反向传播的时候使用。
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    实现relu函数的反向传播。
    :param dA: 代价函数对A的梯度。
    :param cache: 里面保存着前向传播是记录的Z值。
    :return dZ: 代价函数对Z的梯度。
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
```

- 实现前馈神经网络的初始化

```python
def initialize(layer_dims):
    """
    初始化前馈神经网络的各层参数。
    :param layer_dims: 一个包含神经网络各层分别有多少个神经元的python列表。
    :return parameters: 一个包含所有层的参数和偏置的python字典:
            parameters['Wl']: 形状为(layer_dims[l], layer_dims[l-1])的第l层参数矩阵
            parameters['bl']: 形状为(layer_dims[l], 1)的第l层偏置向量
    """
    parameters = {}
    L = len(layer_dims)   # 神经网络的层数

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters
```

- 实现前馈神经网络的前向传播模块

```python
def linear_forward(A, W, b):
    """
    实现前向传播模块中的线性部分，即公式(7)(8)。
    :param A: 前一层的激活值或者是神经网络的输入层的值。
    :param W: numpy数组，形状是(当前层神经元个数, 前一层神经元个数)的参数矩阵W。
    :param b: numpy数组，形状是(当前层神经元个数, 1)的偏置向量b。
    :return Z: numpy数组，线性变换后，传入激活函数之前的值。
    :return cache: 记录A, W, b的值，用于反向传播的计算。
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    合并前向传播模块中的线性和激活部分，即公式(9)。
    :param A_prev: 前一层的激活值或者是神经网络的输入层的值。
    :param W: numpy数组，形状是(当前层神经元个数, 前一层神经元个数)的参数矩阵W。
    :param b: numpy数组，形状是(当前层神经元个数, 1)的偏置向量b。
    :param activation: 该层的激活函数，可以选择relu或者是sigmoid。
    :return A: numpy数组，线性变换后，传入激活函数之前的值。
    :return cache: 一个元组，第一个值是线性部分的cache，第二个值是激活部分的cache，用于反向传播的计算。
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def forward(X, parameters):
    """
    实现整个前馈神经网络的前向传播。
    :param X: 一个表示输入数据的numpy数组，形状是(输入层维度, 样本的数量)。
    :param parameters: initialize函数输出的初始化参数。
    :return AL: 输出层的激活值矩阵。
    :return caches: 所有linear_activation_forward函数返回的cache所组成的列表，标号从1到L-1。
    """
    caches = []
    A = X
    L = len(parameters) // 2      # 每层都有一个W和一个b，除以2刚好是层数。

    # 实现L-1个 [线性变换->激活函数] 层，并将每层算出来的cache添加到caches列表当中。
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)

    # 实现1个 [线性变换->SIGMOID] 层，用于二分类问题，并将cache添加到caches列表的末尾。
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)]， 'sigmoid')
    caches.append(cache)

    return AL, caches
```

- 实现前馈神经网络的代价函数计算

```python
def compute_cost(AL, Y):
    """
    实现交叉熵代价函数的计算，即公式(10)(11)
    :param AL: 神经网络预测的标签概率值。
    :param Y: 真实标签值。
    :return cost: 交叉熵代价函数的值。
    """
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)      # 将numpy数组的维度变为我们想要的维度，如[[5]]会变成5。
    return cost
```

- 实现前馈神经网络的反向传播模块

```python
def linear_backward(dZ, cache):
    """
    实现反向传播的线性部分，即公式(12)(13)(14)。
    :param dZ: 代价函数对Z的梯度。
    :param cache: 当前层前向传播时保存的(A, W, b)元组。
    :return dA_prev: 代价函数对前一层激活值的梯度。
    :return dW: 代价函数对当前层W的梯度。
    :return db: 代价函数对当前层b的梯度。
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)                # 公式(12)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)   # 公式(13)
    dA_prev = np.dot(W.T, dZ)                        # 公式(14)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    合并反向传播模块中的线性部分和激活部分，即公式(15)。
    :param dA: 代价函数对A的梯度。
    :param cache: 当前层前向传播时保存的(linear_cache, activation_cache)元组。
    :param activation: 激活函数种类，可选relu或者sigmoid。
    :return dA_prev: 代价函数对前一层激活值的梯度。
    :return dW: 代价函数对当前层W的梯度。
    :return db: 代价函数对当前层b的梯度。
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def backward(AL, Y, caches):
    """
    实现整个前馈神经网络的反向传播。
    :param AL: 一个预测标签概率向量，前向传播函数forward()的输出。
    :param Y: 真实标签向量。
    :param caches: caches[l]保存着第l+1层的linear_activation_forward()函数输出的cache。
    :return grads: 一个保存着各层dA, dW和db的字典:
                   grads["dA" + str(l)]
                   grads["dW" + str(l)]
                   grads["db" + str(l)]
    """
    grads = {}
    L = len(caches) # 模型层数
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # 将Y和AL变成一样的形状

    # 初始化反向传播，即公式(16)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # 第L层（最后一层）梯度计算。
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    # 从L-1层往前遍历（序号从L-2开始）
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(grads['dA'+str(l+1)], current_cache, 'relu')

    return grads
```

- 实现前馈神经网络的参数更新

```python
def update_parameters(parameters, grads, learning_rate):
    """
    实现神经网络参数的梯度下降法优化和更新。
    :param parameters: 保存着神经网络参数的python字典。
    :param grads: backward()函数输出的反向传播的梯度值。
    :param learning_rate: 学习率。
    :return parameters: 更新后的神经网络参数。
    """
    L = len(parameters) // 2 # 神经网络的层数
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads['dW'+str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads['db'+str(l + 1)]
    return parameters
```

- 将上述功能整合到一个model函数中

```python
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    构建完整的L层前馈神经网络模型，其中L=len(layers_dims)。
    :param X: 训练样本，形状为(m_train, 样本维度)。
    :param Y: 训练标签，形状为(1, m_train)。
    :param layers_dims: 代价函数对A的梯度。
    :param learning_rate: 学习率。
    :param num_iterations: 优化算法要执行的循环数。
    :param print_cost: 如果为True，则每个100个循环打印一次代价函数的值。
    :return parameters: 神经网络经过学习之后的参数值，可以进一步用来预测。
    """
    costs = []
    parameters = initialize(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # 打印学习曲线
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
```