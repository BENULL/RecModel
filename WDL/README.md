# 推荐系统模型之Wide&Deep

Wide&Deep 简称WDL，是2016年Google 发表得一篇论文《Wide & Deep Learning for Recommender Systems》提出的推荐框架。

Wide&Deep 模型由单层的 Wide 部分和多层的 Deep部分组成。这样的组合使得模型兼具了逻辑回归和深度神经网络的优点，能够快速处理并记忆大量历史行为特征, 并且具有强大的表达能力, 不仅在当时迅速成为业界争相应用的主流模型, 而且衍生出了大量以Wide\&Deep 模型为基础结构的混合模型, 影响力一直延续到至今。

[论文地址](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1606.07792.pdf)

## 主要思路

Wide&Deep围绕着“记忆”(Memorization)与“泛化”(Generalization)两个词展开
**记忆能力可以理解为模型直接学习并利用（exploiting）历史数据中物品或者特征的共现频率的能力**
**泛化能力可以理解为模型传递特征的相关性以及发掘（exploring）稀有特征和最终标签相关性的能力**

**Wide部分通过线性模型处理历史行为特征，有利于增强模型的记忆能力，但依赖人工进行特征组合的筛选**
**Deep部分通过embedding进行学习，对特征的自动组合，挖掘数据中的潜在模式有利于增强模型的泛化能力**


## 模型结构
![请添加图片描述](https://img-blog.csdnimg.cn/f436882d44b142878eca236155f33562.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQkVOVUxM,size_20,color_FFFFFF,t_70,g_se,x_16)


### Wide部分

Wide部分善于处理稀疏类的特征，是个广义线性模型

$$
y=W^TX+b
$$

$X$特征部分包括基础特征和交叉特征，其中交叉特征可起到添加非线性的作用。

#### 交叉特征

文中提到过的特征变换为交叉乘积变换（cross-product transformation）

$$
\phi_{k}(\mathbf{x})=\prod_{i=1}^{d} x_{i}^{c_{k i}} \quad c_{k i} \in\{0,1\}
$$

$C\_{ki}$是一个布尔变量，当$i$个特征属于第$k$个特征组合时，值为1，否则为0。

例如对于特征组合"AND(gender=female, language=en)"，如果对应的特征“gender=female” 和“language=en”都符合，则对应的交叉积变换层结果才为1，否则为0

### Deep部分

Deep模型是个前馈神经网络，对稀疏特征(如ID类特征)学习一个低维稠密向量，与原始特征拼接后作为MLP输入前向传播
$$
a^{(l+1)}=f\left(W^{(l)} a^{(l)}+b^{(l)}\right)
$$

### Wide&Deep
Wide部分和Deep部分的输出进行加权求和作为最后的输出
$$
P(Y=1 \mid \mathbf{x})=\sigma\left(\mathbf{w}_{w i d e}^{T}[\mathbf{x}, \phi(\mathbf{x})]+\mathbf{w}_{d e e p}^{T} a^{\left(l_{f}\right)}+b\right)
$$
其中文中提到Wide部分和Deep部分的优化器不相同，Wide部分采用基于L1正则的FTRL而Deep部分采用AdaGrad。
其中FTRL with L1非常注重模型的稀疏性，也就是说W&D是想让Wide部分变得更加稀疏
更多相关可参考[见微知著，你真的搞懂Google的Wide&Deep模型了吗？](https://zhuanlan.zhihu.com/p/142958834)

## 实现

### TensorFlows调用

Wide&Deep模型可以直接通过Tensorflow进行调用

[Tensorflow文档](https://www.tensorflow.org/api_docs/python/tf/keras/experimental/WideDeepModel?hl=en#args_1)

```python
tf.keras.experimental.WideDeepModel(
    linear_model, dnn_model, activation=None, **kwargs
)
```

#### Example:

```python
linear_model = LinearModel()
dnn_model = keras.Sequential([keras.layers.Dense(units=64),
                             keras.layers.Dense(units=1)])
combined_model = WideDeepModel(linear_model, dnn_model)
combined_model.compile(optimizer=['sgd', 'adam'], 'mse', ['mse'])
# define dnn_inputs and linear_inputs as separate numpy arrays or
# a single numpy array if dnn_inputs is same as linear_inputs.
combined_model.fit([linear_inputs, dnn_inputs], y, epochs)
# or define a single `tf.data.Dataset` that contains a single tensor or
# separate tensors for dnn_inputs and linear_inputs.
dataset = tf.data.Dataset.from_tensors(([linear_inputs, dnn_inputs], y))
combined_model.fit(dataset, epochs)
```

### 自己实践

#### Model

```python
class WideDeep(Model):

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64),
                 l2_linear_reg=1e-6, dnn_activation='relu', dnn_dropout=0., ):
        """
        Wide&Deep
        :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
        :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        :param dnn_hidden_units: A list. Neural network hidden units.
        :param dnn_activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param l2_linear_reg: A scalar. The regularizer of Linear.
        """
        super(WideDeep, self).__init__()
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.wide_dense_feature = DenseFeatures(self.linear_feature_columns)
        self.deep_dense_feature = DenseFeatures(self.dnn_feature_columns)
        self.linear = Linear(use_bias=True, l2_linear_reg=l2_linear_reg)
        self.dnn_network = DNN(dnn_hidden_units, dnn_activation, dnn_dropout)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        # Wide
        wide_out = self.wide_dense_feature(inputs)
        # wide_out = self.linear(wide_inputs)
        # Deep
        deep_inputs = self.deep_dense_feature(inputs)
        deep_out = self.dnn_network(deep_inputs)
        both = concatenate([deep_out, wide_out])
        # Out
        outputs = self.output_layer(both)
        return outputs
```

#### Modules

```python
class Linear(Layer):
    def __init__(self, use_bias=False, l2_linear_reg=1e-6):
        super(Linear, self).__init__()
        self.use_bias = use_bias
        self.l2_reg_linear = l2_linear_reg

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        self.w = self.add_weight(name="w",
                                 shape=(int(input_shape[-1]), 1),
                                 regularizer=l2(self.l2_reg_linear),
                                 trainable=True)
        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        linear_logit = tf.tensordot(inputs, self.w, axes=(-1, 0))
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit


class DNN(Layer):

    def __init__(self, dnn_hidden_units, dnn_activation='relu', dnn_dropout=0.):
        """
        Deep Neural Network
        :param dnn_hidden_units: A list. Neural network hidden units.
        :param dnn_activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout number.
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=dnn_activation) for unit in dnn_hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x
```

## 参考

- 深度学习推荐系统-王喆
- [Wide&Deep模型的深入理解](https://mp.weixin.qq.com/s/LRghf8mj1hjUYri_m3AzBg)
- [推荐系统之wide&deep](https://blog.csdn.net/qq_40778406/article/details/90740399)
- [见微知著，你真的搞懂Google的Wide&Deep模型了吗？](https://zhuanlan.zhihu.com/p/142958834)
- [WDL 代码](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/WDL)