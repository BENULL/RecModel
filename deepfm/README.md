# 推荐系统模型之DeepFM
[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)是华为和哈工大在2017发表的论文，在Wide&Deep结构的基础上，使用FM取代Wide部分的LR，不需要再做复杂的特征工程，可以直接把原始数据输入模型。

## 主要思路
DeepFM=DNN+FM

因子分解机(Factorization Machines,FM)，具有自动学习交叉特征的能力，避免了Wide & Deep模型中浅层部分人工特征工程的工作，通过对每一维特征的隐变量内积来提取特征。理论上FM可以对比二阶更高阶的特征组合进行建模，实际上由于计算复杂度的原因，一般只用到二阶特征组合。

而对于高阶的特征组合，很自然想到利用DNN。但是稀疏的One-Hot特征会导致网络参数过多，通过将特征分为不同的field送入dense层，得到高阶特征的组合。

最终将低阶组合特征单独建模，然后融合高阶组合特征，就是DeepFM了。

## 模型结构

![请添加图片描述](https://img-blog.csdnimg.cn/05d9a01770f24a068523c0397baa8299.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQkVOVUxM,size_20,color_FFFFFF,t_70,g_se,x_16)


这个模型分为**FM部分和Deep部分**，和Wide & Deep模型不同的是，DeepFM两部分共享原始输入特征

![latexs](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D%3D%5Coperatorname%7Bsigmoid%7D%5Cleft%28y_%7BF%20M%7D&plus;y_%7BD%20N%20N%7D%5Cright%29)

在输入特征部分，由于原始特征向量大多是高度稀疏的连续和类别混合的分域特征，为了更好的发挥DNN模型学习高阶特征的能力，文中设计了一套子网络结构，将原始的稀疏表示特征映射为稠密的特征向量。

子网络设计时的两个要点：

1. 不同field特征长度可以不同，但是子网络输出embedding向量需具有相同维度k；
2. 利用FM模型的隐特征向量V作为网络权重初始化来获得子网络输出的embedding向量

![请添加图片描述](https://img-blog.csdnimg.cn/9d6c863a17324cb7a33940add5aaf43b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQkVOVUxM,size_20,color_FFFFFF,t_70,g_se,x_16)


这里要注意的一点是，在一些其他DNN做CTR预估的论文当中，会使用预训练的FM模型来进行Deep部分的向量初始化。文中的做法不同，它不是使用训练好的FM来进行初始化，而是和FM模型的部分共享同样的V，将FM和DNN进行整体联合训练，从而实现了一个端到端的模型。

这样做有两个好处

1. 它可以同时学习到低维以及高维的特征交叉信息，预训练的FM来进行向量初始化得到的embedding当中可能只包含了二维交叉的信息。
2. 这样可以避免像是Wide & Deep那样多余的特征工程。


### FM (Factorization Machines)

FM主要是解决稀疏数据下的特征组合问题，并且其预测的复杂度是线性的，对于连续和离散特征有较好的通用性。
[FM论文地址](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
下面是FM二阶部分的数学形式。FM为每个特征学习了一个隐权重向量。在特征交叉时，使用两个向量的内积![latex](https://latex.codecogs.com/gif.latex?\left\langle&space;V_{i},&space;V_{j}\right\rangle)作为交叉特征的权重。

![latex](https://latex.codecogs.com/gif.latex?y_{F&space;M}=\langle&space;w,&space;x\rangle&plus;\sum_{j_{1}=1}^{d}&space;\sum_{j_{2}=j_{1}&plus;1}^{d}\left\langle&space;V_{i},&space;V_{j}\right\rangle&space;x_{j_{1}}&space;\cdot&space;x_{j_{2}})

本质上,FM引入隐向量的做法，与矩阵分解用隐向量代表用户和物品的做法异曲同工。可以说，FM是将矩阵分解隐向量的思想进行了进一步扩展，从单纯的用户、物品隐向量扩展到了所有特征上。

在工程方面,FM同样可以用梯度下降法进行学习，使其不失实时性和灵活性。相比之后深度学习模型复杂的网络结构导致难以部署和线上服务，FM较容易实现的模型结构使其线上推断的过程相对简单，也更容易进行线上部署和服务。因此，FM在2012一2014年前后，成为业界主流的推荐模型之一。

**FM模型优势**

- 在高度稀疏的情况下特征之间的交叉仍然能够估计，而且可以泛化到未被观察的交叉

- 参数的学习和模型的预测的时间复杂度是线性的

FM更详细介绍见[FM（Factorization Machines）的理论与实践](https://zhuanlan.zhihu.com/p/50426292)

#### FM部分

![请添加图片描述](https://img-blog.csdnimg.cn/7bbf27202d574683a96076bdc8c46d02.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQkVOVUxM,size_20,color_FFFFFF,t_70,g_se,x_16)


在实践中，FM模块最终是将一阶项与二阶项进行了简单concat。

上图中的Field为特征组，例如性别属性可以看做是一个Field，其有两个特征分别为“男”、“女”。通常来说一个Field中往往只有一个非零特征，但也有可能为多值Field，需要根据实际输入进行适配。

上图的图例中展示了三种颜色的线条，其中绿色的箭头表示为特征的Embedding过程，即得到特征对应的Embedding vector，通常使用![](https://latex.codecogs.com/gif.latex?v_ix_i)来表示，而其中的隐向量v,则是通过模型学习得到的参数。红色箭头表示权重为1的连接，也就是说红色箭头并不是需要学习的参数。而黑色连线则表示为正常的，需要模型学习的参数w。
## Deep

Deep部分就是经典的**前馈网络DNN**，用来学习特征之间的高维交叉。
![请添加图片描述](https://img-blog.csdnimg.cn/d0349d46fcef422c9be178b89055fd0e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQkVOVUxM,size_20,color_FFFFFF,t_70,g_se,x_16)
## 实现

**[Github仓库地址](https://github.com/BENULL/RecModel/tree/master/deepfm)**

### Model

```python
class DeepFM(Model):

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64),
                 l2_linear_reg=1e-6, dnn_activation='relu', dnn_dropout=0., ):
        super(DeepFM, self).__init__()
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.embedding = DenseFeatures(self.linear_feature_columns)
        self.dense_feature = DenseFeatures(self.dnn_feature_columns)
        self.linear = Linear(use_bias=True, l2_linear_reg=l2_linear_reg)
        self.reshape = Reshape((len(self.linear_feature_columns), -1))
        self.fm = FM()
        self.dnn_network = DNN(dnn_hidden_units, dnn_activation, dnn_dropout)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        # first order term
        embeddings = self.embedding(inputs)
        first_order = self.linear(embeddings)

        # second order term
        embed_inputs = self.reshape(embeddings)
        second_order = self.fm(embed_inputs)

        # dnn term
        dnn_inputs = self.dense_feature(inputs)
        dnn_inputs = concatenate([embeddings, dnn_inputs])
        dnn_out = self.dnn_network(dnn_inputs)

        # out
        both = concatenate([first_order, second_order, dnn_out])
        outputs = self.output_layer(both)
        return outputs
```

### Modules

```python
class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)
        return cross_term

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return None, 1


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
        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        linear_logit = tf.reduce_sum(inputs, axis=-1, keepdims=True)
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
- 深度学习推荐系统——王喆
- [深入浅出DeepFM](https://zhuanlan.zhihu.com/p/248895172)
- [深度推荐模型之DeepFM](https://zhuanlan.zhihu.com/p/57873613)
- [吃透论文——推荐算法不可不看的DeepFM模型](https://www.cnblogs.com/techflow/p/14260630.html)
- [Recommender-System-with-TF2.0](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/DeepFM)
- DeepCTR