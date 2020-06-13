# -*- coding: utf-8 -*-
# heqy write at 20190902
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


trainPath = "./data/iris_training.csv"
testPath = "./data/iris_test.csv"


def parse_csv(line):
    # 设置特征和标签的数据接收格式
    featlab_types = [[0.], [0.], [0.], [0.], [0]]
    # 解析csv数据，以featlab_types的格式接收
    parsed_line = tf.io.decode_csv(line, featlab_types)
    # 提取出特征数据，并转化成张量
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    # 提取出标签数据，并转化成张量
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label


def getFeaturesLables(dataPath):
    # 使用TextLineDataset 读取文件内容
    FeatLabs = tf.data.TextLineDataset(trainPath)
    # 跳过第一行，因为第一行是所有数据内容的总结，不能用于训练或测试
    FeatLabs = FeatLabs.skip(1)
    # 把每一行数据按照parse_csv的格式报错
    FeatLabs = FeatLabs.map(parse_csv)
    # 打乱数据原来的存放位置，
    FeatLabs = FeatLabs.shuffle(buffer_size=1000)
    # 以float32的格式保存数据
    FeatLabs = FeatLabs.batch(32)
    return FeatLabs


def readCategory():
    """
    Args:
        None
    Returns:
        categories: a list of labels
        cat_to_id: a dict of label to id
    """

    categories = ['山鸢尾setosa', '变色鸢尾versicolor', '维吉尼亚鸢尾virginica']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


class NNetConfig():
    num_classes = 3  # 多分类的种类
    num_epochs = 161  # 训练总批次
    print_per_epoch = 20  # 每训练多少批次时打印训练损失函数值和预测准确率值
    layersls = [4, 10, 20, 10, 3]  # 【输入，隐藏各层节点数，输出】
    learning_rate = 0.01  # 网络学习率
    train_filename = './data/iris_training.csv'  # 训练数据
    test_filename = './data/iris_test.csv'  # 测试数据
    best_model_savepath = "./dnn/best_validation"   # 最好模型的存放文件夹


class NNet(object):
    def __init__(self, config):
        self.config = config
        self.layersls = config.layersls
        self.NNet()

    def NNet(self): # 根据给出的输入输出及每层网络的节点数搭建深度学习网络
        """
        根据给出的输入输出及每层网络的节点数搭建深度学习网络
        :param layersls: 【输入，隐藏各层节点数，输出】
        :return:
        """
        model = tf.keras.Sequential()
        for i in range(len(config.layersls)):
            if(i == 0):     # 确定网络的输入和网络的第一层节点数
                model.add(tf.keras.layers.Dense(config.layersls[1],
                                                activation="relu",
                                                input_shape=(config.layersls[0],)))
                i += 1
            elif(i == len(config.layersls)-1): # 确定网络的输出
                model.add(tf.keras.layers.Dense(config.layersls[i]))
            else:  # 网络的各隐藏层节点数
                model.add(tf.keras.layers.Dense(config.layersls[i],
                                                activation="relu"))
        return model


def iris_train():
    # 调用NNet网络，搭建自己的神经网络
    nnet = NNet(config)
    model = nnet.NNet()
    # 获取训练数据
    featsLabs = getFeaturesLables(trainPath)
    # 定义网络优化器：梯度下降，以learning_rate 的速率进行网络的训练优化
    optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
    # 防止网络过拟合的，当准确率大幅度下降时 停止训练，这里没有用到
    flag_train = False
    # 损失函数，使用的是交叉熵
    def loss(model, x, y):
        y_ = model(x)
        return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
    # 当前网络梯度
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return tape.gradient(loss_value, model.variables)

    best_epoch_accuracy = 0
    last_improved = 0
    improved_str = ''
    for epoch in range(config.num_epochs):
        epoch_loss_avg = tf.metrics.Mean()
        epoch_accuracy = tf.metrics.Accuracy()

        # 轮回训练网络
        for x, y in tf.data.Iterator(featsLabs):
            # 优化网络
            grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

            # 当前批次训练的损失函数均值
            epoch_loss_avg(loss(model, x, y))  #
            # 预测的标签值和实际标签值进行对比，得到当前的预测准确率
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

            # 本批次训练结束，保存本批次的损失函数结果和准确率结果
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())
            # 每隔print_per_epoch次 输出损失函数值、准确率值等信息，方便监控网络的训练
            if epoch % config.print_per_epoch == 0:
                if not (epoch_accuracy.result()) > best_epoch_accuracy:
                    # flag_train = True   #防止网络被过度训练
                    # break
                    improved_str = ''
                    pass
                else:
                    best_epoch_accuracy = epoch_accuracy.result()
                    print("当前最高准确率：%.3f:" % best_epoch_accuracy)
                    # 最好模型的保存
                    model.save(os.path.join(config.best_model_savepath,
                                            "model_best.h5"))
                    last_improved = epoch
                    improved_str = '*'

                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}， isBest:{}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result(),
                                                                    improved_str))

        if flag_train:
            break


def iris_test():
    # 加载已经训练好的最优模型（包括网络结构及网络权值矩阵）
    model = tf.keras.models.load_model(
        os.path.join(config.best_model_savepath, "model_best.h5"),
        compile=False)
    # 数据批量处理，把测试数据进行清洗、结构化、张量化
    testFeatsLabs = getFeaturesLables(testPath)
    # 计算测试数据在此模型下的准确率
    test_accuracy = tf.metrics.Accuracy()
    for x, y in tf.Iterator(testFeatsLabs):
        # 模型的预测结果
        prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("测试数据的测试结果为: {:.3%}".format(test_accuracy.result()))
    return test_accuracy.result()


def iris_prediction(features=[]):
    # 预测函数
    # 加载已经训练好的最优模型（包括网络结构及网络权值矩阵）
    # compile=False 表示对此模型，我只用不再次训练
    model = tf.keras.models.load_model(
        os.path.join(config.best_model_savepath, "model_best.h5"),
        compile=False)
    # 当预测特征为空时，使用下面给出的默认值进行预测
    if (len(features)==0):
        predFeats = tf.convert_to_tensor([
        [5.9, 3.0, 4.2, 1.5],
        [5.1, 3.3, 1.7, 0.5],
        [6.9, 3.1, 5.4, 2.1]
        ])
    else:
        # 预测特征数据转换成张量
        predFeats = tf.convert_to_tensor(features)
    # 预测结果存放列表
    cat_probs = []
    # 使用已训练模型预测的预测结果，是张量
    y_probs = model(predFeats)
    # 取出每条预测结果进行处理，取出其中最大值，即最可能的结果，
    # 根据最大值所在下标，取到cat可读文本
    for prob in y_probs:
        top1 = tf.argmax(prob).numpy()
        cat_probs.append(cat[top1])
    return cat_probs


def iris_visualize():
    #损失函数及准确率函数随进一步网络训练的变化曲线
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle("NNet Visualize", fontsize=30)
    axes[0].set_ylabel("Loss", fontsize=24)
    axes[0].set_xlabel("Epoch", fontsize=24)
    axes[0].plot(train_loss_results, linewidth='1.5', color='blue')
    axes[1].set_ylabel("Accuracy", fontsize=24)
    axes[1].set_xlabel("Epoch", fontsize=24)
    axes[1].plot(train_accuracy_results, linewidth='1', color='crimson')
    plt.savefig("./loss.png")
    plt.show()


if __name__=="__main__":
    # 调用模型参数文件
    config = NNetConfig()
    # 保存每步的网络的损失函数值及准确率值，后续训练可视化备用
    train_loss_results = []
    train_accuracy_results = []
    # 标签列表、标签和ID对照字典
    cat, cat_to_id = readCategory()
    # 开始训练
    iris_train()
    # 训练结果可视化
    # iris_visualize()
    print("开始对已训练网络进行测试：")
    s = iris_test()
    # print(s)
    s2 = iris_prediction([])
    print("预测下面3个鸢尾花的类别：")
    print([
        [5.9, 3.0, 4.2, 1.5],
        [5.1, 3.3, 1.7, 0.5],
        [6.9, 3.1, 5.4, 2.1]
        ])
    print(s2)