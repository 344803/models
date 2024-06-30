# coding: utf-8
import time
import numpy as np
import sys
import random
import itertools
import graphviz
from tensorflow.keras.backend import backend as K  # 使用tensorflow.keras.backend
from tensorflow.keras.models import Sequential, model_from_json, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Reshape, Input, \
    Lambda
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import  plot_model
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.regularizers import l1, l2  # 如果需要正则化，请从这里导入
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scipy.io
from tensorflow.keras.utils import to_categorical
# 假设你的'util'模块也进行了相应的更新
from util import *  # 确保util模块与TensorFlow 2.x兼容

# 将接收信号转换成极域图像，训练CNN模型 调整是否使用高斯分布生成图像

# 设置训练数据集的路径，该路径指向一个名为"raw1000_1000_8.mat"的MAT文件
train_path = "D://DIP//自动调制分类//raw1000_1000_8.mat"

# 设置测试数据集的路径，该路径指向一个名为"raw500_1000_8.mat"的MAT文件
test_path = "D://DIP//自动调制分类//raw500_1000_8.mat"
# 设定图片转换的分辨率，这里将信号转换为36x36像素的图片
resolution = 36
# 这里resolution2也被设置为36，但在给定的代码片段中它没有被使用
# 可能是为将来的用途或备份而设置的
resolution2 = 36
# 对训练数据集的路径进行分割，以获取其中的某些信息
# 这里使用'_'作为分隔符，将路径拆分为多个部分
tmp = train_path.split('_')
# 接着，我们取拆分后数组的第三个元素（索引为2，因为索引从0开始），该元素是一个包含文件扩展名之前的文件名部分
# 然后，我们再次使用'.'作为分隔符进行拆分，以获取文件名中的SNR（信噪比）部分
tmp = tmp[2].split('.')
# 从拆分后的数组中取第一个元素（即SNR值），并赋值给变量SNR
SNR = tmp[0]
# R变量被设置为0，可能用于表示是否读取极坐标模型（但在这个代码片段中没有进一步的使用或说明）
R = 0  # Read Polar Model（注释表示读取极坐标模型，但代码中未体现此操作）
# duplicate变量被设置为0
duplicate = 0


def polar(x):       #将复数张量转化为极坐标形式          有效地处理大规模数据
    # 计算平方，对输入张量x的每个元素进行平方操作
    square = tf.square(x)
    # 计算绝对值（模长），对平方后的张量在最后一个维度（axis=2）上进行求和，然后开方得到模长
    # keepdims=True 保持输出的维度与输入一致，除了被操作的维度（axis=2）
    abs_sig = tf.sqrt(tf.reduce_sum(square, axis=2, keepdims=True))
    # 去掉最后一个维度（如果它是1的话），如果由于keepdims=True导致最后一个维度为1，则将其移除
    abs_sig = tf.squeeze(abs_sig, axis=2)
    # 检查除以零的情况，添加一个小的 epsilon 值来避免除以零的情况
    # epsilon 是 TensorFlow Keras 后端提供的一个很小的正数
    epsilon = tf.keras.backend.epsilon()
    # 假设x是一个复数信号，这里用x[:, :, 1]和x[:, :, 0]分别代表复数的虚部和实部
    # 使用虚部除以（实部 + epsilon）来避免除以零
    safe_div = x[:, :, 1] / (x[:, :, 0] + epsilon)
    # 计算角度（反正切），使用atan函数计算得到的是弧度值
    arctan = tf.math.atan(safe_div)
    # 连接模长和角度，将模长（abs_sig）和角度（arctan）沿着第一个维度（axis=1）进行拼接
    return tf.concat([abs_sig, arctan], axis=1)


def pic2gauss(sig, x0, x1, y0, y1, r0, r1, l):  #从输入的信号 sig 中生成二维高斯（或称为高斯核）的网格
    # 在x0和x1之间生成r0个等间隔的点，形成一维数组
    linx = np.linspace(x0, x1, r0)
    # 在y0和y1之间生成r1个等间隔的点，形成一维数组
    liny = np.linspace(y0, y1, r1)
    # 假设sig的前l个通道代表x坐标，后l个通道代表y坐标
    # 将sig的前l个通道reshape为(-1, l, 1)的形状，其中-1表示自动计算该维度的大小
    Px = tf.reshape(sig[:, :l], (-1, l, 1))
    # 将sig的后l个通道reshape为(-1, l, 1)的形状
    Py = tf.reshape(sig[:, l:], (-1, l, 1))
    # 计算Px与linx之间的差的平方，并reshape为(-1, l, r0, 1)的形状
    tmpx = tf.reshape(tf.square(Px - linx), (-1, l, r0, 1))
    # 计算Py与liny之间的差的平方，并reshape为(-1, l, 1, r1)的形状
    tmpy = tf.reshape(tf.square(Py - liny), (-1, l, 1, r1))
    # 将tmpx和tmpy相加，得到二维高斯函数的指数部分（平方和）
    tmp = tmpx + tmpy
    # 计算二维高斯函数值，除以2除以方差（这里假设方差为0.05），然后取指数，并对l维度求和
    # 得到每个高斯核的值
    out = tf.reduce_sum(tf.exp(-1 * tmp / 2 / 0.05), axis=1)
    # 将结果reshape为(-1, r0, r1, 1)的形状
    out = tf.reshape(out, (-1, r0, r1, 1))
    # 返回二维高斯函数的结果
    return out


# 定义一个训练函数，它接受训练集、验证集和测试集作为输入
def train(x_train, y_train, x_val, y_val, x_test, y_test):
    # 设置网络中的一个超参数t，这里通过2的4次方计算得到，即t=16
    t = int(2 ** (int(4)))
    # 定义一个输入层，输入数据的形状为36x36像素的单通道图像
    cnn_input = Input(shape=(36, 36, 1))
    # 添加批量归一化层，用于加速训练并改善模型的泛化能力
    cnn_batch = BatchNormalization()(cnn_input)
    # 第一个卷积层，使用2*t个卷积核，每个卷积核大小为3x3，padding='same'以保持空间维度不变，激活函数为ReLU
    conv1 = Conv2D(2 * t, (3, 3), padding='same', activation='relu')(cnn_batch)
    # 第一个最大池化层，池化窗口大小为3x3
    max1 = MaxPooling2D((3, 3))(conv1)
    # 在池化层后添加批量归一化层
    bat1 = BatchNormalization()(max1)
    # 第二个卷积层，使用1*t个卷积核，配置与第一个卷积层类似
    conv2 = Conv2D(1 * t, (3, 3), padding='same', activation='relu')(bat1)
    # 第二个最大池化层
    max2 = MaxPooling2D((3, 3))(conv2)
    # 将特征图展平为一维向量，以便输入到全连接层
    flat = Flatten()(max2)
    # 第一个全连接层（或称为密集层），使用2*t个神经元和ReLU激活函数
    den1 = Dense(2 * t, activation='relu')(flat)
    # 在全连接层后添加批量归一化层
    den1 = BatchNormalization()(den1)
    # 第二个全连接层，使用1*t个神经元和ReLU激活函数
    den2 = Dense(1 * t, activation='relu')(den1)
    # 再次在全连接层后添加批量归一化层
    den2 = BatchNormalization()(den2)
    # 输出层，使用4个神经元和softmax激活函数（假设是4分类问题）
    den3 = Dense(4, activation='softmax')(den2)
    # 整合所有层，创建模型
    model = Model(cnn_input, den3)
    # 打印模型的概览信息
    model.summary()
    # 设置优化器为带有动量的SGD，并设置学习率、衰减、动量等参数
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.8, nesterov=True)
    # 编译模型，设置损失函数为分类交叉熵，优化器为sgd，并监控准确率
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # 设置早期停止回调，当验证集准确率在连续80个epoch内没有提升时停止训练
    earlystopping = EarlyStopping(monitor='val_accuracy', patience=80, verbose=1, mode='max')
    # 设置模型检查点回调，保存验证集准确率最高的模型权重
    checkpoint = ModelCheckpoint(filepath='model.h5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_accuracy',
                                 mode='max')

    # 记录训练开始时间
    start_time = time.time()
    # 训练模型，并传入训练集、验证集、批大小、迭代次数等参数
    result = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=100, epochs=20, shuffle=True,
                       callbacks=[earlystopping, checkpoint], verbose=1)

    # 打印训练所花费的时间
    print("---Train: %s seconds ---" % (time.time() - start_time))
    # 加载验证集上性能最好的模型权重
    model.load_weights('model.h5')
    # 记录测试开始时间
    start_time = time.time()
    # 使用模型对测试集进行预测，并将预测结果存储在变量y_pred中
    y_pred = model.predict(x_test)
    # 打印模型在测试集上进行预测所花费的时间
    print("---Test: %s seconds ---" % (time.time() - start_time))
    # 使用模型对测试集进行评估，计算损失值和准确率等指标，并将结果存储在变量scores中
    scores = model.evaluate(x_test, y_test)
    # 打印评估结果，通常包括损失值和准确率
    print('scores: ', scores)
    # 函数返回训练好的模型，以便后续使用
    return model

def fine_tune(x_train, y_train, x_val, y_val, x_test, y_test, model):
    """
    对预训练的模型进行微调。
    参数:
    x_train: 训练数据的特征
    y_train: 训练数据的标签
    x_val: 验证数据的特征
    y_val: 验证数据的标签
    x_test: 测试数据的特征
    y_test: 测试数据的标签
    model: 预训练的模型
    返回:
    微调后的模型
    """
    # 遍历模型的每一层，将可训练属性设置为True，所有层的权重被更新
    for layer in model.layers:
        layer.trainable = True
        # 注意：这里使用了'adadelta'作为优化器，但注释中写的是SGD
    # 如果确实要使用SGD，应该将'adadelta'替换为'sgd'并传入之前定义的sgd实例
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # 设置早停回调函数，当验证集准确率在150个epoch内没有提升时停止训练
    earlystopping = EarlyStopping(monitor='val_accuracy', patience=150, verbose=0, mode='max')
    # 设置模型检查点回调函数，保存验证集准确率最高的模型权重
    checkpoint = ModelCheckpoint(filepath='model.h5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_accuracy',
                                 mode='max')
    # 记录开始时间
    start_time = time.time()

    # 训练模型
    result = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                       batch_size=10, epochs=500, shuffle=True,
                       callbacks=[earlystopping, checkpoint], verbose=1)

    # 打印训练时间
    print("---Train: %s seconds ---" % (time.time() - start_time))

    # 加载验证集准确率最高的模型权重
    model.load_weights('model.h5')

    # 评估模型在测试集上的性能
    scores = model.evaluate(x_test, y_test)
    print(scores)

    # 对测试集进行预测
    pre = model.predict(x_test, batch_size=100)
    print(pre)

    # 计算预测概率的均值（取每个样本的最大概率值）
    prob = np.mean(np.amax(pre, axis=1))
    print(prob)

    print("fine-tune完成")

    # 返回微调后的模型
    return model


def topolar(sig):
    """
    将复数信号转换为极坐标表示（幅度和相位）。
    参数:
    sig: 复数数组，表示复数信号
    返回:
    out: 形状为(sig.shape[0], sig.shape[1]*2)的数组，前一半为幅度，后一半为相位
    """
    # 创建一个形状为(sig.shape[0], sig.shape[1]*2)的零数组用于存储结果
    out = np.zeros((sig.shape[0], sig.shape[1] * 2))
    # 遍历每个复数信号
    for i in range(sig.shape[0]):
        # 将复数信号的幅度存储在结果数组的前半部分
        out[i][:sig.shape[1]] = np.abs(sig[i])
        # 计算复数信号的相位（使用反正切函数）并存储在结果数组的后半部分
        # 注意：这里假设sig[i].real和sig[i].imag都是有效的实数，否则可能会引发错误
        out[i][sig.shape[1]:] = np.arctan2(sig[i].imag, sig[i].real)  # 使用np.arctan2更为稳妥
    return out


def main():
    # 加载训练数据
    (x_train, y_train) = load_mat(train_path, 0)  # 从指定路径加载训练数据，并假设第二个参数是数据集的索引或标识

    # 划分训练集和验证集
    (x_train, y_train), (x_val, y_val) = split_data(x_train, y_train, 0.2)  # 将20%的数据作为验证集

    # 打乱训练集和验证集的顺序
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(x_val.shape[0])
    np.random.shuffle(indices)
    x_val = x_val[indices]
    y_val = y_val[indices]

    # 将复数信号转换为极坐标表示
    x_train = topolar(x_train)
    x_val = topolar(x_val)

    # 提取数据的维度以创建输入层
    l = x_train.shape[1]
    # 创建Keras模型
    input_sig = Input(shape=(l, 2))  # 输入层，假设每个信号有两个维度（幅度和相位）
    # 这里应该使用预定义的polar函数（但注意在Lambda层中应该是一个可调用的对象，而不是字符串）
    sig = Lambda(polar, output_shape=[2 * l])(input_sig)  # polar是一个将极坐标转换为其他形式的函数
    # 这里pic2gauss可能是一个自定义的Lambda层，但参数作为arguments传递可能不是标准做法

    output_pic = Lambda(pic2gauss,
                        arguments={'x0': 0, 'x1': 3, 'y0': -1.6, 'y1': 1.6, 'r0': resolution, 'r1': resolution2,
                                   'l': l})(sig)
    # 创建模型
    final = Model(input_sig, output_pic)

    # 记录开始时间
    start_time = time.time()

    # 注释掉的代码是使用模型进行预测，但在这里可能不需要，因为接下来直接转换数据

    # 转换数据到图片形式（假设sig2pic1是这样的函数）
    x_train = sig2pic1(x_train, 0, 3, -1.6, 1.6, resolution, resolution2)
    x_val = sig2pic1(x_val, 0, 3, -1.6, 1.6, resolution, resolution2)

    # 打印数据处理所需时间
    print(time.time() - start_time)

    # 加载测试数据并转换
    (x_test, y_test) = load_mat(test_path, 0)
    x_test = topolar(x_test)
    x_test = sig2pic1(x_test, 0, 3, -1.6, 1.6, resolution, resolution2)
    # 训练模型
    model = train(x_train, y_train, x_val, y_val, x_test, y_test)
    # 对模型进行微调
    model = fine_tune(x_train, y_train, x_val, y_val, x_test, y_test, model)
    # 保存模型结构到JSON文件
    model_json = model.to_json()
    with open("cnnpolar.json", "w") as json_file:
        json_file.write(model_json)

        # 这里SNR是一个变量，应该在函数外部或作为参数传递给main函数
    # 假设SNR已经定义，并代表信噪比或类似的变量
    model.save_weights("cnnpolar_"+SNR+".h5")
    for i in range(4):
        n = int(x_test.shape[0]/4)
        scores = model.evaluate(x_test[n*i:n*(i+1),:],y_test[n*i:n*(i+1),:])
        print('scores: ',scores)
    
if __name__ == "__main__":
    main()
