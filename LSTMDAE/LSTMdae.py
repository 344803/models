import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pickle
import tensorflow as tf
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
import time
import seaborn as sn
import pandas as pd
import h5py

init_notebook_mode()

# parameters
data_path = 'D:\\DIP\\'
file_name = 'RML2016.10a_dict.pkl'

# 定义了两个外部变量，signal_len 表示信号的长度，modulation_num 表示11种不同调制类型
signal_len = 128
modulation_num = 11

def get_amp_phase(data):
    # 假设 data 是一个三维数组，其中第一个维度是样本数，第二个维度是复数信号的实部和虚部（这里是固定的 2），第三个维度是信号长度
    # 将实部和虚部合并为一个复数数组
    X_train_cmplx = data[:, 0, :] + 1j * data[:, 1, :]
    # 计算复数信号的幅度（绝对值）
    X_train_amp = np.abs(X_train_cmplx)
    # 使用 arctan2 函数计算复数信号的相位（弧度），并转换为 [0, 1] 范围内的值（除以π）
    X_train_ang = np.arctan2(data[:, 1, :], data[:, 0, :]) / np.pi
    # 将幅度和相位数组重新塑形为 (-1, 1, signal_len)，其中 -1 表示自动计算该维度的大小
    X_train_amp = np.reshape(X_train_amp, (-1, 1, signal_len))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, signal_len))
    # 将幅度和相位数组在第二个维度（特征维度）上进行拼接
    X_train = np.concatenate((X_train_amp, X_train_ang), axis=1)
    # 将拼接后的数组的维度从 (样本数, 特征数, 信号长度) 转换为 (样本数, 信号长度, 特征数)
    X_train = np.transpose(np.array(X_train), (0, 2, 1))
    # 对每个样本的幅度特征进行 L2 归一化，使其具有单位范数
    for i in range(X_train.shape[0]):
        X_train[i, :, 0] = X_train[i, :, 0] / np.linalg.norm(X_train[i, :, 0], 2)
        # 返回处理后的数据
    return X_train


def set_up_data(data_path, file_name):
    # 使用pickle模块以二进制模式打开文件并加载数据
    with open(data_path + file_name, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

        # 初始化一个空字典来存储数据，格式为 {Modulation: {SNR: IQ数据}}
    dic = {}
    # 遍历加载的数据
    for item in list(data):
        # 提取调制类型和信噪比
        modulation = item[0]
        SNR = int(item[1])

        # 如果调制类型不在字典中，则添加该调制类型，并为其添加一个SNR条目和对应的IQ数据
        if modulation not in dic:
            dic[modulation] = {SNR: data[item]}
            # 如果SNR不在该调制类型的字典中，则添加该SNR条目和对应的IQ数据
        elif SNR not in dic[modulation]:
            dic[modulation][SNR] = data[item]

            # 获取特征长度（即IQ数据的长度）
    len_feature = dic[list(dic)[0]][list(dic[list(dic)[0]])[0]][0].shape[1]
    # 初始化一个空的NumPy数组来存储IQ数据，其中2表示IQ两个分量
    data = np.empty((0, 2, len_feature), float)
    # 初始化一个空列表来存储标签
    label = []

    # 遍历字典，将数据和标签添加到NumPy数组中
    for modulation in dic:
        for snr in dic[modulation]:
            # 为每个IQ数据对生成一个标签列表（调制类型和SNR）
            label.extend(list([modulation, snr] for _ in range(dic[modulation][snr].shape[0])))
            # 将IQ数据添加到data数组中
            data = np.vstack((data, dic[modulation][snr]))

            # 将标签列表转换为NumPy数组
    label = np.array(label)
    # 创建一个索引列表并随机打乱它
    index = list(range(data.shape[0]))
    np.random.seed(2019)  # 设置随机种子以确保结果的可重复性
    np.random.shuffle(index)

    # 设定训练集、验证集和测试集的比例
    train_proportion = 0.5
    validation_proportion = 0.25
    test_proportion = 0.25

    # 根据索引和比例分割数据集
    X_train = data[index[:int(data.shape[0] * train_proportion)], :, :]
    Y_train = label[index[:int(data.shape[0] * train_proportion)]]
    X_validation = data[index[int(data.shape[0] * train_proportion): int(
        data.shape[0] * (train_proportion + validation_proportion))], :, :]
    Y_validation = label[index[int(data.shape[0] * train_proportion): int(
        data.shape[0] * (train_proportion + validation_proportion))]]
    X_test = data[index[int(data.shape[0] * (train_proportion + validation_proportion)):], :, :]
    Y_test = label[index[int(data.shape[0] * (train_proportion + validation_proportion)):]]

    # 创建一个字典，将调制类型映射到其索引（用于后续的机器学习模型）
    modulation_index = {}
    modulations = np.sort(list(dic))
    for i in range(len(list(dic))):
        modulation_index[modulations[i]] = i

        # 返回分割后的数据集和调制索引字典
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test, modulation_index


def zero_mask(X_train, p):
    # 计算每个样本需要设置为零的特征数量
    num = int(X_train.shape[1] * p)
    # 创建一个X_train的副本，这样原始数据不会被修改
    res = X_train.copy()
    # 创建一个二维索引数组，其形状与X_train的第一个和第二个维度相同
    # 第一个维度是样本数，第二个维度是每个样本的所有特征索引
    index = np.array([[i for i in range(X_train.shape[1])] for _ in range(X_train.shape[0])])

    # 对每个样本的特征索引进行随机打乱
    for i in range(index.shape[0]):
        np.random.shuffle(index[i, :])
        # 遍历每个样本，并将其前num个随机特征（IQ分量）设置为零
    for i in range(res.shape[0]):
        res[i, index[i, :num], :] = 0
        # 返回应用零掩码后的数据集
    return res

# set up data
# 假设 set_up_data 是一个函数，它读取数据并返回训练集、验证集和测试集的输入与输出，以及一个调制指数数组
X_train, Y_train, X_validation, Y_validation, X_test, Y_test, modulation_index = set_up_data(data_path, file_name)

# 以下三个循环用于将 Y_train, Y_validation, Y_test 中的标签（可能是索引）替换为对应的调制指数值
# 遍历训练集的每个样本
for i in range(Y_train.shape[0]):
    Y_train[i, 0] = modulation_index[Y_train[i, 0]]

# 遍历验证集的每个样本
for i in range(Y_validation.shape[0]):
    Y_validation[i, 0] = modulation_index[Y_validation[i, 0]]

# 遍历测试集的每个样本
for i in range(Y_test.shape[0]):
    Y_test[i, 0] = modulation_index[Y_test[i, 0]]

# 将 X_train, X_validation, X_test 从二维或三维（如果存在额外的 IQ 分量）重新整理为三维，其中 IQ 分量为第二维度
# 假设 X_train, X_validation, X_test 原本可能是二维的（没有 IQ 分量）或三维的（有 IQ 分量但结构不是 (samples, IQ, features)）
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
Y_train = Y_train.astype(int)  # 将 Y_train 的数据类型转换为整数

X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], X_validation.shape[2]))
Y_validation = Y_validation.astype(int)  # 将 Y_validation 的数据类型转换为整数

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
Y_test = Y_test.astype(int)  # 将 Y_test 的数据类型转换为整数

# 调用 get_amp_phase 函数来处理输入数据，可能是提取振幅和相位信息
# 假设 get_amp_phase 函数返回处理后的数据，这些数据将用于后续的模型训练、验证和测试
X_train = get_amp_phase(X_train)
X_validation = get_amp_phase(X_validation)
X_test = get_amp_phase(X_test)

# 输入层定义，根据训练数据的形状设置输入维度
encoder_inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]), name='encoder_inputs')

# 第一个LSTM编码器层，返回序列和内部状态
encoder_1, state_h_1, state_c_1 = tf.keras.layers.LSTM(units=32, return_sequences=True, return_state=True,
                                                       name='encoder_1')(encoder_inputs)#隐藏单元数量32返回完整的输出序列返回隐藏状态和细胞状态

# Dropout层，用于减少过拟合，但这里drop_prob设为0，实际不会丢弃任何单元
drop_prob = 0
drop_1 = tf.keras.layers.Dropout(drop_prob, name='drop_1')(encoder_1)

# 第二个LSTM编码器层，同样返回序列和内部状态
encoder_2, state_h_2, state_c_2 = tf.keras.layers.LSTM(units=32, return_state=True, return_sequences=True,
                                                       name='encoder_2')(drop_1)

# 时间分布的全连接层作为解码器，每个时间步输出2个单元
decoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2), name='decoder')(encoder_2)

# 分类部分的三个密集层，带有批归一化和Dropout
clf_dropout = 0
clf_dense_1 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu, name='clf_dense_1')(state_h_2)  # 使用LSTM的最后一个隐藏状态
bn_1 = tf.keras.layers.BatchNormalization(name='bn_1')(clf_dense_1)
clf_drop_1 = tf.keras.layers.Dropout(clf_dropout, name='clf_drop_1')(bn_1)

clf_dense_2 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu, name='clf_dense_2')(clf_drop_1)
bn_2 = tf.keras.layers.BatchNormalization(name='bn_2')(clf_dense_2)
clf_drop_2 = tf.keras.layers.Dropout(clf_dropout, name='clf_drop_2')(bn_2)

clf_dense_3 = tf.keras.layers.Dense(units=modulation_num, name='clf_dense_3')(clf_drop_2)

# Softmax层用于多分类，但注意在TensorFlow 2.x中，更推荐使用Activation层配合'softmax'参数
softmax = tf.keras.layers.Softmax(name='softmax')(clf_dense_3)  # 在TensorFlow 2.x中通常使用Activation层

# 定义模型，有两个输出：解码器的输出和分类的softmax输出
model = tf.keras.Model(inputs=encoder_inputs, outputs=[decoder, softmax])
model.summary()

# 设置学习率和正则化项的权重
learning_rate = 10 ** -2  # 学习率设置为0.01
lam = 0.1  # 正则化项的权重（例如L1或L2正则化），但在此代码中似乎用于损失函数的权重

# 编译模型，设置损失函数、损失权重、评估指标和优化器
model.compile(loss=['mean_squared_error', 'categorical_crossentropy'],  # 损失函数，分别对应解码器和分类器
              loss_weights=[1 - lam, lam],  # 损失权重，这里根据lam的值来分配
              metrics=['accuracy'],  # 评估指标，但注意这里仅对分类器有效，解码器通常不使用accuracy
              optimizer=tf.keras.optimizers.Adam(lr=learning_rate))  # 使用Adam优化器并设置学习率

best = 0  # 初始化最高验证准确率
train_acc = []  # 存储每次迭代的训练准确率
val_acc = []  # 存储每次迭代的验证准确率
import time



# 记录开始时间
start_time = time.time()
# 开始迭代训练模型
for ite in range(150):  # 迭代150次
    # 对训练数据进行某种形式的掩码处理（可能是数据增强或正则化）
    X_train_masked = zero_mask(X_train, 0.1)  # zero_mask函数，用于以0.1的概率将X_train的某些元素置为0

    # 训练模型
    history = model.fit(x=X_train_masked,  # 输入数据
                        y=[X_train, tf.keras.utils.to_categorical(Y_train[:, 0])],  # 输出数据，包括解码器的目标和分类器的目标
                        validation_data=(  # 验证数据
                            X_validation, [X_validation, tf.keras.utils.to_categorical(Y_validation[:, 0])]
                        ),
                        batch_size=400,  # 批处理大小
                        epochs=1)  # 每次迭代只训练一个epoch

    # 提取并存储训练准确率和验证准确率（但注意这里可能存在问题，因为metrics只设置了'accuracy'）
    train_acc.append(history.history['softmax_accuracy'][0])  # 假设'softmax_accuracy'是自定义的metrics，但通常不是这样命名的
    val_acc.append(history.history['val_softmax_accuracy'][0])  # 同样，这里假设有一个自定义的验证准确率指标

    # 如果当前验证准确率比之前的高，则保存模型
    if history.history['val_softmax_accuracy'][0] > best:
        best = history.history['val_softmax_accuracy'][0]
        model.save('DAELSTM_test.h5')  # 保存模型到文件

    # 将验证准确率写入文件
    with open('val_result.txt', 'a') as f:
        f.write(str(history.history['val_softmax_accuracy'][0] * 100) + '\n')  # 写入验证准确率的百分比形式
end_time = time.time()

# 计算总耗时（秒）
total_time_seconds = end_time - start_time

# 转换为小时和分钟
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60)

# 打印总耗时
print(f"训练总耗时：{hours}小时{minutes}分钟")
with open('training_time.txt', 'w') as f:
    f.write(f"训练总耗时：{hours}小时{minutes}分钟\n")

# 加载先前保存的模型
clf = tf.keras.models.load_model('DAELSTM_test.h5')

start_time1 = time.time()
# 使用模型对测试集进行预测，假设模型有两个输出（解码器和分类器），只取第二个输出（分类器的输出）
res = clf.predict(X_test)[1]

end_time1 = time.time()

# 计算总耗时（秒）
total_time_seconds1 = end_time1 - start_time1

# 转换为小时和分钟
hours1 = int(total_time_seconds1 // 3600)
minutes1 = int((total_time_seconds1 % 3600) // 60)
print(f"预测验证集总耗时：{hours1}小时{minutes1}分钟")
with open('prediction.txt', 'w') as f:
    f.write(f"预测总耗时：{hours1}小时{minutes1}分钟\n")

# 将分类器的输出（通常是概率分布）转化为预测类别，使用numpy的argmax函数沿着第二个维度（类别维度）取最大值索引
res = np.argmax(res, axis=1)



# 初始化一个字典来存储每个测试集类别的正确预测数量和总测试数量
test_accuracy = {}

# 遍历测试集
for i in range(X_test.shape[0]):
    # 检查当前类别的索引（Y_test[i, 1]）是否已经在字典中
    if Y_test[i, 1] not in test_accuracy:
        # 如果不在字典中，根据预测是否正确初始化字典的条目
        if Y_test[i, 0] == res[i]:
            test_accuracy[Y_test[i, 1]] = [1, 1]  # [正确数量, 总数量]
        else:
            test_accuracy[Y_test[i, 1]] = [0, 1]
    else:
        # 如果已经在字典中，更新正确预测数量和总测试数量
        if Y_test[i, 0] == res[i]:
            test_accuracy[Y_test[i, 1]][0] += 1
            test_accuracy[Y_test[i, 1]][1] += 1
        else:
            test_accuracy[Y_test[i, 1]][1] += 1

        # 初始化变量来存储所有类别的正确预测总数和总测试数
nomi = 0  # 正确预测总数
deno = 0  # 总测试数

# 遍历test_accuracy字典，累加正确预测数和总测试数
for snr in test_accuracy:
    nomi += test_accuracy[snr][0]
    deno += test_accuracy[snr][1]

# 计算平均准确率
best = nomi / deno

# 打开一个文件，用于追加写入结果
with open('result.txt', 'a') as f:
    # 遍历test_accuracy字典，计算每个类别的准确率，并将结果以百分比形式写入文件
    for i in np.sort(list(test_accuracy)):
        accuracy_per_class = test_accuracy[i][0] / test_accuracy[i][1]
        f.write(str(accuracy_per_class * 100) + '\n')
        print(" ")

        # 写入平均准确率
    f.write(str(best * 100) + '\n')
    f.write('\n')  # 写入一个空行，使结果更易读