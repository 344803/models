import numpy as np  # 导入NumPy库，用于数组和数值计算
import h5py  # 导入h5py库，用于处理HDF5文件

##############################全局参数#######################################
# 打开一个HDF5文件，只读模式，该文件包含数据集
f = h5py.File('.\RML2018/GOLD_XYZ_OSC.0001_1024.hdf5', 'r')
# 定义提取数据集保存的目录路径
dir_path = 'ExtractDataset'
# 定义每个模块和SNR（信噪比）需要提取的样本数量
modu_snr_size = 1200
############################################################################

# 遍历24个模块
for modu in range(24):
    # 初始化X、Y和Z数据的列表，用于存储每个模块的数据
    X_list = []
    Y_list = []
    Z_list = []
    print('part ', modu)  # 输出当前处理的模块编号
    # 计算当前模块的起始索引
    start_modu = modu * 106496
    # 遍历26个SNR（信噪比）值
    for snr in range(26):
        # 计算当前SNR的起始索引
        start_snr = start_modu + snr * 4096
        # 随机选择4096个样本中的modu_snr_size个样本，不重复
        idx_list = np.random.choice(range(0, 4096), size=modu_snr_size, replace=False)
        # 从HDF5文件中提取对应索引的X数据
        X = f['X'][start_snr:start_snr + 4096][idx_list]
        # 将提取的X数据添加到X_list
        X_list.append(X)
        # 从HDF5文件中提取对应索引的Y数据并添加到Y_list
        Y_list.append(f['Y'][start_snr:start_snr + 4096][idx_list])
        # 从HDF5文件中提取对应索引的Z数据并添加到Z_list
        Z_list.append(f['Z'][start_snr:start_snr + 4096][idx_list])
    
    # 定义当前模块保存数据的文件名
    filename = dir_path + '/part' + str(modu) + '.h5'
    # 创建一个新的HDF5文件，用于保存提取的数据
    fw = h5py.File(filename, 'w')
    
    fw['X'] = np.vstack(X_list)
  
    fw['Y'] = np.vstack(Y_list)
    
    fw['Z'] = np.vstack(Z_list)
    # 输出保存的数据的形状信息
    print('X shape:', fw['X'].shape)
    print('Y shape:', fw['Y'].shape)
    print('Z shape:', fw['Z'].shape)
    # 关闭新的HDF5文件
    fw.close()

# 关闭原始HDF5文件
f.close()
