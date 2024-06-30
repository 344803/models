# 论文介绍

题目：Enhanced Low SNR Radio Signal Classification using Deep Learning Final Report

使用深度学习增强的低信噪比无线电信号分类最终报告

Final Project for AI Wireless 2020 in National Chiao Tung University
*Top final project in the AI Wireless 2020 NCTU (Fall 2020)*

github链接：[GitHub - alexivaner/Deep-Learning-Based-Radio-Signal-Classification: Final Project for AI Wireless](https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification?tab=readme-ov-file)

本文章是基于论文《[Over the Air Deep Learning Based Radio Signal Classification](https://arxiv.org/pdf/1712.04578.pdf)》中R-ResNet的改进。

# **高SNR模型（改进的残差网络ResNet）**

1. 作者对ResNet进行了修改，引入了新的优化器**LazyAdam**，并在残差单元中添加了**批量归一化（Batch Normalization）**。该模型可以有效地处理高SNR信号，提高了分类准确性和训练效率。

   ![image-20240615104202518](C:\Users\xql\AppData\Roaming\Typora\typora-user-images\image-20240615104202518.png)

   ![image-20240614175204307](C:\Users\xql\AppData\Roaming\Typora\typora-user-images\image-20240614175204307.png)

# **低SNR模型（Transformer网络）**

![image-20240614175731835](C:\Users\xql\AppData\Roaming\Typora\typora-user-images\image-20240614175731835.png)

# 数据集与实验

论文使用了DeepSig无线电信号数据集，数据集包含了24种不同类型的信号调制方式。数据集包括了干净的信号和噪声信号，能够很好地模拟现实生活中信号的状态。RML2018。

'32PSK'， '16APSK'， '32QAM'， 'FM'， 'GMSK'， '32APSK'， 'OQPSK'， '8ASK'， 'BPSK'， '8PSK'， 'AM-SSB-SC'， '4ASK'， '16PSK'， '64APSK'， '128QAM'， '128APSK'，'AM-DSB-SC'， 'AM-SSB-WC'， '64QAM'， 'QPSK'， '256QAM'， 'AM-DSB-WC'， 'OOK'， '16QAM'

