import streamlit as st
import numpy as np
import pandas as pd

tab1, tab2 = st.tabs(['1.LSTM', '2.Bi-RNN'])

with tab1:
    """1. 标准循环神经网络训练的优化算法面临长期依赖问题----由于网络结构的变深使得模型丧失了学习到先前信息的能力。"""
    """2. 由于简单RNN遇到时间步（timestep）较大时，容易出现梯度消失或爆炸问题，且随着层数的增加，网络最终无法训练，无法实现长时记忆。"""
    """3.解决RNN中的梯度消失方法很多，常用的有："""
    """$ \qquad $ 1）选取更好的激活函数，如ReLU激活函数。ReLU函数的左侧导数为0，右侧导数恒为1，这就避免了“梯度消失”的发生。"""
    """$ \qquad $ 2）加入BN层，其优点包括可加速收敛、控制过拟合。"""
    """$ \qquad $ 3）修改网络结构，LSTM结构可以有效地解决这个问题。"""
    """- RNN是一种死板的逻辑，越晚的输入影响越大，越早的输入影响越小，且无法改变这个逻辑。"""
    """- LSTM可以保留长序列数据中的重要信息，忽略不重要的信息。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./四NLP/图片19.png', caption='图19')
    """- 长短时记忆神经网络（Long Short-Term Memery,LSTM）最早由Hochreiter & Schmidhuber于1997年提出，能够有效解决信息的长期依赖，避免梯度消失或爆炸。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./四NLP/图片20.jpg', caption='图20')

with tab2:
    """- 在有些任务中，一个时刻的输出不但和过去时刻的信息有关，也和后续时刻的信息有关。"""
    """- 给定一个句子，其中一个词的词性由它的上下文决定，即包含左右两边的信息。因此，在这些任务中，可以增加一个按照时间的逆序来传递信息的网络层，增强网络的能力。"""
    """- 双向循环神经网络由两层循环神经网络组成，它们的输入相同，只是信息传递的方向不同。"""
    """- 双向循环神经网络（Bidirectional Recurrent Neural Networks,Bi-RNN）模型由Schuster、Paliwal于1997年首次提出，和LSTM同年。"""
    """- 假设第一层按时间顺序，第2层按时间顺序，在时刻t时的隐藏状态定义为:"""
    st.latex(r"""h_t^{(1)}=f(U^{(1)}h_{t-1}^1+W^{(1)}x_t+b^{(1)})""")
    st.latex(r"""h_t^{(2)}=f(U^{(2)}h_{t-1}^2+W^{(2)}x_t+b^{(2)})""")
    st.latex(r"""h_t=h_t^{(1)}\bigoplus{h_t^{(2)}}""")
    r"""$ \qquad $ 其中$\bigoplus$为向量拼接操作。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./四NLP/图片21.png', caption='图21 按时间展开的双向循环神经网络')
    """1.采用Bi-RNN能提升模型效果。百度语音识别就是通过Bi-RNN综合上下文语境，提升模型准确率。"""
    """2.双向循环神经网络的基本思想是提出每一个训练序列向前和向后分别是两个循环神经网络（RNN），而且这两个都连接着一个输出层。这个结构提供给输出层输入序列中每一个完整的过去和未来的上下文信息。"""
