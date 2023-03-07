import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

tab1, tab2 = st.tabs(['1.卷积', '2.池化'])

with tab1:
    """- 卷积定义："""
    r"""$ \qquad $ 1.卷积：数学上卷积是两个实变函数的一种积分运算。"""
    st.latex(r"""S(t)=\int{x(a)w(t-a)da}=(x*w)(t)""")
    r"""$ \qquad $ 在上式中w被称为卷积核kernel，或者滤波器filter。"""

    r"""$ \qquad $ 2.互相关：表示两个信号的在任意两个时刻的相关程度。"""
    st.latex(r"""R(t)=\int{x(a)w(a+t)da}=(x*w)(t)""")

    r"""$ \qquad $ 3.互相关和卷积几乎相同，不同之处在于前者不用做核翻转。机器学习的库中实现的是互相关函数但却称之为卷积。"""

    """- 一维卷积："""
    r"""$ \qquad $ 1.一维卷积经常用在信号处理中用于计算信号的延迟累积。"""
    r"""$ \qquad $ 2.假设一个信号发生器每个时刻t产生一个信号$𝑥_𝑡$，其信息的衰减率为$𝑤_𝑘$,即在k-1个时间步长之后信息为原来的$𝑤_𝑘$倍。假设，$𝑤_1=1,𝑤_2=\frac{1}{2},𝑤_3=\frac{1}{4}$，那么在t时刻收到的信号$𝑦_𝑡$可以表示为：
"""
    st.latex(r"""y_t=1\times{x_t}+\frac{1}{2}\times{x_{t-1}}+\frac{1}{4}\times{x_{t-2}}=\sum_{k=1}^3{w_kx_{t-k+1}}""")
    r"""$ \qquad $ 称$𝑤_1,𝑤_2,𝑤_3$为卷积核，假设卷积核的长度为K，它和一个信号序列$𝑥_1,𝑥_2,…$的卷积表示为:"""
    st.latex(r"""y_t=\sum_{k=1}^K{w_kx_{t-k+1}}""")
    r"""$\qquad $ 3.信号序列x和滤波器w的卷积定义为："""
    st.latex(r"""y=w*x""")
    r"""$\qquad $ (*表示卷积运算)"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./三卷积CNN/图片1.jpg', caption='图1 一维卷积示例图')
    r"""$ \qquad $ 滤波器为[-1,0,1].一般情况下滤波器的长度要远远小于信号序列的长度。"""
    """- 二维卷积"""
    r"""$ \qquad $ 输入二维图像I，卷积核K，二维离散卷积为："""
    st.latex(r"""S(i,j)=(I*K)(i,j)=\sum_{m,n}{I(m,n)K(i-m,j-n)}""")
    r"""$ \qquad $ 卷积具有可交换性："""
    st.latex(r""" S(i,j)=(K*I)(i,j)""")
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./三卷积CNN/图片2.gif', caption='图2 二维卷积的计算过程')
    st.latex(r""" 图2中卷积核为： \left( \begin{matrix}
   1 & 0 & 1 \\
   0 & 1 & 0 \\
   1 & 0 & 1
  \end{matrix} \right) 
$$
""")
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./三卷积CNN/图片3.png', caption='图3 卷积层的三维结构表示')

    """- 卷积的动机"""
    r"""$ \qquad $ 卷积运算通过三个重要的思想来帮助改进机器学习系统："""
    r"""$ \qquad \qquad $ 稀疏交互(sparse interactions)、参数共享(parameter sharing)、等变表示(equivariant representations)。"""
    r"""$ \qquad $ 1.稀疏交互：卷积网络具有系数交互（也叫做稀疏连接或稀疏权重）的特征。这是使核的大小远小于输入的大小来达到的。"""
    _, col1, col2,_ = st.columns([1, 2, 2, 1])
    with col1:
        st.image('./三卷积CNN/图片4.png', caption='图4')
    with col2:
        st.image('./三卷积CNN/图片5.png', caption='图5')
    r"""$ \qquad $ 2.参数共享"""
    r"""$ \qquad \qquad $ 参数共享是指在一个模型的多个函数中使用相同的参数。"""
    r"""$ \qquad \qquad $ 卷积运算中的参数共享保证了我们只需要学习一个参数集合，而不是对于每一位置都需要学习一个单独的参数集合。"""
    r"""$ \qquad \qquad $ 虽然没有改变前向传播的运行时间（仍然是𝑂(𝑘×𝑛)），但它现住地把模型的存储需求降低至k个参数，并且k通常要比m小很多个数量级。"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./三卷积CNN/图片6.png', caption='图6 参数共享')
    r"""$ \qquad $ 边缘提取的效率："""
    r"""$ \qquad \qquad $ 下图说明了稀疏连接+核参数共享是如何显著提高线性函数在一张图像上进行边缘检测的效率的。 """
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./三卷积CNN/图片7.png', caption='图7')

    r"""$ \qquad $ 3.等变表示"""
    r"""$ \qquad \qquad $ 如果一个函数满足输入改变，输出也以同样的方式改变这一性质，我们就说它是等变的。特别的是，如果函数$𝑓(𝑥)$与$𝑔(𝑥)$满足$𝑓(𝑔(𝑥))=𝑔(𝑓(𝑥))$，我们就说$𝑓(𝑥)$对于变换g具有等变性。"""
    r"""$ \qquad \qquad $ 对于卷积，参数共享的特殊形式使得神经网络具有对平移等变(equivariance)的性质。"""
    r"""$ \qquad \qquad $ 卷积操作对图片的旋转或者缩放不是等变的。"""

    """- 填充和步长"""
    r"""$ \qquad $ 1.填充(padding)：对图像边缘进行填充的操作。"""
    r"""$ \qquad $ 2.步长(stride)：跳过核中的一些位置，降低计算开销，可以看做是对全卷积操作的下采样。"""
    r"""$ \qquad $ 3.填充的几种方式：0填充、复制填充、对称填充。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./三卷积CNN/图片8.png', caption='图8')
    _, col1, col2, col3 = st.columns([0.5, 2.7, 1.5, 1.7])
    with col1:
        st.image('./三卷积CNN/图片9.gif', caption='图9 padding:(0,0),stride:(1,1)')
    with col2:
        st.image('./三卷积CNN/图片10.gif', caption='图10 padding:(2,2),stride:(1,1)')
    with col3:
        st.image('./三卷积CNN/图片11.gif', caption='图11 padding:(0,0),stride:(2,2)')
    """- 卷积的变种："""
    r"""$ \qquad $ 1. 常规卷积"""
    r"""$ \qquad $ 2. 空洞卷积"""
    r"""$ \qquad $ 3. 分组卷积"""
    r"""$ \qquad $ 4. 反卷积"""
    r"""$ \qquad $ 5. 图卷积"""
    r"""$ \qquad $ 6. 部分卷积、可变性卷积、因果卷积等等...."""

with tab2:
    """- 定义：池化(Pooling)使用某一位置的相邻输出的总体统计特征来代替网络在该位置的输出。"""
    """- 池化的几种方式："""
    r"""$ \qquad $ 1. 最大池化(Max pooling)"""
    r"""$ \qquad $ 2. 平均池化(Averag pooling)"""
    r"""$ \qquad $ 3. 其它如$𝐿^2$范数以及基于距中心像素距离的加权平均函数等。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./三卷积CNN/图片12.jpg', caption='图12 池化的计算过程')
    """- 池化的作用："""
    r"""$ \qquad $ 1. 无限强先验：不管采用什么样的池化函数，当输入作出少量平移时，池化能够帮助输入的表示近似不变（invariant）。局部平移不变性是一个很有用的性质，尤其是当我们关心某个特征是否出现而不关心它出现的具体位置时。"""
    r"""$ \qquad $ 2.极大地提高计算效率，下一层的输入少了大约k倍。"""
    r"""$ \qquad $ 3.减少了参数存储需求。"""
    r"""$ \qquad $ 4.学习不变形。"""
    r"""$ \qquad \qquad $ 对空间区域进行池化产生了平移不变性，但当我们对分离参数的卷积的输出进行池化时，特征能够学得应该对于哪种变换具有不变性。如下图： """
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./三卷积CNN/图片13.png', caption='图13')
