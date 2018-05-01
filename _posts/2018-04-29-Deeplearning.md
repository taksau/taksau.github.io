---
layout: post
title: "深度学习预备知识(Part 1)"
author: "Metapunk"
categories: notes
tags: [notes]
image: motoko-1.jpg
---



## 线性代数

#### 向量的**范数**(norm)

$$
||x||_{p}=(\sum_{i}|x|^{p})^\frac{1}{p} \\
其中p\in \mathbb{R},p\ge 1
$$



$L^{1}$范数经常作为表示非零元素数目的替代函数

**最大范数$L^{\infty}$：**表示向量中具有最大幅值的元素的绝对值
$$
||x||_{\infty}=\max_{i}|x_{i}|
$$
 在深度学习中，会衡量矩阵的大小，最常用**Frobenius范数** (Frobenius norm)
$$
||A||_{F}=\sqrt{\sum_{i,j}A^{2}_{i,j}}
$$

### 特殊类型的矩阵和向量

* **对角矩阵**(diagonal matrix)只在主对角线上含有非零元素，其他位置都是零。用diag(v)表示。

* **对称矩阵**(symmetric matrix)是转置和自己相等的矩阵。

* **正交矩阵**(orthogonal matrix)指行向量和列向量是分别标准正交的方阵，即
  $$
  A^{\top}A=AA^{\top}=I \\
   A^{-1}=A^{\top}
  $$












### 特征分解

方阵$A$的**特征向量** (eigenvector)是指与$A$相乘后相当于对该向量进行缩放的非零向量**v**：
$$
\mathit{Av=\lambda v}
$$
其中标量$\mathit{\lambda}$称为这个特征向量对应的**特征值**(eigenvalue)。

假设矩阵$A$ 有$n$ 个线性无关的特征向量$\{v^{(1)},\cdots,v^{(n)}\}$，对应着特征值$\{\lambda_{1},\cdots,\lambda_{n}\}$。 将特征向量连接成一个矩阵，使得每一列是一个特征向量：$V=[v^{(1)},\cdots,v^{(n)}]$ 。类似地，将特征值连接成一个向量$\lambda=[\lambda_{1},\cdots,\lambda_{n}]^{\top}$。因此$A$的**特征分解**(eigendecomposition)可以记作：
$$
A=Vdiag(\lambda)V^{-1}
$$

每个实对称矩阵都可以分解成实特征向量和实特征值：

$$
A=Q\Lambda Q^{\top}
$$

其中$Q$是$A$的特征向量组成的正交矩阵，$\Lambda$是对角矩阵。特征值$\Lambda_{i,i}$对应的特征向量是矩阵$Q$的第$i$列，记作$Q_{:,i}$。

如果两个或多个特征向量拥有相同的特征值，那么在由这些特征向量产生的生成子空间中，任意一组正交向量都是该特征值对应的特征向量。因此，我们可以等价地从这些特征向量中构成$Q$作为替代。按照惯例，我们通常按降序排列$\Lambda$的元素。在该约定下，特征分解唯一，当且仅当所有的特征值都是唯一的。

所有特征值都是正数的矩阵称为**正定**(positive definite)；所有特征值都是非负数的矩阵称为**半正定**(positive semidefinite)。同样地，所有特征值都是负数的**负定**(negative definite)；所有特征值都是非正数的矩阵称为**半正定**(negative semidefinite)。半正定矩阵受到关注是因为它们保证$\forall x,x^{\top}Ax\ge0$。此外，正定矩阵还保证$x^{\top}Ax=0\Rightarrow x=0$。

### 奇异值分解

将矩阵分解为**奇异向量**(singular vector)和**奇异值**(singular value)。通过奇异值分解，我们会得到一些特征分解相同类型的信息。每个实数矩阵都有一个奇异值分解，但不一定都有特征分解。

将矩阵$A$分解成三个矩阵的乘积:
$$
A=UDV^{\top}
$$
假设$A$是一个$m\times n$的矩阵，那么$U$是一个$m\times m$的矩阵，$D$是一个$m\times n$的矩阵，$V$是一个$n\times n$的矩阵。

矩阵$U$和$V$都定义为正交矩阵，而矩阵$D$定义为对角矩阵。矩阵$D$不一定是方阵。

对角矩阵$D$对角线上的元素称为矩阵$A$的**奇异值**(singular value)。矩阵$U$列向量称为**左奇异向量**(left singular vector)，矩阵$V$的列向量称**右奇异值**(right singular vector)。

我们可以用与**A**相关的特征分解去解释**A**的奇异值分解。$A$的**左奇异向量**(left singular vector)是$AA^{\top}$的特征向量。$A$的**右奇异向量**(right singular vector)是$A^{\top}A$的特征向量。$A$的非零奇异值是$A^{\top}A$特征值的平方根，同时也是$AA^{\top}$特征值的平方根。

### Moore-Penrose 伪逆

$$
A^{+}=\lim_{a\to 0}(A^{\top}A+\alpha I)^{-1}A^{-1}
$$

计算伪逆的实际算法没有基于这个定义，而是使用下面的公式
$$
A^{+}=VD^{+}U^{\top}
$$
对角矩阵$D$的伪逆$D^{+}$是其非零元素取倒数之后再转置得到的。

当矩阵$A$的列数多于行数时，使用伪逆求解线性方程是众多可能解法中的一种。特别地，$x=A^{+}y$是方程所有可行解中欧几里得范数$\Arrowvert x \Arrowvert_{2}$最小的一个。

当矩阵$A$的行数多于列数时，可能没有解。在这种情况下，通过伪逆得到的$x$使得$Ax$和$y$的欧几里得距离$\Arrowvert Ax-y \Arrowvert_{2}$最小。

### 迹运算

迹运算返回的是矩阵对角元素的和：

$$
Tr(A)=\sum_{i}A_{i,i}.
$$
Frobenius范数：
$$
\Arrowvert A \Arrowvert_{F}=\sqrt{Tr(AA^{\top})}
$$
性质：
$$
Tr(AB)=Tr(BA)
$$
标量在迹运算后仍然是它自己：$a=Tr(a)$。

## 概率与信息论

### 高斯分布

$$
\mathcal{N}(x;\mu,\sigma^{2})=\sqrt{\frac{1}{2\pi \sigma^{2}}}exp(-\frac{1}{2\sigma^{2}}(x-\mu)^{2})
$$

### 常用函数的有用性质

* $logistic$ $sigmoid$ 函数：

$$
\sigma(x)=\frac{1}{1+exp(-x)}
$$

* $softplus$函数：

$$
\zeta(x)=log(1+exp(x))
$$

### 信息论

**KL散度**对于同一随机变量x有两个单独的概率分布$P(x)$和$Q(x)$，可以使用**KL散度**(Kullback-Leibler divergence)来衡量这两个分布的差异：
$$
D_{KL}(P||Q)=\mathbb{E}_{x\sim P}[log\frac{P(x)}{Q(x)}]=\mathbb{E}_{x\sim P}[logP(x)-logQ(x)]
$$
**交叉熵**(cross-entropy)：
$$
H(P,Q)=H(P)+D_{KL}(P||Q)=-\mathbb{E}_{x\sim P}logQ(x)
$$

## 数值计算

### 病态条件

条件数指的是函数相对于输入的微小变化而变化的快慢程度。

 考虑函数$f(x)=A^{-1}x$。当$A\in \mathbb{R}^{n\times n}$具有特征值分解时，其条件数为
$$
\max_{i,j}|\frac{\lambda_{i}}{\lambda_{j}}|
$$
当该数很大时，矩阵求逆对输入的误差特别敏感。

### 基于梯度的优化方法

我们把要最小化或最大化的函数称为**目标函数**(objective function)或**准则**(criterion)。当我们对其进行最小化时，也把它称为**代价函数**(cost function)、**损失函数**(loss function)或**误差函数**(error function)。

**最速下降法**(method of steepest descent)或**梯度下降**(gradient descent)。最速下降建议新的点为
$$
x^{\prime}=x-\epsilon \nabla_{x}f(x)
$$
其中$\epsilon$为**学习率**(learning rate)，是一个确定步长大小的正标量。

$Hessian$矩阵：
$$
H(f)(x)_{i,j}=\frac{\partial ^{2}}{\partial x_{i} \partial x_{j}}f(x)
$$
**最优步长**：
$$
\epsilon^{*}=\frac{g^{\top}g}{g^{\top}Hg}
$$
最坏的情况下，$g$与$H$最大特征值$\lambda _{max}$对应的特征向量对齐，则最优步长是$\frac{1}{\lambda _{max}}$。当我们要最小化的函数能用二次函数很好地近似的情况下，$Hessian$的特征值决定了学习率的量级。

在深度学习的背景下，限制函数满足$Lipschitz$**连续**(Lipschitz continuous)或其导数$Lipschitz$连续可以获得一些保证。$Lipschitz$连续函数的变化速度以$Lipschitz$**常数**(Lipschitz constant) $\mathcal{L}$为界：
$$
\forall x,\forall y,|f(x)-f(y)|\le \mathcal{L}||x-y||_{2}
$$

### 约束优化

**Karush-Kuhn-Tucker**(KKT)方法是针对约束优化非常通用的解决方案。为介绍KKT方法，我们引入一个称为**广义Lagrangian**(generalized Lagrangian)。
$$
\mathbb{S}=\{ x|\forall i,g^{(i)}(x)=0\ and \ \forall j,h^{(j)}(x)\le0\}
$$
我们为每个约束引入新的变量$\lambda_{i}$ 和$\alpha_{j}$，这些新变量被称为KKT乘子。
$$
L(x,\lambda,\alpha)=f(x)+\sum_{i}\lambda_{i}g^{(i)}(x)+\sum_{j}\alpha_{j}h^{(j)}(x)
$$
