---
layout: post
title: 初涉反向传播
author: Metapunk
categories: notes
tags: [notes]
image: thor.jpg
---



## 求谁的梯度？

$W,b,x_{i}$都是作为$loss function$的变量，但是学习的过程中，要优化的是$W,b$ ，所以在反向传播算法中，我们计算的都是$W,b$的梯度。

## 计算图

一些简单的多元函数的偏导：
$$
f(x,y)=xy \longrightarrow\frac{\partial f}{\partial x}=y,\frac{\partial f}{\partial y}=x
\\
f(x,y)=x+y\longrightarrow\frac{\partial f}{\partial x}=\frac{\partial f}{\partial y}=1
\\
f(x,y)=max(x,y)\longrightarrow\frac{\partial f}{\partial x}=\mathbb{I}(x\ge y),\frac{\partial f}{\partial y}=\mathbb{I}(y\ge x)
$$
$\mathbb{I}$是标识运算符，当括号内为真时，值为1，否则为0。

#### 对复合函数求导

使用链式法则：
$$
f(x,y,z)=(x+y)z,q=x+y
\\
f=qz,则\frac{\partial f}{\partial x}=\frac{\partial f}{\partial q}\frac{\partial q}{\partial x}=z\cdot1
$$

``` python
x = -2;y = 5; z = -4
q = x + y
f = q * z

dfdz = q
dfdq = z
dfdx = 1.0 * dfdq
dfdy = 1.0 * dfdq
```

各个gate的偏导依照链式法则相乘求得最终各变量的偏导。

*backpropagation is a beautifully local process*

图中每个gate在接受到输入数据后可以马上进行两件事：

1. 计算当前gate的output
2. 计算各个变量对这个output的局部偏导

网路跑完前馈之后，反向传播过程便将数据流过路径的偏导相乘起来。

#### sigmoid gate

$$
\sigma(x)=\frac{1}{1+e^{-x}}
\\
\frac{\partial \sigma(x)}{\partial x}=(1-\sigma(x))\sigma(x)(过程略)
$$

```python
sigmoid = 1.0 / (1 + math.exp(-x))
dsigdx = (1-sigmoid)*sigmoid
```

**有时候变量会在多个复合函数之中，不要忘记在最后将他们相加**

如：
$$
f(x,y)=\frac{x+\sigma(y)}{\sigma(x)+(x+y)^{2}}
$$

```python
sigy=1.0/(1+math.exp(-y))
num=x+sigy
sigx=1.0/(1+math.exp(-x))
xpy=x+y
xpysqr=xpy**2
den=sigx+xpysqr
invdev=1.0/den
f=num*invden

#backpropagation
dnum=invden
dinvden=num
dden=(-1.0/(den**2))*dinvden
dsigx=1*dden
dxpysqr=1*dden
dxpy=(2*xpy)*dxpysqr
dx=1*dxpy
dy=1*dxpy
dx+=((1-sigx)*sigx)*dsigx
dx+=1*dnum
dsigy=1*dnum
dy+=((1-sigy)*sigy)*dsigy
```

**一个隐性的问题**:如果一个很大的数与一个很小的数相乘时，根据乘法gate求导规则，很小的数对应的变量得到的梯度会很大，很大的数对应的变量得到的梯度会很小。比如在linear classifiers中，$w^{T}x_{i}$ 对于很大的输入，权重更新的梯度也会很大，则需要减小学习率来进行补偿。所以，需要在数据输入之前对其进行适当的预处理。