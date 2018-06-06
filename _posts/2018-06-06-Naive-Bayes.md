---
layout: post
title: 生成模型与朴素贝叶斯方法
author: Takchatsau
categories: notes
tags: [notes]
image: cyber-girl.jpg
---

## 生成模型

对$p(y)$和$p(x\arrowvert y)$建模，然后利用贝叶斯法则来求得$p(y\arrowvert x)$：


$$
p(y\arrowvert x)=\frac{p(x\arrowvert y)p(y)}{p(x)}
\\
p(x)=p(x\arrowvert y=1)p(y=1)+p(x\arrowvert y=0)p(y=0)
$$


但在做似然估计的时候不需要直接算$p(y\arrowvert x)$：


$$
\begin{align*}
arg\max_{y}p(y\arrowvert x)&=arg\max_{y}{\frac{p(x\arrowvert y)p(y)}{p(x)}}\\
&=arg\max_{y}{p(x\arrowvert y)p(y)}
\end{align*}
$$


### 高斯判别分析(Gaussian discriminant analysis)

#### 多元高斯正态分布

n维高斯正态分布，参数为一个均值n维向量$n\in R^{n}$，以及一个协方差矩阵$\Sigma\in R^{n\times n}$，其中$\Sigma\ge0$是一个对称的半正定矩阵。多元高斯分布可用N(μ，Σ)来表示，密度函数为：


$$
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu))\\
where,E[X]=\int_{x}xp(x;\mu,\Sigma)dx=\mu
\\
Cov(X)=E[(X-E(X))(X-E(X))^{T}]=\Sigma
$$

#### 高斯判别分析模型(Gaussian Discriminant Analysis model)

对于分类问题，输入特征x是一系列的连续随机变量，可使用高斯判别分析模型，其中对$p(x\vert y)$用多元正态分布来进行建模：


$$
y\ \sim\ Bernoulli(\phi)\\
x\vert y=0\ \sim\ \mathcal{N}(\mu_{0},\Sigma)\\
x\vert y=1\ \sim\ \mathcal{N}(\mu_{1},\Sigma)
$$


即：


$$
\begin{align*}
p(y)&=\phi^{y}(1-\phi)^{1-y}\\
p(x\vert y=0)&=\frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x-\mu_{0})^{T}\Sigma^{-1}(x-\mu_{0}))\\
p(x\vert y=1)&=\frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x-\mu_{1})^{T}\Sigma^{-1}(x-\mu_{1}))\\
\end{align*}
$$


似然函数：


$$
\begin{align*}
l(\phi,\mu_{0},\mu_{1},\Sigma)&=log\prod^{m}_{i=1}p(x^{(i)},y^{(i)};\phi,\mu_{0},\mu_{1},\Sigma)\\
&=log\prod^{m}_{i=1}p(x^{(i)}\vert y^{(i)};\mu_{0},\mu_{1},\Sigma)p(y^{(i)};\phi)\\
\end{align*}
$$


求解得参数最大似然估计：


$$
\phi=\frac{1}{m}\sum^{m}_{i=1}1\{y^{(i)}=1\}\\
\mu_{0}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=0\}x^{(i)}}{\sum^{m}_{i=1}1\{y^{(i)}=0\}}\\
\mu_{1}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=1\}x^{(i)}}{\sum^{m}_{i=1}1\{y^{(i)}=1\}}\\
\Sigma=\frac{1}{m}\sum^{m}_{i=1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^{T}
$$


模型的结果如下图所示：

![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229note2f6.png)

##### GDA 与 logistic regression

如果把$p(y=1\vert x;\phi,\mu_{0},\mu_{1})$作为x的一个函数，可以得到以下形式：


$$
p(y=1\vert x;\phi,\mu_{0},\mu_{1})=\frac{1}{1+exp(-\theta^{T}x)}
$$


其中θ是对$\phi,\Sigma,\mu_{0},\mu_{1}$的函数。

<u>*如果$p(x\vert y)$是一个多元高斯分布，那么$p(y\vert x)$可由一个逻辑函数表达，但反之命题不成立。*</u>因为，对泊松分布$x\vert y=0\sim Possion(\lambda_{1})$，也可以推出$p(x\vert y)$适合逻辑回归。

因而GDA比logistic regression的假设更加严格，在数据接近或符合GDA模型假设的情况下，对数据的利用更加有效。而logistic regression的假设更弱，更加具有鲁棒性。

### 朴素贝叶斯方法

对$x_{j}\in\{0,1\}$，而$x\in\mathbb{R}^{n}$，利用贝叶斯法则：


$$
\begin{align*}
&\quad\ p(x_{1},\dots,x_{n}\vert y)\\
&=p(x_{1}\vert y)p(x_{2}\vert y,x_{1})\dots p(x_{n}\vert y,x_{1},\dots,x_{n})
\end{align*}
$$


朴素贝叶斯假设：x各个分量相互独立。则上式变为：


$$
p(x_{1},\dots,x_{n})=\prod^{m}_{j=1}p(x_{j}\vert y)
$$


给出模型参数：$\phi_{j\vert y=1}=p(x_{j}=1\vert y),\phi_{j\vert y=0}=p(x_{j}=0\vert y),\phi_{y}=p(y=1)$，则联合似然函数为：


$$
\begin{align*}
&l(\phi_{y},\phi_{j\vert y=0},\phi_{j\vert y=1})\\
&=ln\prod^{m}_{i=1}p(x^{(i)},y^{(i)};\phi_{y},\phi_{j\vert y=0},\phi_{j\vert y=1})\\
&=ln\prod^{m}_{i=1}(\prod^{n}_{j=1}p(x^{(i)}_{j}\vert y^{(i)};\phi_{y},\phi_{j\vert y=0},\phi_{j\vert y=1}))p(y^{(i)};\phi_{y},\phi_{j\vert y=0},\phi_{j\vert y=1})\\
&=\sum^{m}_{i=1}\lbrack lnp(y^{(i)};\phi_{y},\phi_{j\vert y=0},\phi_{j\vert y=1})+\sum^{n}_{j=1}lnp(x^{(i)}_{j}\vert y^{(i)};\phi_{y},\phi_{j\vert y=0},\phi_{j\vert y=1})\rbrack\\
&=\sum^{m}_{i=1}\lbrack y^{(i)}ln\phi_{y}+(1-y^{(i)})ln(1-\phi_{y})\\&\quad\quad+\sum^{n}_{j=1}(x^{(i)}_{j}ln\phi_{j\vert y^{(i)}}+(1-x_{j}^{(i)})ln(1-\phi_{j\vert y^{(i)}}))\rbrack\\
\end{align*}
$$


求出各参数的最大似然估计：


$$
\begin{align*}
&\nabla_{\phi_{j\vert y=0}}l(\phi_{y},\phi_{j\vert y=0},\phi_{j\vert y=1})\\
&=\nabla_{\phi_{j\vert y=0}}\sum^{m}_{i=1}(x^{(i)}_{j}ln(\phi_{j\vert y=0})1\{y^{(i)}=0\}\\&\quad \quad +(1-x^{(i)}_{j})ln(1-\phi_{j\vert y=0})1\{y^{(i)}=0\})\\
&=\sum^{m}_{i=1}(x^{(i)}_{j}\frac{1}{\phi_{j\vert y=0}}1\{y^{(i)}=0\} -(1-x^{(i)}_{j})\frac{1}{1-\phi_{j\vert y=0}}1\{y^{(i)}=0\})
\end{align*}
$$


令上式为0即得：


$$
\begin{align*}
\sum^{m}_{i=1}x^{(i)}_{j}\frac{1}{\phi_{j\vert y=0}}1\{y^{(i)}=0\}&=\sum^{m}_{i=1}(1-x^{(i)}_{j})\frac{1}{1-\phi_{j\vert y=0}}1\{y^{(i)}=0\}\\
\sum^{m}_{i=1}x^{(i)}_{j}(1-\phi_{j\vert y=0})1\{y^{(i)}=0\}&=\sum^{m}_{i=1}(1-x^{(i)}_{j})\phi_{j\vert y=0}1\{y^{(i)}=0\}\\
\sum^{m}_{i=1}x^{(i)}_{j}1\{y^{(i)}=0\}&=\sum^{m}_{i=1}\phi_{j\vert y=0}1\{y^{(i)}=0\}\\
\sum^{m}_{i=1}1\{y^{(i)}=0\land x^{(i)}_{j}=1\}&=\phi_{j\vert y=0}\sum^{m}_{i=1}1\{y^{(i)}=0\}\\
\end{align*}
\\\quad\\\quad\\
\Longrightarrow \phi_{j\vert y=0}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=0\land x^{(i)}_{j}=1\}}{\sum^{m}_{i=1}1\{y^{(i)}=0\}}
$$


同理可得：


$$
\phi_{j\vert y=1}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=1\land x^{(i)}=1\}}{\sum^{m}_{i=1}1\{y^{(i)}=1\}}
$$


对$\phi_{y}$：


$$
\nabla_{\phi_{y}}l(\phi_{y},\phi_{j\vert y=0},\phi_{j\vert y=1})=\sum^{m}_{i=1}(y^{(i)}\frac{1}{\phi_{y}}-(1-y^{(i)})\frac{1}{1-\phi_{y}})
$$


令上式为0，有：


$$
\sum^{m}_{i=1}y^{(i)}=\sum^{m}_{i=1}\phi_{y}\Longrightarrow \phi_{y}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=1\}}{m}
$$


所以对于新样本，可以计算其后验概率：


$$
\begin{align*}
p(y=1\vert x)&=\frac{p(x\vert y=1)p(y=1)}{p(x)}\\
&=\frac{\prod^{n}_{j=1}\phi_{j\vert y=1}\phi_{y}}{\prod^{n}_{j=1}\phi_{j\vert y=0}(1-\phi_{y})+\prod^{n}_{j=1}\phi_{j\vert y=1}\phi_{y}}
\end{align*}
$$


对于特征向量为多个离散值，即$x_{j}\in \{1,\dots,k\}$，也可以使用朴素贝叶斯方法，只需将$p(x_{j}\vert y)$从伯努利分布改成多项式分布。

#### 拉普拉斯光滑(Laplace smoothing)

对于新的输入样本中，出现了新的特征$x_{k}$，由于对训练集从未出现该特征，针对该特征的参数的最大似然估计均为0：


$$
\phi_{k\vert y=0}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=0\land x^{(i)}_{k}=1\}}{\sum^{m}_{i=1}1\{y^{(i)}=0\}}=0\\
\phi_{k\vert y=1}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=1\land x^{(i)}_{k}=1\}}{\sum^{m}_{i=1}1\{y^{(i)}=1\}}=0
$$


则显然，


$$
\begin{align*}
p(y=1\vert x)&=\frac{\phi_{k\vert y=1}\prod^{n}_{j=1}\phi_{j\vert y=1}\phi_{y}}{\phi_{k\vert y=1}\prod^{n}_{j=1}\phi_{j\vert y=0}(1-\phi_{y})+\phi_{k\vert y=0}\prod^{n}_{j=1}\phi_{j\vert y=1}\phi_{y}}\\
&=\frac{0}{0}=NaN
\end{align*}
$$


为了避免以上情况，引入拉普拉斯光滑：


$$
\phi_{j}=\frac{\sum^{m}_{i=1}1\{z^{(i)}=j\}}{m+k}\\
where\quad z^{(i)}_{j}\in\{1,\dots,k\}
$$


则对于之前的最大似然估计可写成：


$$
\phi_{k\vert y=0}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=0\land x^{(i)}_{k}=1\}+1}{\sum^{m}_{i=1}1\{y^{(i)}=0\}+2}\\
\phi_{k\vert y=1}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=1\land x^{(i)}_{k}=1\}+1}{\sum^{m}_{i=1}1\{y^{(i)}=1\}+2}
$$


#### 多项式事件模型

这里，用$x_{j}$来表示邮件中第j个单词，将词汇表映射到集合$\{1,\dots,\vert V\vert\}$，则$x_{j}\in\{1,\dots,\vert V\vert\} $，由n个单词组成的邮件可表示为向量$(x_{1},\dots,x_{n})$。而且引入假设：对于任何j的值，$p(x_{j}\vert y)$都相等，即单词的分布与词的位置无关。

则，似然函数：


$$
\begin{align*}
\mathcal{L}(\phi,\phi_{k\vert y=0},\phi_{k\vert y=1})&=\prod^{m}_{i=1}p(x^{(i)},y^{(i)})\\
&=\prod^{m}_{i=1}(\prod^{n_{i}}_{j=1}p(x^{(i)}_{j}\vert y;\phi_{k\vert y=0},\phi_{k\vert y=1}))p(y^{(i)};\phi_{y})
\end{align*}
$$


参考朴素贝叶斯方法的推导可得到最大似然的参数：


$$
\phi_{k\vert y=0}=\frac{\sum^{m}_{i=1}\sum^{n_{i}}_{j=1}1\{y^{(i)}=0\land x^{(i)}_{j}=k\}}{\sum^{m}_{i=1}1\{y^{(i)}=0\}n_{i}}\\
\phi_{j\vert y=1}=\frac{\sum^{m}_{i=1}\sum^{n_{i}}_{j=1}1\{y^{(i)}=1\land x^{(i)}_{j}=k\}}{\sum^{m}_{i=1}1\{y^{(i)}=1\}n_{i}}\\
\phi_{y}=\frac{\sum^{m}_{i=1}1\{y^{(i)}=1\}}{m}
$$


引入拉普拉斯光滑:


$$
\phi_{k\vert y=0}=\frac{\sum^{m}_{i=1}\sum^{n_{i}}_{j=1}1\{y^{(i)}=0\land x^{(i)}_{j}=k\}+1}{\sum^{m}_{i=1}1\{y^{(i)}=0\}n_{i}+\vert V\vert}\\
\phi_{j\vert y=1}=\frac{\sum^{m}_{i=1}\sum^{n_{i}}_{j=1}1\{y^{(i)}=1\land x^{(i)}_{j}=k\}+1}{\sum^{m}_{i=1}1\{y^{(i)}=1\}n_{i}+\vert V \vert}
$$
尽管以上方法的假设对问题进行了很大的简化，但性能通常出乎意料地好，所以可通常作为解决问题的首发选择。

