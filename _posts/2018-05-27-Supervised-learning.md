---
layout: post
title: Some Supervised learning models
author: Takchatsau
categories: notes
tags: [notes]
image: cybercity-1.jpg
---

## 监督学习(Supervised learning)的基本任务

监督学习的基本任务就是，对于给定的training set，需要训练出一个function h(被定义为hypothesis)，使得h(x)对y有较好的预测。

![](https://ws1.sinaimg.cn/large/86223c22ly1frq7dwbd8nj20c208dmxx.jpg)

* 当y是连续的，这个学习任务被称为**回归问题**。
* 当y是离散且有限的，这个学习任务被称为**分类问题**。

### 线性回归

![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/500px-Linear_regression.svg.png)

[^图源]: wikimedia

线性回归中hypothesis为一个线性函数：



$$
h_{\theta}(x)=\theta_{0}+\sum^{n}_{i=1}\theta_{i}x_{i}
$$



其中$\theta$是$h_{\theta}(x)$的参数，实现从x到y的线性映射，令$x_{0}=1$,则，$h_{\theta}(x)$变为：


$$
h_{\theta}(x)=\sum^{n}_{i=0}\theta_{i}x_{i}=\theta^{T}x
$$


为了衡量y和$h_{\theta}(x)$之间的误差，定义**cost function**:


$$
J(\theta)=\frac{1}{2}\sum^{m}_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^{2}
$$



#### LMS algorithm

利用梯度下降法更新参数来minimize$J(\theta)$。

对单个参数$\theta_{j}$:


$$
\theta_{j}:=\theta_{j}-\alpha\frac{\partial J(\theta)}{\partial \theta_{j}}
\\
\begin{align*}
\frac{\partial J(\theta)}{\partial \theta_{j}} &=\frac{1}{2}\frac{\partial }{\partial \theta_{j}}(h_{\theta}(x)-y)^{2}\\
&=(h_{\theta}(x)-y)\frac{\partial }{\partial \theta_{j}}(h_{\theta}(x)-y)\\
&=(h_{\theta}(x)-y)\frac{\partial }{\partial \theta_{j}}(\sum^{n}_{i=0}\theta_{i}x_{i}-y)\\
&=(h_{\theta}(x)-y)x_{j}\\
\end{align*}
$$


所以，对于单一样本：


$$
\theta_{j}:=\theta_{j}+\alpha(y^{(i)}-h_{\theta}(x^{(i)}))x^{(i)}_{j}
$$


对全体样本：


$$
\theta_{j}:=\theta_{j}+\alpha\sum_{i=1}^{m}(y^{(i)}-h_{\theta}(x^{(i)}))x^{(i)}_{j}
$$



* 批梯度下降(batch gradient descent)：每次参数更新遍历全体样本。
* 随机梯度下降(stochastic/incremental gradient descent)：每次参数更新只随机遍历一个样本。

![](https://cdn-images-1.medium.com/max/1600/1*PV-fcUsNlD9EgTIc61h-Ig.png)

[图源]: https://towardsdatascience.com/@ImadPh

####  迹运算的性质

定义:
$$
\sum^{m}_{i=1}A_{i,i}
$$


性质：


$$
trAB=trBA
\\
trABC=trCAB=trBCA
\\
\nabla_{A}trAB=B^{T}
\\
tr(a)=a
\\
\nabla_{A}trABA^{T}C=CAB+C^{T}AB^{T}
$$



证明$\nabla_{A}trAB=B^{T}$ :


$$
\begin{align*}
\frac{\partial trAB}{\partial A_{i,j}}&=\frac{\partial \sum^{m}_{k=1}(AB)_{k,k}}{\partial A_{i,j}}
\\
&=\frac{\partial (AB)_{i,i}}{\partial A_{i,j}}
\\
&=\frac{\partial \sum^{m}_{l=1}A_{i,l}B_{l,i}}{\partial A_{i,j}}
\\
&=\frac{\partial A_{i,j}B_{j,i}}{\partial A_{i,j}}
\\
&=B_{j,i}
\end{align*}
$$


所以$\nabla_{A}trAB=B^{T}$。

证明$\nabla_{A}trABA^{T}C=CAB+C^{T}AB^{T}$：


$$
\begin{align*}
\nabla_{A}trABA^{T}C&=\nabla_{A}tr\underbrace{AB}_{u(A)}\underbrace{A^{T}C}_{v(A^{T})}
\\
&=\nabla_{A:u(A)}tr(u(A)v(A^{T}))+\nabla_{A:v(A^{T})}tr(u(A)v(A^{T}))
\\
&=(v(A^{T}))^{T}\nabla_{A}u(A)+(\nabla_{A^{T}:v(A^{T})}tr(u(A)v(A^{T})))^{T}
\\
&=C^{T}AB^{T}+((u(A))^{T}\nabla_{A^{T}}v(A^{T}))^{T}
\\
&=C^{T}AB^{T}+(B^{T}A^{T}C^{T})^{T}
\\
&=C^{T}AB^{T}+CAB
\end{align*}
$$


#### 利用迹运算的性质推导J(θ)的梯度

将training set写成如下形式：


$$
X=\begin{bmatrix} (x^{(1)})^{T} \\ \vdots \\  (x^{(m)})^{T}
\end{bmatrix}
$$


m为training set数目。

则有：


$$
X\theta=\begin{bmatrix}
(x^{(1)})^{T}\theta \\ \vdots \\  (x^{(m)})^{T}\theta
\end{bmatrix}
=
\begin{bmatrix}
h_{\theta}(x^{(1)}) \\ \vdots \\  h_{\theta}(x^{(m)})
\end{bmatrix}
\\
J(\theta)=\frac{1}{2}(X\theta-y)(X\theta-y)^{T}
\\
\nabla_{\theta}J(\theta)=\frac{1}{2}\nabla_{\theta}(\theta^{T}X^{T}X\theta-\theta^{T}X^{T}y-y^{T}X\theta+y^{T}y)
$$


由于J(θ)为一个实数，则有：


$$
\begin{align*}
\nabla_{\theta}J(\theta)&=\frac{1}{2}\nabla_{\theta}tr(\theta^{T}X^{T}X\theta-\theta^{T}X^{T}y-y^{T}X\theta+y^{T}y)
\\
&=\frac{1}{2}\nabla_{\theta}tr(\theta\theta^{T}X^{T}X)-\nabla _{\theta}tr(y^{T}X\theta)
\\
\end{align*}
$$
其中：


$$
\begin{align*}
\nabla_{\theta}tr(\theta\theta^{T}X^{T}X)&=\nabla_{\theta}tr(\underbrace{\theta}_{A}\underbrace{I}_{B}\underbrace{\theta^{T}}_{A^{T}}\underbrace{X^{T}X}_{C})
\\
&=X^{T}X\theta I+X^{T}X\theta I
\end{align*}
$$




所以：


$$
\begin{align*}
\nabla_{\theta}J(\theta)
&=\frac{1}{2}\nabla_{\theta}tr(\theta\theta^{T}X^{T}X)-\nabla _{\theta}tr(y^{T}X\theta)
\\
&=X^{T}X\theta-X^{T}y
\end{align*}
$$


要使J(θ)达到最小，令$\nabla_{\theta}J(\theta)=0$，即得到normal equation：


$$
X^{T}X\theta=X^{T}y
$$


求解得：


$$
\theta=(X^{T}X)^{-1}X^{T}y
$$


#### LWR(Locally weighted linear regression)

cost function变成了：


$$
J(\theta)=\sum_{i}w^{(i)}(y^{(i)}-\theta^{T}x^{(i)})^{2}
\\
w^{(i)}=exp(-\frac{(x^{(i)}-x)^{2}}{2\tau^{2}})
$$


$w^{(i)}$被称为权重(weights)，$\tau$被称为**波长函数(bandwidth)**。

由于对于每一个x，都要重新计算一次weights，所以每求一个h(x)，θ也要再求一遍。

##### Parametric learning algorithm & Non-parametric learning 

* parametric learning algorithm：用若干个固定、有限的参数来拟合训练集的数据，训练完成后，只保存这些参数用于测试过程，而不会再参考训练集的数据。
* non-parametric learning algorithm：测试的过程需要利用训练集的数据进行计算，随着训练集数目的增长，hypothesis h的数目线性增长。

#### 用概率论解释J(θ)的意义

在目标变量y和h(x)之间引入误差项：


$$
y^{(i)}=\theta^{T}x^{(i)}+\epsilon^{(i)}
$$


误差项符合均值为0的高斯分布：


$$
\epsilon^{(i)}\thicksim \mathcal{N}(0,\sigma^{2})
$$


则y的条件概率可以表示为：


$$
P(y^{(i)}| x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^{T}x^{(i)})^{2}}{2\sigma^{2}})
$$


由于对每个训练样本的误差项是独立同分布(IID)的，那么整个训练集的似然(likelihood)可以写成：


$$
L(\theta)=\prod_{i=1}^{m}P(y^{(i)}| x^{(i)};\theta)
$$


取对数：


$$
\begin{align*}
l(\theta)=ln(L(\theta))&=\sum^{m}_{i=1}lnP(y^{(i)}| x^{(i)};\theta)
\\
&=-mln(\sqrt{2\pi}\sigma)-\frac{1}{2\sigma^{2}}\sum^{m}_{i=1}(y^{(i)}-\theta^{T}x^{(i)})^{2}
\end{align*}
\\
so,maximize \ L(\theta) \ \iff \ minimize \ J(\theta)=\frac{1}{2}\sum^{m}_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^{2}
$$


### 逻辑回归(Logistic regression)

y只取0和1两个值，h(x)输出y=1的概率值：


$$
h_{\theta}(x)=g(\theta^{T}x)=\frac{1}{1+e^{-\theta^{T}x}}
$$


![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sigmoid-function-2.svg/500px-Sigmoid-function-2.svg.png)

[^图源]: wikimedia

g(z)的导数：


$$
\begin{align*}
\frac{dg(z)}{dz}&=\frac{e^{-z}}{(1+e^{-z})^{2}}
\\
&=\frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}}) 
\\
&=g(z)(1-g(z))
\end{align*}
$$


假设：


$$
P(y=1|x;\theta)=h_{\theta}(x)
\\
P(y=0|x;\theta)=1-h_{\theta}(x)
$$


那么：


$$
p(y|x;\theta)=(h_{\theta}(x))^{y}(1-h_{\theta}(x))^{1-y}
$$


似然函数：


$$
\begin{align*}
L(\theta)&=\prod^{m}_{i=1}p(y^{(i)}|x^{(i)};\theta)
\\
&=\prod^{m}_{i=1}(h_{\theta}(x^{(i)}))^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}
\end{align*}
\\
l(\theta)=ln(L(\theta))=\sum^{m}_{i=1}y^{(i)}lnh(x^{(i)})+(1-y^{(i)})ln(1-h(x^{(i)}))
$$


求导：


$$
\begin{align*}
\frac{\partial l(\theta)}{\partial\theta_{j}}&=(y\frac{1}{g(\theta^{T}x)}-(1-y)\frac{1}{1-g(\theta^{T}x)})\frac{\partial g(\theta^{T}x)}{\partial \theta_{j}} 
\\
&=(y\frac{1}{g(\theta^{T}x)}-(1-y)\frac{1}{1-g(\theta^{T}x)})g(\theta^{T}x)(1-g(\theta^{T}x))\frac{\partial \theta^{T}x}{\partial \theta_{j}} 
\\
&=(y(1-g(\theta^{T}x))-(1-y)g(\theta^{T}x))x_{j}
\\
&=(y-h_{\theta}(x))x_{j}
\end{align*}
$$


所以梯度下降法的update rule为：


$$
\theta_{j}:=\theta_{j}+\alpha\sum_{i=1}^{m}(y^{(i)}-h_{\theta}(x^{(i)}))x^{(i)}_{j}
$$



#### 感知机算法(The perceptron learning algorithm)


$$
h_{\theta}(x)=g(\theta^{T}x)
\\
where \quad g(z)=\left\{\begin{matrix}1\quad if\ z\ge0\\0\quad if\ z<0\end{matrix}
\right.
$$


其实，g(z)可由sigmoid函数取极限求得：


$$
g(z)=\lim_{\beta\rightarrow\infty}\sigma(\beta z)=\frac{1}{1+e^{-\beta z}}\\
$$


所以：


$$
\begin{align*}
&\quad\frac{\partial l(\theta)}{\partial\theta_{j}}\\
&=\lim_{\beta\rightarrow \infty}(y\frac{1}{\sigma(\theta^{T}\beta x)}-(1-y)\frac{1}{1-\sigma(\theta^{T}\beta x)})\sigma(\theta^{T}\beta x)(1-\sigma(\theta^{T}\beta x))\frac{\partial \theta^{T}\beta x}{\partial \theta_{j}} 
\\
&=\lim_{\beta\rightarrow\infty}(y(1-\sigma(\theta^{T}\beta x))-(1-y)\sigma(\theta^{T}\beta x))\beta x_{j}
\\
&=\lim_{\beta\rightarrow\infty}(y-h_{\theta}(x))\beta x_{j}
\end{align*}
$$


令β>0，则向量$v_{j}=(y-h_{\theta}(x))x_{j}​$的方向与l(θ)梯度$\nabla_{\theta}l(\theta)​$的方向相同，所以感知机的update rule仍然可以写成：


$$
\theta_{j}:=\theta_{j}+\alpha\sum_{i=1}^{m}(y^{(i)}-h_{\theta}(x^{(i)}))x^{(i)}_{j}
$$


### 牛顿法求极值

![](https://ecourses.ou.edu/ebook/math/ch03/sec034/media/dia03422.gif)

目标：f(θ)  find  θ  s.t.  f(θ)=0.


$$
\theta:=\theta-\frac{f(\theta)}{f^{'}(\theta)}
$$


对于l(θ)，其在最大值处的导数为0，所以update rule：


$$
\theta:=\theta-\frac{l^{'}(\theta)}{l^{''}(\theta)}
$$


写成矩阵形式：


$$
\theta:=\theta-H^{-1}\nabla_{\theta}l(\theta)
$$


H为Hessian矩阵：


$$
H_{i,j}=\frac{\partial^{2}l(\theta)}{\partial \theta_{i}\partial\theta_{j}}
$$


*<u>Newton method虽然收敛得很快，但面对数目较大的训练集，需要计算一个很大的Hessian 矩阵并求逆，反而降低了更新的速度。</u>*

### 广义线性模型(Generalized Linear Models)

#### 指数分布族(The exponential family)

分布函数具有以下形式：


$$
p(y;\eta)=b(y)exp(\eta^{T}T(y)-a(\eta))
$$


其中，η被称为自然参数(natural parameter)，T(y)被称为充分统计量(sufficient statistic，一般T(y)=y)，a(η)为log partition function，使得分布函数的积分为1。

##### Bernoulli分布：

对Ber(Φ)：P(y=1;Φ)=Φ，P(y=0;Φ)=1-Φ，则：


$$
\begin{align*}
P(y;\phi)&=\phi^{y}(1-\phi)^{(1-y)}
\\
&=exp[yln\phi+(1-y)ln(1-\phi)]
\\
&=exp[\underbrace{y}_{T(y)}\underbrace{ln\frac{\phi}{1-\phi}}_{\eta}+\underbrace{ln(1-\phi)}_{a(\eta)}]
\end{align*}
\\
\eta=ln\frac{\phi}{1-\phi}\ \Longrightarrow \ \phi=\frac{1}{1+e^{-\eta}}
\\
a(\eta)=-ln(1-\phi)=ln(1+e^{-\eta})
$$


##### Gaussian分布：


$$
\begin{align*}
p(y;\mu,\sigma^{2})&=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y-\mu)^{2}}{2\sigma^{2}})
\\
&=\underbrace{\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{y^{2}}{2\sigma^{2}})}_{b(y)}exp(\underbrace{\mu}_{\eta}\underbrace{\frac{y}{\sigma^{2}}}_{T(y)}-\underbrace{\frac{\mu^{2}}{2\sigma^{2}}}_{a(\eta)})
\end{align*}
\\
a(\eta)=\frac{\eta^{2}}{2\sigma^{2}}
$$

#### GLM:

* 假设$y\vert x;θ$ 满足指数族分布
* 对给定x，输出目标为$E[T(y)\vert x]$，所以拟合目标为$h(x)=E[T(y)\vert x]$
* η是x的线性运算结果:


$$
\eta=\theta^{T}x(\eta_{i}=\theta_{i}^{T}x , if \ \eta\in\mathbb{R}^{k})
$$


*对于二分类问题*：


$$
h_{\theta}(x)=E[y|x;\theta]=P(y=1|x;\theta)=\phi=\frac{1}{1+e^{-\eta}}=\frac{1}{1+e^{-\theta^{T}x}}
$$


#### Softmax Regression

对于多分类问题，$y\in \{1,\dots,k\}$，则可以让GLM输出k-1个参数：$\phi_{1}\dots\phi_{k-1}$，另外$\phi_{k}=1-\sum^{k-1}_{i=1}\phi_{i}$，同样令$P(y=i)=\phi_{i}$，y的分布为多项式分布，为了让其能表达成指数分布族的形式，将T(y)写成向量形式：


$$
T(1)=\begin{bmatrix}1\\0\\0\\\vdots\\0\end{bmatrix},\dots,T(k-1)=\begin{bmatrix}0\\0\\0\\\vdots\\1\end{bmatrix},
\dots,T(k)=\begin{bmatrix}0\\0\\0\\\vdots\\0\end{bmatrix}\\
T(y)\in\mathbb{R}^{k-1}
$$


引入符号1{.}(1{True}=1,1{False}=0),则：


$$
(T(y)){i}=1{y=i},E[(T(y)){i}]=P(y=i)=\phi_{i}
$$
证明p(y;Φ)可写成指数分布族的形式：


$$
\begin{align*}
p(y;\phi)&=\prod^{k}_{i=1}\phi_{i}^{1\{y=i\}}\\
&=\phi_{k}^{1-\sum^{k-1}_{l=1}(T(y))_{l}}\prod^{k-1}_{i=1}\phi_{i}^{(T(y))_{i}}\\
&=exp[\sum^{k-1}_{i=1}(T(y))_{i}ln(\phi_{i})+(1-\sum^{k-1}_{l=1}(T(y))_{l})ln(\phi_{k})]\\
&=exp[\sum^{k-1}_{i=1}(T(y))_{i}ln(\frac{\phi_{i}}{\phi_{k}})+ln(\phi_{k})]\\
&=b(y)exp[\eta^{T}T(y)-a(\eta)]
\end{align*}
\\
\quad \\
\quad \\
where\quad\eta_{i}=ln(\frac{\phi_{i}}{\phi_{k}}),a(\eta)=-ln(\phi_{k}),b(y=1)
$$


然后：


$$
\begin{align*}
e^{\eta_{i}}&=\frac{\phi_{i}}{\phi_{k}}\\
\phi_{k}e^{\eta_{i}}&=\phi_{i}\\
\phi_{k}\sum^{k}_{i=1}e^{\eta_{i}}&=\sum^{k}_{i=1}\phi_{i}=1\\
\phi_{k}&=\frac{1}{\sum^{k}_{i=1}e^{\eta_{i}}}
\end{align*}
$$


所以：


$$
\begin{align*}
p(y=i|x;\theta)&=\phi_{i}\\
&=\frac{e^{\eta_{i}}}{\sum^{k}_{j=1}e^{\eta_{j}}}\\
&=\frac{e^{\theta^{T}_{i}x}}{\sum^{k}_{j=1}e^{\theta_{j}^{T}x}}
\end{align*}
$$



h(x)表示y=i的概率，所以：


$$
\begin{align*}
h_{\theta}(x)&=E[T(y)|x;\theta]\\
&=E\begin{bmatrix}\begin{matrix}1\{y=1\}\\\vdots\\1\{y=k-1\}\end{matrix}\Bigg | x;\theta\end{bmatrix}\\
&=\begin{bmatrix}\phi_{i}\\\vdots\\\phi_{k-1}\end{bmatrix}\\
&=\begin{bmatrix}\frac{exp(\theta_{1}^{T}x)}{\sum^{k}_{j=1}exp(\theta^{T}_{j}x)}\\\vdots\\\frac{exp(\theta_{k-1}^{T}x)}{\sum^{k}_{j=1}exp(\theta^{T}_{j}x)}\end{bmatrix}
\end{align*}
$$


而：


$$
p(y=k|x;\theta)=\phi_{k}=1-\sum^{k-1}_{i=1}\phi_{i}
$$


似然函数l(θ)：


$$
\begin{align*}
l(\theta)&=\sum^{m}_{i=1}logp(y^{(i)}|x^{(i)};\theta)\\
&=\sum^{m}_{i=1}log\prod^{k}_{l=1}\begin{pmatrix}\frac{e^{\theta_{l}^{T}x^{(i)}}}{\sum^{k}_{i=1}e^{\theta_{j}^{T}x^{(i)}}}\end{pmatrix}^{1\{y^{(i)}=l\}}
\end{align*}
$$


然后，可用梯度上升或者牛顿法来求得θ，使得似然函数l(θ)取得极大值。
