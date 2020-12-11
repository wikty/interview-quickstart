[TOC]



# Introduction

**Softmax regression** (or **multinomial logistic regression**) is a generalization of logistic regression to the case where we want to handle multiple classes. In logistic regression we assumed that the labels were binary: $y^{(i)} \in \{0,1\}$. Softmax regression allows us to handle $y^{(i)} \in \{1,\ldots,K\}$ where $K$ is the number of classes.

In the softmax regression setting, we are interested in multi-class classification (as opposed to only binary classification), and so the label $y$ can take on $K$ different values, rather than only two. Thus, in our training set $\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$ where $K$ is the number of classes. For example, in the MNIST digit recognition task, we would have $K=10$ different classes.

Given a test input $x$, we want our hypothesis to estimate the probability that $P(y=k|x)$ for each value of $k=1,2,\ldots,K$. I.e., we want to estimate the probability of the class label taking on each of the $K$ different possible values. Thus, our hypothesis will output a K-dimensional vector (whose elements sum to 1) giving us our $K$ estimated probabilities. Our hypothesis takes form:
$$
\begin{align}
h_\theta(x) =
\begin{bmatrix}
P(y = 1 | x; \theta) \\
P(y = 2 | x; \theta) \\
\vdots \\
P(y = K | x; \theta)
\end{bmatrix}
=
\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) }}
\begin{bmatrix}
\exp(\theta^{(1)\top} x ) \\
\exp(\theta^{(2)\top} x ) \\
\vdots \\
\exp(\theta^{(K)\top} x ) \\
\end{bmatrix}
\end{align}
$$
I.e. $P(y=k|x;\theta)$ takes form:
$$
P(y=k|x;\theta)=\frac{\exp(\theta^{(k)\top} x)}{\sum_{j=1}^K \exp(\theta^{(j)\top} x)}
$$
where $\theta^{(1)},\theta^{(2)},\ldots, \theta^{(K)}$ are the parameters of model. The denominator is a normalization term.

For convenience, we will also write $\theta$ (it's a n-by-K matrix) to denote all the parameters of our model:
$$
\theta = \left[\begin{array}{cccc}| & | & | & | \\
\theta^{(1)} & \theta^{(2)} & \cdots & \theta^{(K)} \\
| & | & | & |
\end{array}\right]
$$

# Learning

## Objective function

First, we can write the likelihood function of training examples:
$$
\begin{split}
L(\theta) &= \prod_i^m \prod_k^K P(y^{(i)}=k|x^{(i)};\theta) \\
		  &= \prod_i^m \prod_k^K \left( \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})} \right)^{1\{y^{(i)}=k\}}
\end{split}
$$
where $1\{.\}$ is the indicator function. Then our cost function is defined as the negative log-likelihood function:
$$
\begin{split}
J(\theta) &= -\log L(\theta) \\
		  &= -\sum_i^m \sum_k^K 1\{ y^{(i)}=k \} \log \left( \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})} \right)
\end{split}
$$

## Optimization

We cannot solve for the minimum of $J(\theta)$ analytically, and thus as usual we’ll resort to an iterative optimization algorithm. Taking derivatives, one can show that the gradient is:
$$
\begin{align}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right]  }
\end{align}
$$
Note $\theta^{(k)}$ is a vector, thus $\nabla_{\theta^{(k)}} J(\theta)$ is also a vector.

# Overparameterized

Softmax regression has an unusual property that it has a “redundant” set of parameters. To explain what this means, suppose we take each of our parameter vectors $\theta^{(j)}$, and subtract some fixed vector $\psi$ from it, so that every $\theta^{(j)}$ is now replaced with $\theta^{(j)} - \psi$ for every $j=1, \ldots, k$
$$
\begin{align}
P(y^{(i)} = k | x^{(i)} ; \theta)
&= \frac{\exp((\theta^{(k)}-\psi)^\top x^{(i)})}{\sum_{j=1}^K \exp( (\theta^{(j)}-\psi)^\top x^{(i)})}  \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})} \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}.
\end{align}
$$
In other words, subtracting $\psi$ from every $\theta^{(j)}$ does not affect our hypothesis’ predictions at all! More formally, we say that our softmax model is ”**‘overparameterized**,”’ meaning that for any hypothesis we might fit to the data, there are multiple parameter settings that give rise to exactly the same hypothesis function $h_{\theta}$ mapping from inputs $x$ to the predictions. 

Further, if the cost function $J(\theta)$ is minimized by some setting of the parameters $(\theta^{(1)}, \theta^{(2)},\ldots, \theta^{(k)})$, then it is also minimized by $(\theta^{(1)} - \psi, \theta^{(2)} - \psi,\ldots, \theta^{(k)} - \psi)$ for any value of $\psi$. Thus, the minimizer of $J(\theta)$ is not unique. Interestingly, $J(\theta)$ is still convex, and thus gradient descent will not run into local optima problems. But the Hessian is singular/non-invertible, which causes a straightforward implementation of Newton’s method to run into numerical problems.

If we set $\psi=\theta^{(K)}$, then the parameters are changed to: $(\theta^{(1)} - \theta^{(K)}, \theta^{(2)} - \theta^{(K)},\ldots,  \vec{0})$ which have the same solution with the parameters $(\theta^{(1)}, \theta^{(2)},\ldots, \theta^{(K)})$. Indeed, rather than optimizing over the $K \cdot n$ parameters $(\theta^{(1)}, \theta^{(2)},\ldots, \theta^{(K)})$, one can instead set $\theta^{(K)}=\vec{0}$ and optimize only with respect to the $(K-1) \cdot n$ remaining parameters.

# Relationship to Logistic Regression

In the special case where $K=2$, one can show that softmax regression reduces to logistic regression. This shows that softmax regression is a generalization of logistic regression. Concretely, when $K=2$, the softmax regression hypothesis outputs:
$$
\begin{align}
h_\theta(x) &=

\frac{1}{ \exp(\theta^{(1)\top}x)  + \exp( \theta^{(2)\top} x^{(i)} ) }
\begin{bmatrix}
\exp( \theta^{(1)\top} x ) \\
\exp( \theta^{(2)\top} x )
\end{bmatrix}

\end{align}
$$
Taking advantage of the fact that this hypothesis is overparameterized and setting $\psi = \theta^{(2)}$:
$$
\begin{align}
h(x) &=

\frac{1}{ \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) + \exp(\vec{0}^\top x) }
\begin{bmatrix}
\exp( (\theta^{(1)}-\theta^{(2)})^\top x )
\exp( \vec{0}^\top x ) \\
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\frac{\exp( (\theta^{(1)}-\theta^{(2)})^\top x )}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) }
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
1 - \frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\end{bmatrix}
\end{align}
$$
