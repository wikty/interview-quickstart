# Introduction

In the linear regression, we know how to predict **continuous variable** (e.g., housing prices) as a linear function of input values (e.g., the size of the house). Sometimes we will instead wish to predict a **discrete variable** such as predicting whether a grid of pixel intensities represents a “0” digit or a “1” digit. This is a **classification problem**. **Logistic regression** is a simple classification algorithm for learning to make such decisions.

In logistic regression we use a hypothesis class (different from linear regression) to try to predict the probability that a given example belongs to the “1” class versus the probability that it belongs to the “0” class. Specifically, we will try to learn a function of the form:
$$
\begin{align}
P(y=1|x) &= h_\theta(x) = \frac{1}{1 + \exp(-\theta^\top x)} \equiv \sigma(\theta^\top x),\\
P(y=0|x) &= 1 - P(y=1|x) = 1 - h_\theta(x).
\end{align}
$$
where $\sigma(z) \equiv \frac{1}{1 + \exp(-z)}$ is often called the “sigmoid” or “logistic” function – it is an S-shaped function that “squashes” the value of $z$ into the range $[0, 1]$ so that we may interpret $h_{\theta}(x)$ as a probability.

Our goal is to search for a value of $\theta$ so that the probability $P(y=1|x) = h_\theta(x)$ is large when $x$ belongs to the “1” class and small when $x$ belongs to the “0” class (so that $P(y=0|x)$ is large). The cost function as follows:
$$
J(\theta) = - \sum_i^m \left(y^{(i)} \log( h_\theta(x^{(i)}) ) + (1 - y^{(i)}) \log( 1 - h_\theta(x^{(i)}) ) \right).
$$
Then we want to minimize $J(\theta)$ by the choice of $\theta$. Note that only one of the two terms in the summation is non-zero for each training example

To minimize $J(\theta)$ we can use the same tools as for linear regression. We need to provide a function that computes $J(\theta)$ and $\nabla_{\theta} J(\theta)$ for any requested choice of $\theta$. The derivative of $J(\theta)$ as given above with respect to $\theta_j$ is:
$$
\frac{\partial J(\theta)}{\partial \theta_j} = \sum_i x^{(i)}_j (h_\theta(x^{(i)}) - y^{(i)}).
$$
And we can write $\nabla_{\theta} J(\theta)$ as vector form:
$$
\nabla_\theta J(\theta) = \sum_i x^{(i)} (h_\theta(x^{(i)}) - y^{(i)})
$$
This is essentially the same as the gradient for linear regression except that now $h_\theta(x) = \sigma(\theta^\top x)​$.