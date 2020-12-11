# Introduction

The main idea of this section is to get familiar with objective functions, computing their gradients and optimizing the objectives over a set of parameters.

Our goal in **linear regression** is to predict a target value $y \in \R$ starting from a vector of input values $\mathbf{x} \in \R^n$ given training examples $\{ (x^{(1), y^{(1)}}), (x^{(2), y^{(2)}}), \ldots, (x^{(m), y^{(m)}})\}$, i.e. to find a function $y=h(x)$ so that we have $y^{(i)} \approx h(x^{(i)})$ for each training example, and also we hope $h(x)$ is a good predictor of the examples not in the training set.

The representation of function $h(x)$ is a linear function:
$$
h_\theta(x) = \sum_j \theta_j x_j = \mathbf{\theta}^\top x
$$
where $\theta$ is the parameters for the function; thus $h(x)$ represents a large family of functions parametrized by the choice of $\theta$.

Our cost function as follows:
$$
J(\theta) = \frac{1}{2} \sum_i^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 = \frac{1}{2} \sum_i^m \left( \theta^\top x^{(i)} - y^{(i)} \right)^2
$$
Then We want to find the choice of $\theta$ that minimizes $J(\theta)$.

# Optimization

There are many algorithms for minimizing functions like $J(\theta)​$ and we will describe some very effective ones that are easy to implement in later. For now, let’s take for granted the fact that most commonly-used algorithms for function minimization require us to provide two pieces of information about $J(\theta)​$:

- we need to compute cost function: $J(\theta)$ for any choice of $\theta$. 
- we need to compute the gradient of cost function: $\nabla_\theta J(\theta)$ for any choice of $\theta$. 

After that, the rest of the optimization procedure to find the best choice of $\theta$ will be handled by the optimization algorithm.

The $\nabla_\theta J(\theta)$ of the above cost function $J(\theta)$ as follows:
$$
\nabla_\theta J(\theta) = 

\begin{align}
\left[
\begin{array}{c} 
\frac{\partial J(\theta)}{\partial \theta_1}  \\
\frac{\partial J(\theta)}{\partial \theta_2}  \\
\vdots\\
\frac{\partial J(\theta)}{\partial \theta_n} 
\end{array}
\right]
\end{align}
$$
where $\textstyle \frac{\partial J(\theta)}{\partial \theta_j}$:
$$
\frac{\partial J(\theta)}{\partial \theta_j} = \sum_i x^{(i)}_j \left(h_\theta(x^{(i)}) - y^{(i)}\right)
$$

Also we can write $\nabla_{\theta} J(\theta)$ as vector form:
$$
\nabla_\theta J(\theta) = \sum_i x^{(i)} (h_\theta(x^{(i)}) - y^{(i)})
$$
