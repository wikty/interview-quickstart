资源

[Linear Regression](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/)

[Logistic Regression](http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/)

[Softmax Regression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)

[Linear model by Andrew Ng](http://cs229.stanford.edu/notes/cs229-notes1.pdf)

[Linear Classify by cs231n](https://cs231n.github.io/linear-classify/)

[liblinear -- A Library for Large Linear Classification](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)

要点

- LR 和 SVM 大一统，令 $y_i \in \{-1, +1\}$
  - 不含约束的最优化问题：
  - LR cross-entropy loss: $\log(1+\exp(-y_iw^Tx_i))$
  - SVM hinge/max-margin loss: $\max(0, 1-y_iw^Tx_i)$ or $\max(0, 1-y_iw^Tx_i)^2$
- 不同线性模型只是 loss 不同吗
- 感知机跟 LR 的关系？
- LR 和 SVM 从二分类进阶多分类的两种方式
  - 分类模型：LR -> Softmax, Binary SVM -> Multiclass SVM
  - 分类策略：OVR
- 基于 LR 和 Softmax 的交叉熵损失函数求导
- 线性模型的损失函数导数形式一致
- 线性模型最优化解和SGD解

