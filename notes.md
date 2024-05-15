# Notes
+ content and some notes based on the free course on https://www.coursera.org/learn/machine-learning  Instructor: Andrew Ng
+ Linear Algebra
  + 
## Machine Learning Concepts
+ Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."
+ Supervised Learning and Unsupervised Learning
  + Supervised Learning: given a data set and all right answers of it
  + Regression: predict results within a continuous output(distribution);map input variables to some continuous function
  + Classification:map input variables into discrete categories
  + Unsupervised Learning: given a data set without any labels
    + e.g. cocktail party problem algorithm using svd
  + Semi-supervised Learning: also make use of unlabeled data for training
  + low-shot/one-shot learning: learn object categories from few examples. Attention: it is defined in different ways from semi-supervised learning.
+ Model Representation
  + $x_i$ : input variables or input features
  + $y_i$ : output or target variable
  + $(x_i,y_i)$ training example.
  + Data sets
    + Training set: list/set of training examples.
    + Training set -> Learning Algorithm -> hypothesis(standard terminology): function map from features to lables.
+ Cost functions/ojbective function
  + for $h =t0 + t1x$
  + $$minimize_{\theta_0,\theta_1} sum_i^m (h(x_i)-y_i)^2$$
  + squared error function $$J(\theta_0,\theta_1)=\frac{1}{m}\sum_i^m(h(x_i)-y_i)^2$$
+ Gradient Descent
  + minimize arbitary cost function J
  + start from initial point.
  + decrease along the gradient of loss function with fix step(learning rate)
  + learning rate
    + too small: slow
    + too large: can overshot the minimum so may fail to converge or even diverge
  + SGD
  + momentum
  + Adam

## Algorithms
+ Random Forest
  + Decision Tree
    + binary tree elements of random forests
    + classification and Regression Trees(CART)
      + root node represents a single input
      + split point on varaible
    + greedy splitting
      + very best split point chosen eatch time
    + criterion
      + measure the quality of a split
      + Gini impurity
        + randomly pick a data point
        + randomly classify it according to the class distribution in the data
        + probability that we classify the datapoint wrong
        + for C classes
        $$ G=\sum_{i=1}^C p(i)*(1-p(i))$$
        where p(i) is probability the data point belongs to class i
        + 0 is the lowest and best possible
        + maximize Gini Gain
          +substract the weightedd impurities of the branches from the oiginal impurity
      + Entropy and information gain
        + Entropy - how much variance the ddata has
          $$E = - \sum_i^C p_i log_2 p_i$$
          $p_i$ is probability randomly picking an element of class i
        + weight the entropy of each bracn by how many elelments it has
        + information Gain = how much engropy we removed after weight
    + Grow tree as follows
      + training set is N, sample N cases at random, wt replacement
      + M input variables, number m << M($m=\sqrt{m}$ or $m=1/3M$) sepecified such that each node, m variables are selected at random out of the M, best split on thes m used to split the node, m is held constatn during the forest growing
      + each tree is grown to the largest extent possible
    
    
    + Purpose
      + motivation
    + Algorithm overview
    + comparison algorithms
+ Mean Shift
