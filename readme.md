<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Repo Introduction

+ codes written mainly in python to implement some popular machine learning algorithms
+ content and some notes based on the free course on https://www.coursera.org/learn/machine-learning  Instructor: Andrew Ng

## Notes

+ Machine Learning
  + Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

+ Classfied into Supervised Learning and Unsupervised Learning
  + Supervised Learning: given a data set and all right answers of it
  + Regression: predict results within a continuous output(distribution);map input variables to some continuous function
  + Classification:map input variables into discrete categories
  + Unsupervised Learning: given a data set without any labels
    + e.g. cocktail party problem algorithm using svd
  + Semi-supervised Learning: also make use of unlabeled data for training
  + low-shot/one-shot learning: learn object categories from few examples. Attention: it is defined in different ways from semi-supervised learning.
+ Model Representation
  + $$ x_i $$ : input variables or input features
  + $$ y_i $$ : output or target variable
  + (x_i,y_i) training example.
  + Data sets
    + Training set: list/set of training examples.
    + Training set -> Learning Algorithm -> hypothesis(standard terminology): function map from features to lables.
+ Cost functions
  + for $$ h =\theta_0 + \theta_1x$$
  + $$minimize_{\theta_0,\theta_1} sum_i^m (h(x_i)-y_i)^2$$
  + squared error function $$J(\theta_0,\theta_1)=\frac{1}{m}\sum_i^m(h(x_i)-y_i)^2$$
+ Gradient Descent
  + minimize arbitary cost function J
  + start from initial point.
  + decrease along the gradient of loss function with fix step(learning rate)
