Download Link: https://assignmentchef.com/product/solved-homework-5-cs-260-machine-learning-algorithms
<br>
<h1>1           Bias-Variance Tradeo↵</h1>

Consider a dataset with <em>n </em>data points (<strong>x</strong><em><sub>i</sub>,y<sub>i</sub></em>), <strong>x</strong><em><sub>i </sub></em>2R<em><sup>p</sup></em><sup>⇥1</sup>, drawn from the following linear model:

<em>y </em>= <strong>x</strong><sup>&gt; <em>? </em></sup>+ <em>“,</em>

where <em>” </em>is a Gaussian noise and the star sign is used to di↵erentiate the true parameter from the estimators that will be introduced later. Consider the <em>L</em><sub>2 </sub>regularized linear regression as follows:

b = argmin<em>,</em>

whereeach row. <a href="https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Affine_transformation">Properties of an</a>0 is the regularization parameter. Let<a href="https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Affine_transformation">a</a> <a href="https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Affine_transformation">ne transformation</a> <em>X</em>of a gaussian random variable will be useful throughout2R<em><sup>n</sup></em><sup>⇥<em>p </em></sup>denote the matrix obtained by stackingin this problem.

<ol>

 <li>Find the closed form solution for and its distribution.</li>

 <li>Calculate the bias term E[<strong>x</strong><sup>&gt; </sup>b⇣] b<strong>x</strong>b<sup>&gt; <em>? </em></sup>as a function ofb ⌘ and some <em>fixed </em>test point <strong>x</strong>.</li>

</ol>

2

<ol>

 <li>Calculate the variance term E <strong>x</strong><sup>&gt; </sup>E[<strong>x</strong><sup>&gt;             </sup>]              as a function of   and some <em>fixed </em>test point <strong>x</strong>.</li>

 <li>Use the results from parts (b) and (c) and the bias–variance theorem to analyze the impact of in the squared error. Specifically, which term dominates when is small or large?</li>

</ol>

<h1>2           Kernelized Perceptron</h1>

Given a set of training samples (learns a weight vector <em>w </em>by iterating through all training samples. For each<em>x</em>1<em>,y</em>1)<em>,</em>(<em>x</em>2<em>,y</em>2)<em>,</em>··· <em>,</em>(<em>x</em><em>N,y</em><em>N</em>) where <em>y </em>2 { <em>x</em>1<em><sub>i</sub>,</em>, if the prediction is incorrect,1}, the Perceptron algorithm

we update <em>w </em>by <em>w</em> <em>w </em>+ <em>y</em><em>i</em><em>x</em><em>i</em>. Now we would like to<em>‘</em>, and we want to learn a new weight vector<em>kernelize </em>the Perceptron algorithm. Assume we map<em>w </em>that makes <em>x </em>to <em>‘</em>(<em>x</em>) through a nonlinear feature mapping prediction by <em>y </em>= sign(<em>w</em><sup>&gt;</sup><em>‘</em>(<em>x</em>)). Further assume that we initial the algorithm with <em>w </em>= 0.

<ol>

 <li>Show that <em>w </em>is always a linear combination of feature vectors, i.e. <em>w</em></li>

 <li>Show that while the update rule for <em>w </em>for a kernelized Perceptron does depend on the explicit feature mapping <em>‘</em>(<em>x</em>), the prediction can be re-expressed and thus depends only on the inner products between nonlinear transformed features.</li>

 <li>Show that we do not need to explicitly store <em>w </em>at training or test time. Instead, we can implicitly use it by maintaining all the <em>↵<sub>i</sub></em>. Please give the outline of the algorithm that would allow us to not store <em>w</em>. You should indicate how <em>↵<sub>i </sub></em>is initialized, when to update <em>↵<sub>i</sub></em>, and how it is updated.</li>

</ol>

<h1>3           Kernels</h1>

Mercer’s theorem implies that a bivariate function <em>k</em>(·<em>,</em>·) is a positive definite kernel function i↵, for any <em>N </em>and any <strong>x</strong><sub>1</sub><em>,</em><strong>x</strong><sub>2</sub><em>,</em>··· <em>,</em><strong>x</strong><em><sub>N</sub></em>, the corresponding kernel matrix <em>K </em>is positive semidefinite, where <em>K<sub>ij </sub></em>= <em>k</em>(<strong>x</strong><em><sub>i</sub>,</em><strong>x</strong><em><sub>j</sub></em>). Recall that a matrix <em>A </em>2R<em><sup>n</sup></em><sup>⇥<em>n </em></sup>is positive semidefinite if all of its eigenvalues are non-negative, or equivalently, if <strong>x</strong><sup>&gt;</sup><em>A</em><strong>x </strong>0 for arbitrary vector <strong>x </strong>2R<em><sup>n</sup></em><sup>⇥<a href="#_ftn1" name="_ftnref1">[1]</a></sup>.

Suppose <em>k</em><sub>1</sub>(·<em>,</em>·) and <em>k</em><sub>2</sub>(·<em>,</em>·) are positive definite kernel functions with corresponding kernel matrices <em>K</em><sub>1 </sub>and <em>K</em><sub>2</sub>. Use Mercer’s theorem to show that the following kernel functions are positive definite.

<ol>

 <li><em>K</em><sub>3 </sub>= <em>a</em><sub>1</sub><em>K</em><sub>1 </sub>+ <em>a</em><sub>2</sub><em>K</em><sub>2</sub>, for <em>a</em><sub>1</sub><em>,a</em><sub>2 </sub></li>

 <li><em>K</em><sub>4 </sub>defined by <em>k</em><sub>4</sub>(<strong>x</strong><em>,</em><strong>x</strong><sup>0</sup>) = <em>f</em>(<strong>x</strong>)<em>f</em>(<strong>x</strong><sup>0</sup>) where <em>f</em>(·) is an arbitrary real valued function.</li>

 <li><em>K</em><sub>5 </sub>defined by <em>k</em><sub>5</sub>(<strong>x</strong><em>,</em><strong>x</strong><sup>0</sup>) = <em>k</em><sub>1</sub>(<strong>x</strong><em>,</em><strong>x</strong><sup>0</sup>)<em>k</em><sub>2</sub>(<strong>x</strong><em>,</em><strong>x</strong><sup>0</sup>).</li>

</ol>

<h1>4           Soft Margin Hyperplanes</h1>

The function of the slack variables used in the optimization problem for soft margin hyperplanes has the form:. Instead, we could use, with <em>p &gt; </em>1.

<ol>

 <li>Give the dual formulation of the problem in this general case.</li>

 <li>How does this more general formulation (<em>p &gt; </em>1) compare to the standard setting (<em>p </em>= 1) discussed in lecture? Is the general formulation more or less complex? Justify your answer.</li>

</ol>

<h1>5           Programming</h1>

In this problem, you will experiment with SVMs on a real-world dataset. You will implement a linear SVM (i.e., an SVM using the original features. You will also use a widely used SVM toolbox called LibSVM to experiment with kernel SVMs.

<strong>Dataset</strong>: We have provided the <em>Splice Dataset </em>from UCI’s machine learning data repository.<sup>1 </sup>The provided binary classification dataset has 60 input features, and the training and test sets contain 1,000 and 2,175 samples, respectively (the files are called splice train.mat and splice test.mat).

<h2>5.1         Data preprocessing</h2>

Preprocess the training and test data by

<ol>

 <li>computing the mean of each dimension and subtracting it from each dimension</li>

 <li>dividing each dimension by its standard deviation</li>

</ol>

Notice that the mean and standard deviation should be estimated from the <em>training data </em>and then applied to both datasets. Explain why this is the case. Also, report the mean and the standard deviation of the third and 10th features on the test data.

<h2>5.2         Implement linear SVM</h2>

Please fill in the Matlab functions trainsvm in trainsvm.m and testsvm.m in testsvm.m.

The input of trainsvm contain training feature vectors and labels, as well as the tradeo↵ parameter <em>C</em>. The output of trainsvm contain the SVM parameters (weight vector and bias). In your implementation, you need to solve SVM in its primal form

<em>,b,</em>

s.t.                                            <em>⇠<sub>i</sub>,</em>8<em>i</em>

<em>⇠<sub>i          </sub></em>0<em>,</em>8<em>i</em>

Please use the quadprog function in Matlab to solve the above quadratic problem.

For testsvm, the input contains testing feature vectors and labels, as well as SVM parameters. The output contains the test accuracy.

<h2>5.3         Cross validation for linear SVM</h2>

Use 5-fold cross validation to select the optimal <em>C </em>for your implementation of linear SVM.

<ol>

 <li>Report the cross-valiation accuracy (averaged accuracy over each validation set) and average training time (averaged over each training subset) on di↵erent <em>C </em>taken from {4 <sup>6</sup><em>,</em>4 <sup>5</sup><em>,</em>·· <em>,</em>4<em>,</em>4<sup>2</sup>}. How does the value of <em>C </em>a↵ect the cross validation accuracy and average training time? Explain your observation.</li>

 <li>Which <em>C </em>do you choose based on the cross validation results?</li>

 <li>For the selected <em>C</em>, report the test accuracy.</li>

</ol>

<h2>5.4         Use linear SVM in LibSVM</h2>

LibSVM is widely used toolbox for SVMs, and it has a Matlab interface. Download LibSVM from <a href="http://www.csie.ntu.edu.tw/~cjlin/libsvm/">http: </a><a href="http://www.csie.ntu.edu.tw/~cjlin/libsvm/">//www.csie.ntu.edu.tw/</a><a href="http://www.csie.ntu.edu.tw/~cjlin/libsvm/">~</a><a href="http://www.csie.ntu.edu.tw/~cjlin/libsvm/">cjlin/libsvm/</a> and install it according to the README file (make sure to use the Matlab interface provided in the LibSVM toolbox). For each <em>C </em>from {4 <sup>6</sup><em>,</em>4 <sup>5</sup><em>,</em>··· <em>,</em>4<em>,</em>4<sup>2</sup>}, apply 5-fold cross validation (use -v option in LibSVM) and report the cross validation accuracy and average training time.

<ol start="5">

 <li>Is the cross validation accuracy the same as that in 3? Note that LibSVM solves linear SVM in dual form while your implementation does it in primal form.</li>

 <li>How does LibSVM compare with your implementation in terms of training time?</li>

</ol>

<h2>5.5         Use kernel SVM in LibSVM</h2>

LibSVM supports a number of kernel types. Here you need to experiment with the polynomial kernel and RBF (Radial Basis Function) kernel.

<ol>

 <li><strong>Polynomial kernel</strong>. Please tune <em>C </em>and degree in the kernel. For each combination of (<em>C</em>, degree), where <em>C </em>2 {4 <sup>3</sup><em>,</em>4 <sup>4</sup><em>,</em>·· <em>,</em>4<sup>6</sup><em>,</em>4<sup>7</sup>} and degree 2 {1<em>,</em>2<em>,</em>3}, report the 5-fold cross validation accuracy and average training time.</li>

 <li><strong>RBF kernel</strong>. Please tune <em>C </em>and gamma in the kernel. For each combination of (<em>C</em>, gamma), where <em>C </em>2 {4 <sup>3</sup><em>,</em>4 <sup>4</sup><em>,</em>·· <em>,</em>4<sup>6</sup><em>,</em>4<sup>7</sup>} and gamma 2 {4 <sup>7</sup><em>,</em>4 <sup>6</sup><em>,</em>··· <em>,</em>4 <sup>1</sup><em>,</em>4 <sup>2</sup>}, report the 5-fold cross validation accuracy and average training time.</li>

</ol>

Based on the cross validation results of Polynomial and RBF kernel, which kernel type and kernel parameters will you choose? Report the corresponding test accuracy for the configuration with the highest cross validation accuracy.


