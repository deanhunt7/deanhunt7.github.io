# MoM

I was really confused about method of moments for a while. Textbooks and videos made it look super simple; it had something to do with the first two moments, which just happened to be mean and variance, and systems of equations, which could easily be solved to find the estimator.

What I struggled most with was the intuition behind this method. Why does it work? What do those equations represent?

The missing key for me was understanding that *moments, like the mean and variance, tend to rely on the parameters we are trying to estimate.* Sample moments are made from *sample data*, where we can calculate sample means and variances easily. Then, we solve the equation *relating the parameters to the moments* to find the estimators.

For example, for a poisson distribution, we know that E(x) and Var(x) = $$\lambda$$. If we are trying to estimate $$\lambda$$, all we need to do is get a sample and find the expected value. This expected value will be our best estimate for $$\lambda$$.

It becomes more tricky if we don't know the distribution modeling our data. If all we are given is a density function, we'll have to find our own relationship between the desired parameters and moments. This can be done with the typical formulas for E(x) and Var(x), given the density functions.

# MLE

If we can assume a set of data is represented by a distribution, then regardless of whether or not we know the parameters for the distribution, *there is a single value for each parameter that makes the model fit the data best.* We're just trying to get close enough to this parameter.

The way we do this is through maximization. But what to maximize? We don't know our parameter (for example, $$\theta$$), so how do we know what to maximize? We can leverage the power of conditionals to help us solve this.

Imagine the possible values for all $$\theta$$ plotted on the x-axis of a graph. On the y-axis, we plot the "level of fit" (sum of mean squared errors from actual values). 

![image](https://user-images.githubusercontent.com/83550862/201781751-2a3b9b95-0c1f-4afe-b491-29040e8e41d4.png)


We are looking for the maximum of this graph! 

However, instead of plotting this graph (whose derivative would be difficult to compute), we can instead use a clever trick. We can take random samples X1, ... , Xn. For each sample, we multiply the probability of attaining that value, given our parameter is theta, $$P(X_i = x_i | \theta)$$. Then, for each $$i$$, we multiply these probabilities together. Maximizing **this** function will give us the same result as the mean squared error function, since the smallest errors will correspond to a maximum value on the graph. So we maximize the function $$\displaystyle \prod_{i=1}^{n} P(X_i = x_i | \theta)$$
As we know, maximization is synonymous with taking the derivative. Another clever trick to expedite this is to take the log of both sides of the equation prior to differentiating (since differentiating sums is much easier than differentiating products). Finally, we set this derivative equal to 0, and solve for $$\hat{\theta}$$.

