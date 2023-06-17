One critical piece of the [state estimation process of sensor fusion](https://deanhunt7.github.io/2023/06/13/Sensor-Fusion.html) is the [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter). I'm writing this blog post mainly to brush up on my linear algebra chops, and to make a cool application of the filter to something I find interesting.

# What the Kalman filter is

It took me a little while to get a fundamental understanding of what the Kalman filter is doing under the hood. I finally came to this conclusion: **the Kalman filter is all about abstracting a pattern from noise**, just like in Ender's game when he worked with Mazer Rackham and pinpointed the epicenter of the alien's movements. Applications of the Kalman filter range from signal noise reduction to F-35 sensor fusion. In all of these use cases, the critical function the filter plays is being able to filter out the pattern from the noise. The filter allows our predictions to converge on the true value of an unobservable process through the emitting of measurements (exactly like an HMM). I'll expand on this intuition further on in the post. Kalman filtering is a Bayesian process that lends itself quite nicely to dynamic programming. In fact, the filter treats any process it's modeling like a Markov process, keeping the context requirements very small.

I'm going to start with a 1-D introduction to the filter. This kind of introduction [here](http://www.ilectureonline.com/lectures/subject/SPECIAL%20TOPICS/26/190) (there's a better formatted playlist on YouTube) really helped me understand the basics of the filter. As a side note, I've found that being able to fully understand a topic in lower dimensions lends itself to abstraction to higher dimensions, as often the same basic principles apply. After that, I'll dive into the multidimensional matrix version of the filter (which, at its core, is the same as the 1-D approximation).

# One-Dimensional

Below is an extremely helpful diagram detailing the processes involved in one iteration of the Kalman filter. I'll reference it throughout this section, so I'm just introducing it before anything else.

![kalman_diagram1](https://github.com/deanhunt7/deanhunt7.github.io/assets/83550862/05cfb2ac-c4c9-4cc3-aad9-81e8751636a3)

While running through the steps of the filter, I found it helpful to attack it with a top-down approach. First, we know what we want to do: we want to **reduce noise in our measurements in order to converge on some true value**. We know we have two sources of potential values: 1) our measurements that we periodically make, and 2) our predictions we make given those measurements. We need some way to figure out how much to trust each of these sources, so that we can use all possible information in the optimal way to come to a prediction of the next state. Finally, after we take the next measurement, we need to compare it to our last prediction and update our estimate error (which is necessary for our conceptualization of "trust").

Here's the overall Kalman estimate equation:

$$est_{t} = est_{t-1} + KG(mea - est_{t-1})$$

To put this into some more context, I'll give a concrete example. Let's pretend like we're modeling the falling of some object, straight down, and only affected by the force of gravity. We wish to know its position at any given time. Unfortunately, we can only periodically sample the position, and thus can't know the true path (pretend as if we are ignorant of basic kinematics for the time being). We begin with an initial measurement where we measure the initial speed and velocity of our object. That's great! Unfortunately, our sensors aren't perfect; we have some errors to contend with. That's fine, because we can factor that in later. Referring to the diagram above, our next step will be to calculate the Kalman gain, where we'll need a previous estimate. Since this is our first measurement, we can really give any estimate; it'll eventually converge anyway.


### Kalman gain

The easiest way to think about the Kalman gain is basically as a measure of trust of your sensors. The Kalman gain is calculated based on the following formula:

$$\frac{E_{est}}{E_{est} + E_{measure}}$$

where $$E_{est}$$ represents the error in the estimate (we'll get to that soon) and $$E_{measure}$$ represents an error in your measurement. This equation is pretty intuitive when viewed as a metric of trust; as $$E_{measure} { \rightarrow }\infty$$, the fraction approaches zero (so your trust is extremely low). Similarly, as $$E_{measure} { \rightarrow }0$$, the gain approaches 1 (so your trust is absolute). This is multiplied against the difference between measurement and prediction values, shown in 1) above.

Now that we have the Kalman gain calculated, we can use this equation:

$$est_{t} = est_{t-1} + KG(mea - est_{t-1})$$

to calculate our newest estimate of position. (Note: $$est_{t}$$ is actually the estimate for time $$t$$, but is *used* at time $$t+1$$, once we actually have information (our measurement) on how good our estimate was. That was confusing for me for a while.) This equation is very intuitive: to make your new estimate, you take your old estimate and add the weighted difference between your measurement and previous estimate. In the falling object example, we'll take a measurement after the object has been falling for some time. We have a previous estimate for where we think the object *should* be, and we have a measurement for where the sensors tell us the object *actually* is. We'll use the Kalman gain in the above equation to split this difference, and settle on some midpoint between the two values.

### Estimate error

After this, we also need to update the estimate in our error. This is necessary for calculating the Kalman gain for the next iteration of the filter. Two equivalent equations for estimate error are given below.

$$\frac{(E_{mea})\cdot(E_{est_{t-1}})}{E_{mea} + E_{est_{t-1}}} \ \ \ \    or \ \ \ \   (1-KG)E_{est_{t-1}}$$

The intuition is clear in the second equation. As the Kalman gain increases (the measurements are more trustworthy), our estimation error decreases (since we are more certain we are estimating the actual position of the object).

As shown in the diagram, this error feeds into the next iteration of the cycle, and is used to calculate the next Kalman gain.

This is the intuition that we can build upon for the matrix versions of this process in the next section. Make sure you understand, from a top-down perspective, the motivations behind the filter and the general purpose of each step in this simplified example.



# Matrices

The matrix version of the Kalman filter is, naturally, the multi-dimensional version. Since most applications of the Kalman filter are at least two-dimensional, the matrix version is the most widely applied. To understand the need for a multi-dimensional model, consider the movement of a mouse across the ground. We can't simply find the variance in measurements in one direction, since the mouse's position varies in both the X and Y directions. Instead, we'll need to use matrices to store all of our state information and variances.

Similar to the 1-D example, here's a map of the multi-dimensional Kalman filter process.

![thumbnail_11413ACC-A3FB-4181-9231-608DBF3801DA](https://github.com/deanhunt7/deanhunt7.github.io/assets/83550862/b4b3bd0f-e37e-47cf-83ba-6efb2a67494c)

Just like before, we start with an initial state estimate $$X_{0}$$ , and something new: $$P_{0}$$, which we call our "process covariance matrix". I'll go more into it later, but for right now picture it as the initial process error estimate. These are fed into the "previous state" part of the loop, and continue on to the new state prediction.

### Predict new state

In this part of the process, we predict the new state, given a few pieces of information. (1) represents the predicted state update equation, with explanations of variables listed below.
$$X_{kp} = AX_{k-1} + B\mu_{k} + \omega_{k} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (1)$$

$$p$$: denotes a prediction

$$k$$: denotes a discrete time step k 

$$X_{k-1}$$: previous prediction given from the Kalman filter

$$A$$: linear transformation matrix. Computes linear extrapolation from state $$X_{k-1}$$.

$$\mu_{k}$$: control variable matrix. A "control variable" is any variable that represents an outside force acting on the system (essentially how Kalman takes care of non-linearity)

$$B$$: control transformation matrix. Computes linear transformation for control variable matrix

$$\omega_{k}$$: predicted noise matrix. Assumed $$w_k \sim \mathcal{N}(0, Q_k)$$ , with covariance $$Q_k$$ explained below. Takes care of any outside measurement or environmental noise. I'm probably not going to use this much in this explanation, but it's helpful to keep in mind.

Equation (1) is pretty intuitive. You're essentially adding up all of the state updates from interior and exterior processes, plus some noise. 

Next, equation (2) represents the update to the predicted error matrix, with variable explanations.
$$P_{kp} = AP_{k-1}A^{T} + Q_{k} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (2)$$

$$P_{k}$$: predicted process covariance matrix at time $$k$$ (basically the error in the estimate)

$$Q_k$$: process noise covariance matrix. This is the covariance value from the Gaussian $$w_k$$. This also keeps $$P$$ from going to 0, which would lead to inaccurate estimation of error.

This equation is also fairly straightforward. $$A$$ and $$A^T$$ **transform the error matrix according to the dynamics described by $$A$$.** This took me a little bit of time to understand, but I think I get it now. The covariance matrix needs to be propagated according to a set of rules, and these are defined by $$A$$ (the system dynamics). The reason we don't use $$B$$ is because this denotes control variable dynamics, which are forces not contained within the system. Those wouldn't have any effect on the actual process error, so we only use $$A$$ to transform $$P_{k-1}$$. The final $$A^T$$ transformation is to map the product $$AP$$ back into the correct dimensions.

Now that we have our predictions, we'll need to take a measurement in order to compare the two. The measurement equation 
$$Y_k = HX_{km} + z_k \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (3)$$ shows the process of collecting such a measurement.
$$H$$: observation model. Basically a mask that maps the entire state space onto the observed state space (the space you care about observing, like position and velocity).
$$z_k$$: observation noise. $$z_k \sim \mathcal{N}(0, R_k)$$, with $$R_k$$ being used in the Kalman gain equation (4).

Now, let's revisit our 1-D intuition. After we have 1) a prediction, and 2) a measurement, we then used these two to calculate the Kalman gain, a measure of sensor trust. The same is true in this case; here's the Kalman gain equation. It should look very similar to the 1-D equation.
$$K = \frac{P_{kp}H^T}{HP_{kp}H^T+R_k}$$\n
Compare this to $$\frac{E_{est}}{E_{est} + E_{measure}}$$ and the parallel is obvious.

Using the Kalman gain, we update the official state $$X_k$$:
$$X_k = X_{kp} + K(Y - HX_{kp})$$
which, again, is the exact same as the 1-D version $$est_{t} = est_{t-1} + K(mea - est_{t-1})$$.\n
Now there's a state to report! The only thing left is to report the updated error $$P$$, given by $$P_k = (I-KH)P_{kp}$$. As always, this is a parallel to the 1-D version $$(1-K)E_{est_{t-1}}$$, where the error becomes smaller whenever the measurements are trusted more (higher gain).

And that just about does it! The outputs $$X_k$$ and $$P_k$$ feed back into the system as the previous observation, and the filter recurses on these inputs, slowly converging on the true value of the modeled process.

There's the 1-D and multi-D versions of the Kalman filter. Initially I was going to include my ipynb in here, but this post is so long that I'll probably just break it out into its own post. Thanks for reading!


## What is the variance-covariance matrix?

When beginning to learn about the Kalman filter, I had a lot of trouble understanding what covariance had to do with error. In the multi-dimensional version, the covariance matrices are representations of error.
$$
\begin{bmatrix}
\sigma_{1}^2 & \sigma_{12} \\
\sigma_{21} & \sigma_{2}^2 \\
\end{bmatrix}
$$
In this matrix, $$\sigma_{1}^2$$ represents the variance of the first variable, $$\sigma_{2}^2$$ represents the variance of the second variable, and $$\sigma_{12}$$ and $$\sigma_{21}$$ represent the covariance between the two variables. Obviously, var-covar matricies are symmetrical, and it turns out they are positive definite (as they should be, since the eigenvalues represent uncertainty in each eigenvector direction, and these uncertainties can't be negative). My understanding of this matrix is that the larger the values in the matrix, the larger the uncertainties of measurement are. That's why you can use this (denoted as $$P$$ in the post) to represent error. What I don't understand is how they make an initial prediction of $$P$$. I'm guessing they take into account known errors in the sensors and possibly a noise coefficient, but I'm not entirely sure. I'll probably ask one of the algo engineers at work about it and update this post based on what they tell me.
