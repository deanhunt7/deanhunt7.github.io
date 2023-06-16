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
where $E_{est}$ represents the error in the estimate (we'll get to that soon) and $E_{measure}$ represents an error in your measurement. This equation is pretty intuitive when viewed as a metric of trust; as $E_{measure} { \rightarrow }\infty$, the fraction approaches zero (so your trust is extremely low). Similarly, as $E_{measure} { \rightarrow }0$, the gain approaches 1 (so your trust is absolute). This is multiplied against the difference between measurement and prediction values, shown in 1) above.

Now that we have the Kalman gain calculated, we can use this equation:
$$est_{t} = est_{t-1} + KG(mea - est_{t-1})$$
to calculate our newest estimate of position. (Note: $est_{t}$ is actually the estimate for time $t$, but is *used* at time $t+1$, once we actually have information (our measurement) on how good our estimate was. That was confusing for me for a while.) This equation is very intuitive: to make your new estimate, you take your old estimate and add the weighted difference between your measurement and previous estimate. In the falling object example, we'll take a measurement after the object has been falling for some time. We have a previous estimate for where we think the object *should* be, and we have a measurement for where the sensors tell us the object *actually* is. We'll use the Kalman gain in the above equation to split this difference, and settle on some midpoint between the two values.

### Estimate error

After this, we also need to update the estimate in our error. This is necessary for calculating the Kalman gain for the next iteration of the filter. Two equivalent equations for estimate error are given below.

$$\frac{(E_{mea})\cdot(E_{est_{t-1}})}{E_{mea} + E_{est_{t-1}}} \ \ \ \    or \ \ \ \   (1-KG)E_{est_{t-1}}$$
The intuition is clear in the second equation. As the Kalman gain increases (the measurements are more trustworthy), our estimation error decreases (since we are more certain we are estimating the actual position of the object).

As shown in the diagram, this error feeds into the next iteration of the cycle, and is used to calculate the next Kalman gain.

This is the intuition that we can build upon for the matrix versions of this process in the next section. Make sure you understand, from a top-down perspective, the motivations behind the filter and the general purpose of each step in this simplified example.
