<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: { skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'], inlineMath: [['$','$']] } }); </script> <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> 
# Convolution

Finally got around to looking at convolution. I'd used it a lot with CNNs but hadn't ever really stopped to think about what it's actually doing. This post is going to go over a few of the insights that I had with convolution (and my connection with fusion, to tie it all together).

I mostly used [this](https://www.youtube.com/watch?v=KuXjwB4LzSA) and [this](https://www.youtube.com/watch?v=IaSGqQa5O-M) to gain the intuition for this. His videos never cease to amaze me, and his visual intuitions are great.

### What are convolutions

Convultions are, simply put, an operation which applies one function to another function that returns a third, new function. Just like multiplying or adding two functions, _convolving_ two functions gives you a new function. Although they seem really foreign (and they are, since they aren't taught very much at the beginnings of the math journey), they're pretty simple at their core. Obviously, convolutions can become really complex, and there's tons of cool applications of them. But the simplest convolution can be explained really well with two die. I thought the die example 3b1b gave was really good. It helped me get a concrete (if contrived) understanding of why I care about convolutions.

He also looked at convolutions in image processing and classification, which was pretty interesting. He went over the gaussian blur (which totally makes sense why it's gaussian now), and about how to do edge detection with special kernels. The edge detection was done by making a kernel with negative values on one half and positive values on the other half. This way, you can tell when one half of the kernel has entered a new pixel range, so it's crossed over an edge. You can grab horizontal, vertical, and I'm sure diagonal edges too (just make the kernel diagonal).

Side note to myself: Gaussian blurs totally make sense now. Essentially, the kernel mimics a 2D gaussian, so it weights things close to the center pixel higher than it does stuff further out. This helps get a much better blur than if you just take the average of surrounding pixels.

## the math

I was a little confused on the integral definition of a continuous convolution. It helped me to see the discrete case first, given by:

$$(f * g)[t]=\sum_{k=0}^{t} f[k] g[t-k] \nonumber$$

This one confused me first as well, until I got the nice sliding window window example and animation, after which everything made sense.

It all comes down to flipping and sliding the window (in the discrete case). The reason for the flip is so that every pair lines up to add to the same number. (Note: I just realized that in the discrete case, our array has to be all the natural numbers up to whatever $t$ we choose. But that makes sense, since everything else will just be 0.) Then, the integral definition
$$(f * g)[t]=\int_{-\infty}^{\infty} f[k] g[t-k] \nonumber$$
is just the continuous extension from the discrete case above.

# fft

I ran into some difficulties understanding the FFT algorithm. I haven't spent enough time looking at it to figure out why I care yet. There are some good Veritasium and a 3b1b video solely dedicated to this, so I'll take a look at those this week.

# fusion

I'll need to think more about how to satisfactorially explain the connection with fusion. 
