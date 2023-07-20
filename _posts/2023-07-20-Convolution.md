# Convolution

Finally got around to looking at convolution. I'd used it a lot with CNNs but hadn't ever really stopped to think about what it's actually doing. This post is going to go over a few of the insights that I had with convolution (and my connection with fusion, to tie it all together).

I mostly used 3b1b (https://www.youtube.com/watch?v=KuXjwB4LzSA) and (https://www.youtube.com/watch?v=IaSGqQa5O-M) to gain the intuition for this. His videos never cease to amaze me, and his visual intuitions are great.

### What are convolutions

Convultions are, simply put, an operation which applies one function to another function that returns a third, new function. Just like multiplying or adding two functions, _convolving_ two functions gives you a new function. Although they seem really foreign (and they are, since they aren't taught very much at the beginnings of the math journey), they're pretty simple at their core. Obviously, convolutions can become really complex, and there's tons of cool applications of them. But the simplest convolution can be explained really well with two die.
