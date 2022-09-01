This lesson was...fine. I really didn't learn a whole lot of new things. A large, large part of the lesson (about the first 1.5 hours) was spent going over deploying a model and all the UI packages you can use to spice up your MVP. I think I'm relatively good on this front; I'm more worried about learning the math behind machine learning at the moment. I can worry about deployment once I have something to deploy, after all!

One really cool thing I got from this lesson was the use of tensors. They seem to be a super powerful data structure in python. One interesting thing that Jeremy Howard did with a tensor is use it to quickly average data. What he had was a set of 2000 or so images of the number 3. What he wanted to do was grab the average darkness of each pixel (28 x 28) in the image. Rather than use a loop (like I would have done), what he did was *stack the images to make a rectangular prism*. I thought this was so cool! After this, using the magic of tensors he could basically shoot a ray through each of the pixels and easily calculate its average.

Another very useful thing I got from this lesson was obviously SGD. This is the fundamental method of training a machine learning model, so it was nice to finally see how it worked. It was surprisingly un-complex, which makes me think we aren't getting the full story. I'm still fuzzy about the whole "backwards" thing and what that has to do with taking the derivative.

Regardless, this was an interesting, if slightly drawn-out lesson. I feel good about what I really learned from it, and I'm excited for the next lesson.

What is a confusion matrix?
A confusion matrix shows the incorrect predictions and their actual labels.

What does export save?
It "pickles" the model into basically an exe, a portable, trained model you can give data to.

What is a "rank-3 tensor"?
This is a three-dimensional tensor.

What is the difference between tensor rank and shape? How do you get the rank from the shape?
The rank of a tensor is the number of dimensions. The shape is a tuple that shows the size of each dimension. The rank is the length of the shape tuple.

What are RMSE and L1 norm?
RMSE is Root Mean Square Error, where you square the difference, take the mean, and then square root it. L1 norm is also known as "taxicab norm", where you add up the values of each component. In this case, you would subtract the values of each component and use that as your difference function.

How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
You can input a tensor into the function, rather than looping through each number in the tensor.

What is broadcasting?
Broadcasting is really cool. Basically, when you put a tensor into a function that uses another sized tensor, it will "spread out" the smaller tensor to match the dimensions of the larger one.

Are metrics generally calculated using the training set, or the validation set? Why?
Obviously validation set. I don't know why I struggled with this question. This helps to avoid overfitting to the training data.

What is SGD?
Stochastic Gradient Descent is a method of parameter-tuning that uses the derivative of the curve modeled by the model to calculate the best possible change in metrics to make to maximize accuracy.

Why does SGD use mini-batches?
It uses mini-batches so it doesn't over-tune the parameters. In small jumps it can much more accurately tune the parameters to model the real function most closely.

What is "loss"?
Loss is a more granular way of calculating accuracy. With accuracy, it's more of a step function (right/wrong), but with loss you can see that even if you were wrong, you were at least *closer* to the right answer.

Why can't we always use a high learning rate?
A high learning rate can actually make you more inaccurate because you may be overtuning the parameters far too much.

What is a "gradient"?
Derivative of a function (function that gives the slope at a certain point).
