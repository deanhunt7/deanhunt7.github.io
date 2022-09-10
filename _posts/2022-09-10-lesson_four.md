I'm not feeling great about this lesson. I feel too lazy to go back and do it over again, but I feel like I didn't get a solid intuition for things like I did in the previous lessons. I do understand ReLU, softmax, nll, but I don't feel firm about them. Also, I haven't built the notebook from scratch, which is also probably a problem. I just don't feel good about this lesson. Hopefully the next one will be better and might even convince me to come back to this one.

What I'm going to do in the next video:
1. Actually build the notebook from scratch when I'm done. I know this one will probably take a long time, but I don't feel firm with FastAI in terms of actually coding it, and I'll need to be much better with it for my current project.
2. Write more notes. I've been too passive in these lectures. It's too easy to just sit back and "understand" what's going on, and then end up with it slipping away. Being active is going to help me build a good fundamental intuition for these kinds of things.


What is the difference between a loss function and a metric?
A loss function calculates <b>loss</b> which is a metric used to gauge performance and make adjustments using SGD.

What is the function to calculate new weights using a learning rate?
calc_grad

What does the DataLoader class do?
This provides batches of data for the GPU to use.

Write pseudocode showing the basic steps taken in each epoch for SGD.
1. Initialize parameters (init_params())
2. Predict preds = predict(data)
3. Calculate loss (loss = rms(preds, labels))
4. Step parameters based on loss (use small step rate)
5. Repeat


Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. What is special about that output data structure?
list(x, y)
  
What does view do in PyTorch?
tensor.view converts a tensor into a different size while keeping the same data and number of elements. You can swap rows, string it out into one long line, etc.

What are the "bias" parameters in a neural network? Why do we need them?
Bias is a constant added to the weight * activation. It basically gives you another axis to move your function around on (the weight lets you rotate the function, the bias lets you move it up and down)

What does the @ operator do in Python?
Matrix multiplication!

What does the backward method do?
backward() calculates the derivative of a function.

Why do we have to zero the gradients?
Gradients need to be zeroed because backward() adds the current value to the new derivative it calculates.

What information do we have to pass to Learner?
We need to pass:
1. A DataLoader
2. A neural net architecture
3. Optimization function
4. Loss function
5. What we want to print out

What is "ReLU"? Draw a plot of it for values from -2 to +2.
A "REctified Linear Unit" is really cool. Basically, it adds nonlinearity to your neural net. Without this, all you're doing is linear function composition, which could then just be modeled by one layer. It is y=x until you get to negative X values, where it becomes zero. What ReLU basically allows you to do is create "bends" in your linear function wherever you want. Make enough of these bends, and you can approximate any curve. This is what makes UAT possible.

What is an "activation function"?
What's the difference between F.relu and nn.ReLU?
nn is a Neural Net class. F is the functional import of nn, which just contains all of the functions associated with neural networks.

The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
We don't need as many layers if we use multiple nonlinearities.

What are the two pieces that are combined into cross-entropy loss in PyTorch?
1. Softmax - basically making probabilities for each of a given number of features and having them all add up to 1, amplifies small differences so that something with a slightly higher activation will be more confident
2. Negative Log Likelihood Loss - take the negative log of your loss giving you this shape:
![image](https://user-images.githubusercontent.com/83550862/189504240-21aac2c3-c781-4a48-969f-7335b6cda297.png)

This image shows that the loss when you're close to 0 (a bad prediction) will be high, and loss when close to 1 (good prediction) will be low.

What are the two properties of activations that softmax ensures? Why is this important?
1. They all add up to 1. This is important because it keeps us normalized and we can compare relative confidences much more easily.
2. Less confidence = lower score. This is crucial because without this our loss function wouldn't work!
