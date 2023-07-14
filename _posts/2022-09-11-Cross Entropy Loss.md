# Cross-entropy loss

This lesson finally made sense to me. I've been slacking a lot on it, and I'm really happy it all finally clicked in the end. It took me forcing myself to revisit certain parts of the video, but I got it. This post is to help me consolidate what I learned and fill in any gaps I discover while writing.

## What is cross-entropy loss, at the highest level?

Cross-entropy loss is a way of calculating loss and parameters for problems with more than two categories (non-binary problems). With binary problems, you can find confidence of one category, and subtract that from 1 to get the confidence of the second category. However, you obviously can't do this with multi-category problems. You need a new way of 1) Calculating predictions and normalizing them, and 2) Calculating loss from those predictions to make your gradient steps.

## Deeper dive into cross-entropy loss

Cross-entropy loss encompasses two functions:

1. Softmax

2. Negative log likelihood

### Softmax

I think of the softmax function as another magic way of normalizing predictions for multi-category classification problems. What it does is basically sigmoids all of the values, then divides by the total of the sigmoids, to create a normalized set of predictions that add up to 1. This is useful because 1) lower numbers means less confident/higher means more confident, which is essential for our loss function, and 2) they all add up to 1, so we can easily compare relative confidences because they are all on the same scale.

To use this function, all you do is pass in your set of predictions, and specify the output dimension:
```
sm_acts = torch.softmax(acts, dim=1)
```
### Negative Log Likelihood (NLL)

Negative log likelihood is pretty clever. It's the loss part of cross-entropy loss. What it does is takes the softmax predictions we calculated before, and applies the negative log to the prediction for the **correct** category.

For example:

![image](https://user-images.githubusercontent.com/83550862/189551085-49ac812f-82f4-4e87-9a31-5c4293b521fb.png)

This final tensor contains the predictions for the correct classifications. We then apply a negative log to these numbers. This makes the following true: If the confidences are **low**, then the loss will be **high**, and vice-versa, which is exactly what we need from a loss function.

![image](https://user-images.githubusercontent.com/83550862/189551131-c805644e-40a1-4d48-9b34-bf6a87722b17.png)

# Questionnaire

**1. Look up the documentation for L and try using a few of the new methods that it adds.**

L is just a special type of list that fastai likes to return. I'll look at the new methods when I need them.

**2. Look up the documentation for the Python pathlib module and try using a few methods of the Path class.**

I have done this with my project.

**3. Give two examples of ways that image transformations can degrade the quality of the data.**

Twists and untwists can lose data. Similarly, lossy compressions can make the data less thorough and give the computer less to train on.

**4. What method does fastai provide to view the data in a DataLoaders?**

You can show_batch().

**5. What method does fastai provide to help you debug a DataBlock?**

The data_block.summary() function gives you an example batch from your data. If it fails, you can see why your batch didn't work and adjust your DataBlock accordingly.

**6. Should you hold off on training a model until you have thoroughly cleaned your data?**

Probably not. Training your model can highlight imperfections in your data with that function in fastai that shows the highest loss data items. I think I intuitively understand that jumping in for things like this with short iteration times is usually better.

**7. What are the two pieces that are combined into cross-entropy loss in PyTorch?**

1. Softmax
2. NLL (negative log likelihood)
I outlined these above!

**8. What are two good rules of thumb for picking a learning rate from the learning rate finder?**

1. Pick the steepest downward slope on the graph.
2. Pick the minimum, then divide that X value by 10 to get a good learning rate.

**9. What two steps does the fine_tune method do?**

1. Freezes the first layers, makes random outputs for the last layer, trains for an epoch (or any specified number of epochs).
2. Unfreezes and trains the whole algorithm on the specified number of epochs.

**10. In Jupyter Notebook, how do you get the source code for a method or function?**

??

**11. What are discriminative learning rates?**

Since each layer of the model gets more specific the further we go, we want smaller learning rates for earlier layers (they don't need to be changed much) and larger ones for the more specific layers near the end (they need to be retrained).

**12. How is a Python slice object interpreted when passed as a learning rate to fastai?**

It's given a min and max value, and basically is a start/end object. When you give it to the learning rate, it divides the difference between the two slice values, and makes each learning rate, starting at the beginning, a linear addition of the start index of the slice. That way, the first layer has a learning rate of the start slice index, the last layer has the last slice index, and each layer in between is equidistant from its neighbors.

**13. Why is early stopping a poor choice when using 1cycle training?**

one_cycle_train increases and then decreases the learning rate, so if you stop early, you're going to get a bad learning rate.

**14. What is the difference between resnet50 and resnet101?**

One has 50 layers, the other has 101.

**15. What does to_fp16 do?**

Half precision floating points (less accurate but much faster).
