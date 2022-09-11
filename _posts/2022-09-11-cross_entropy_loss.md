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
