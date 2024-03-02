# Fourier Transformations

I've heard about these for so long, and in so many fields. Yet, I never took the time to fully dive deep into them and really understand when and how they were useful tools. When I learned about a Fourier infrared spectography method used in chemistry to determine the structure of a compound, I decided to try to understand what I was missing.

The best place I found to start was thinking of an elementary analogy: that of a smoothie. We can think of a Fourier transformation, in the most basic sense, as being able to determine the amounts of ingredients of a smoothie. If you give me a specific smoothie, I could tell you that you used:
- 1 oz banana
- 3 oz strawberries
- 1 cup of milk
and so on. However, there is a constraint. *You must know the list of all possible ingredients.* I wouldn't be able to find all of the ingredients if I didn't know one of them even existed. After having all of the ingredients of the smoothie, I can get our original smoothie back by simply mixing the ingredients together.

Now let's move up a level, and look at the field of audio recording. Say you're an amateur band, recording your first song for release. You set up your microphone and instruments, and begin to warm up. You start a practice recording with an A chord. How would this graph look?

![download](https://user-images.githubusercontent.com/83550862/213028807-d1456868-5ca2-4e6c-8639-4d07961974dc.png)


In this graph, the $x$-axis represents DB (sound level), and the $y$-axis represents time. However, say a friend of yours takes your recording, and wants to decompose which note you played. How would they go about doing this?

Obviously, this isn't possible by looking at the graph. Although there is a clear pattern, you can't easily turn it into a set of notes. What the above graph represents, rather, is the interference pattern between the 3 notes you were playing. 

![figure1](https://user-images.githubusercontent.com/83550862/213028707-11a756c0-80d5-4bdc-8392-1a3f396d64a6.png)


Fourier transformations can take this composite, **time valued** function, and turn it into a set of **discrete frequencies**. This is the core of a Fourier transform; *turning a time series into a set of discrete frequencies which overlap to produce the time series*. You could see how this has many potential applications, from optics to audio recording. But how does it work?

Recall the 3b1b video about Fourier transforms. What he did was **wrap the linear function around a circle**, turning our sinusoid into a circular curve. Imagine taking a pointer which turns around the circle at a constant speed, maybe 1HZ. The length of this pointer is defined by the height of our sinusoidal graph, so it draws a curve around and around that models our composite function. One interesting modification you can make is the speed of the drawing pointer, making it spin around faster or slower. The key is to calculate the **center of mass** of the circular curve you draw. As you vary the speed of the drawing pointer, you'll also vary the center of mass, as the curve will look different. Graph the center of mass vs the speed of the drawing pointer, and you'll end up with a graph similar to the one below.

![Screenshot 2023-01-17 164435](https://user-images.githubusercontent.com/83550862/213028691-9c57f8ca-a2a7-4ea4-b119-27b296fe9d03.png)


It turns out that when you look at this graph, you'll see a spike right at the frequency of your original function. If there are 3 frequencies, you'll see 3 spikes. Thus, graphing the A chord in this manner will show you 3 spikes, right on the A, C# and E frequencies.

<hr>

# Chem Application

This is applied in the chemistry spectrometry in a clever way. Remember the diagram for the FTIR spectrometer:

![image](https://user-images.githubusercontent.com/83550862/215231681-401ca070-9203-4d4c-befd-6dbb20ab7185.png)


Furthermore, recall the effects of two sound frequencies overlapping. Now we have a similar thing happening, this time with radiation frequencies. The beam splitter splits up the light: part goes to the fixed mirror, and part ot the moving mirror. This makes the light **out of phase**, which gives us something to measure against time essentially. Then, this light combines and goes through the sample, where it is absorbed. The moving mirror allows the IR to scan throughout the whole spectrum, and after the whole spectrum has been accounted for, it goes through a fourier transform to be transformed into spectral data.
