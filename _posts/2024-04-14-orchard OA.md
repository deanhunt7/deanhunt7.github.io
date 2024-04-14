<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: { skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'], inlineMath: [['$','$']] } }); </script> <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> 
I had planned to make this a really technical post, but instead it's turned into more of a speculative one. I'm not sure where I've gone wrong or what I'm missing, but I'll have to revisit this problem in the future. I'm stuck right now.

# the problem

Precision agriculture is going to be huge in the future. When I looked at some of the literature, I was shocked at how inefficient agriculture currently is. I was even more shocked at how easy it was to make a gigantic improvement on this. [ecorobotix](https://ecorobotix.com/en/) has been able to make reductions in pesticide use of up to 95%, and from what I can tell, they're just using directed spray and a CNN to identify what plants are weeds to be sprayed. When you think about it, spraying the entire field would amount to about 95% waste, since it's likely that more than 95% of the field is *not* weeds.

It would also be really useful to be able to track the health of individual plants. Companies like [orchard](https://www.orchard-robotics.com/) are working on this right now. I interviewed for an intern position there, and [this](https://github.com/deanhunt7/deanhunt7.github.io/blob/master/not_posted/orchard_oa.md) was the problem I was given.

Given orchard trees that look like this:

<img src="https://github.com/deanhunt7/deanhunt7.github.io/assets/83550862/cda5699f-fcd2-4988-a782-a90e823c228b" width="350" height="500" />

we need to create depth maps, stitch all the depth maps together to create a continuous map of the orchard, and as a bonus, calculate the volume of the trees.

# task 1: depth maps

A [depth map](https://en.wikipedia.org/wiki/Depth_map) is an image that contains information and the relative distances of objects in the image from the viewpoint of the camera. We often care about this in robotics because it allows us to build a model of the environment surrounding our robot. One very common way to create these depth maps is to use stereo vision, which is the same way our eyes work. Essentially, we use two cameras at a known distance apart, take two separate images, and calculate how far corresponding pixels in one image moved in the other image. Object that are closer will appear to move more than objects that are far away. From this, we can create a relative depth map for each pixel in the image.

[OpenCV](https://github.com/opencv/opencv) provides some really nice methods for doing this. In this challenge, I tried using both [StereoBM](https://docs.opencv.org/3.4/d9/dba/classcv_1_1StereoBM.html) and [StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html) algorithms for computing depth maps.

### StereoBM and StereoSGBM

They mentioned [this](https://docs.opencv.org/3.4/dd/d53/tutorial_py_depthmap.html) tutorial in the challenge, and it was the starting point for all my code.

Before processing the images, we have to [undistort and rectify](https://www.cs.cmu.edu/~16385/s17/Slides/13.1_Stereo_Rectification.pdf) them. Essentially, we make it such that corresponding pixels will be on the same axis for both images. This cuts processing time by a factor of $n$, because we only have to search for a corresponding pixel on one axis of length $n$, rather than searching the whole $n^2$ picture.

The StereoBM and StereoSGBM algorithms are [block-matching algorithms](https://en.wikipedia.org/wiki/Block-matching_algorithm). Matching individual pixels is a problem for two main reasons:
- It's not guaranteed that pixels will be the exact same in both images. Errors would be really high and the map would be extremely pixelated.
- It would be prohibitively expensive to compute disparities for each pixel.

Block-matching instead matches blocks of pixels between sets of images, creating disparity maps with lower resolution but higher accuracy. The *SG* in SGBM stands for "semi-global", and refers to the fact that the block-matching algorithm takes into account the disparities of neighboring pixels.

# the code

Grabbing some test images from the image repository:

```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

# download files, so you guys can run with the test pictures on my google drive

!rm -r top
!rm -r bottom
!mkdir top
!mkdir bottom
! gdown 1VZWBVzEAy_QhZMdSYfKb01-ZDDIVj0Wy -O "/content/bottom/bottom1.jpeg"
! gdown 1vKbtgSe8LAmHpufoXtnVqoQrECl3TYX1 -O "/content/bottom/bottom2.jpeg"
! gdown 1_XAX3_kb5owfJVC9QSoOYPNFPES-AYS5 -O "/content/bottom/bottom3.jpeg"
! gdown 1JMXzDLEiGRi_GdrzDEyAQ-UGa2KK47F7 -O "/content/bottom/bottom4.jpeg"
! gdown 1E-kK4WToK5VrAs9uuqw4deLRoDCI9y-_ -O "/content/bottom/bottom5.jpeg"
! gdown 19clqOj4XZ9SzKXqTirapneK-V0-mtLpi -O "/content/bottom/bottom6.jpeg"
! gdown 12rj_AA8Z5bN79-9uBXuy7oLDGrf8l9g6 -O "/content/bottom/bottom7.jpeg"
! gdown 1A3qIlfIONMD4icP9barHXHmt9BpkptPU -O "/content/bottom/bottom8.jpeg"
! gdown 11FqMO0acS6xmc3tw9dr7PqewcxoSQxXe -O "/content/bottom/bottom9.jpeg"
! gdown 1-HJKBI6nO_QVdt96kfBb_mP30eWL0BRh -O "/content/bottom/bottom99.jpeg" # calling it left10 won't sort correctly
! gdown 1mQbB4HttrFKoNrlMd6zpWhtwXvW__pdE -O "/content/top/top1.jpeg"
! gdown 1MBR1GXDlxcG_pNU1XuQRazz9MK_SMASa -O "/content/top/top2.jpeg"
! gdown 1qV8Yp6SFtVjACGGbXck7eMRJ7eV3FNpz -O "/content/top/top3.jpeg"
! gdown 10SjNXqPUwULWbVaO2NzlwddDuiHbC6Wg -O "/content/top/top4.jpeg"
! gdown 1SzFQJS2GUw-OlrLkMA_i9RYyA_9zNBRC -O "/content/top/top5.jpeg"
! gdown 17ZWSGls26kYvFJ_xPESHWABPL_ALu-Mn -O "/content/top/top6.jpeg"
! gdown 1zGTwLzSacbfTiVZKVl9jup_idxrUau5- -O "/content/top/top7.jpeg"
! gdown 1lNnYy2BfmYqKFNFfkAzDyabyCVDm2ODB -O "/content/top/top8.jpeg"
! gdown 188eqpeSgf2gLlL05WzaH83n61dkN4gFp -O "/content/top/top9.jpeg"
! gdown 1Xx6CTrHjLwQ0q10MWDwFwdT2E3QRqQAQ -O "/content/top/top99.jpeg"
```

I then read all top images into the `top_imgs` and all bottom into `bottom_imgs`. I then sort alphanumerically, to ensure the correct top/bottom pairs are in the same indexes for both arrays. I also added a `printMap` helper function, which helped me during debugging to visualize my disparity maps.

```
top_imgs = []
bottom_imgs = []
depth_maps = []

def printMap(x, title=""): # testing function, so I can see the maps
    plt.imshow(x, 'gray')
    plt.title(title)
    plt.show()

for img1 in os.listdir('top'):
    if img1 != '.ipynb_checkpoints': # sometimes colab adds this folder, messing up my image reading
        top_imgs.append(img1)
for img2 in os.listdir('bottom'):
    if img2 != '.ipynb_checkpoints':
        bottom_imgs.append(img2)

top_imgs.sort() # match up corresponding stereo images
bottom_imgs.sort()
```

Now to create the disparity maps. I used the block-matching StereoBM, rather than Semi-global block matching (SGBM), because it ended up giving me better disparity map results. This was interesting, because the semi-global algorithm takes into account the disparities of neighboring pixels as well. I would have assumed this would be better for the complex images of the leaves, and for 3-D reconstruction. But, StereoBM still ended up giving me better results at the end.

I may have incorrectly tuned parameters when I tried the StereoSGBM algorithm, giving me a better result with the StereoBM, but I'm not sure what I would have done wrong.

For the parameters, I chose minDisparity to be 0 (default), since we won't have any negative disparities for a linear translation of the cameras. numDisparities was 16, because the image didn't have extremely far-reaching objects to calculate disparities for, so I decided that a small number of disparities would work well. The BlockSize was chosen to be 5, which is on the smaller end, because I wanted more granular disparities to take into account the small leaves and fruits on the trees. Finally, I chose the uniquenessRatio to be 20. This means that for each pixel, the chosen value for it's disparity must be 20x "better" than the next possible value, otherwise it gets filtered out. This helps improve disparity map confidence.


```
for i in range(len(top_imgs)):

    imgT = cv.imread(os.path.join('top', top_imgs[i]),0) # 0 = read as grayscale
    imgB = cv.imread(os.path.join('bottom',bottom_imgs[i]),0)

    top_matcher = cv.StereoBM_create()

    top_matcher.setNumDisparities(16)
    # numDisparities: number of possible disparities from minDisparity (default 0), must be divisible by 16
    top_matcher.setBlockSize(5)
    # blockSize: matched block size for algorithm. Smaller values give more granular disparity values but are more computationally intensive
    top_matcher.setUniquenessRatio(20)
    # uniquenessRatio: this one ensures that your disparities are more confident.
    # each disp value must be 20x more confident than the next approximation (from my understanding)
    # otherwise the value is filtered out

    bottom_matcher = cv.ximgproc.createRightMatcher(top_matcher);
    # This is just a shortcut method for making the opposite side disparity matcher, for later filtering

    top_disp = top_matcher.compute(imgT, imgB); # compute disparity from top to bottom image
    bottom_disp = bottom_matcher.compute(imgB,imgT); # compupte disparity going opposite way for testing purposes
    printMap(top_disp, "pre-filtering")
```

![Untitled](https://github.com/deanhunt7/deanhunt7.github.io/assets/83550862/5e41c083-f0f6-44b2-aacb-331544254ae2)


After creating the initial disparity map, I began filtering the data. The disparity maps were quite noisy, and had important edges I needed to preserve, so I chose first to use a WLS filter. This filter is widely used for noise filtering with edge retention.

```
    wls_filter = cv.ximgproc.createDisparityWLSFilter(top_matcher);
    filtered_disp = wls_filter.filter(disparity_map_left = top_disp, left_view = imgT, disparity_map_right=bottom_disp)
    # Weighted least squares filter is used to smooth disparity values while retaining edges (important for tree boundaries)
    # this WLS implementation also uses the bottom matcher to compare disparity values to increase accuracy and confidence
	
    printMap(filtered_disp, "post-filtering") # print filtered output

    new = cv.bilateralFilter(filtered_disp.astype(np.float32), 10, 300, 300)
    # try bilateral filtering, for later stitching (to smooth out points and give more point correspondence)
    # for some reason, I couldn't get this function to modify my map at all. I'm not sure where I went wrong

    new = cv.cvtColor(new.astype(np.uint8), cv.COLOR_GRAY2RGB) #convert back to rgb (needed for stitching func)
    depth_maps.append(new)


    # NOTE: I tried filtering speckles (below), but the problem was that this function replaces speckles with
    # a constant value (newVal), rather than the average of all the speckles. If I could write a function
    # to instead replace the speckles with their neighborhood average, this function would work well

    # cv.filterSpeckles(img=filtered_disp, newVal=50, maxSpeckleSize=50, maxDiff=.5)
```
![Untitled-1](https://github.com/deanhunt7/deanhunt7.github.io/assets/83550862/93b14699-530d-4d2b-9a56-89bee2e421d1)

Honestly, this looks even worse than the unfiltered map.

# task 2 - combined 3-D reconstruction

Stitching works by finding multiple point correspondencies between two images, calculating rotation and translation matricies to line up these points, and then applying these linear mappings to the second picture to align it with the first.

My problem is that my "filtered" images are still too noisy, and there aren't enough point correspondencies to make stitching feasible. If I had nice smooth contours like the example depth map, I think the stitching would work much better.

I ended up trying lots and lots of filtering options in OpenCV, but none of them seemed to produce an output similar to the example given in the challenge picture folder. I tried bilateral filtering, speckle filtering, and a wls filter, all to little effect.

```
stitcher = cv.Stitcher_create(mode=1)

error, stitched = stitcher.stitch(depth_maps)
print(error)
plt.imshow(stitched)
plt.show()
```

I gave up on this stitching function approach, because my images weren't good enough to stitch automatically like this.

Rather, I began to look at manually calculating the corresponding points and making my own transformation matricies.

```
# NOTE: This cell is a demo, not working for all stitching


descriptor = cv.SIFT.create()
# I used a SIFT algorithm (scale invariant feature transform) as from what I found, this is a reliable way
# to find point correspondencies by first finding key points in both images and then matching these points

matcher = cv.DescriptorMatcher.create(1)
# the descriptor matcher helps find matching key points between two images

(kps1, desc1) = descriptor.detectAndCompute(depth_maps[0], mask=None) # find key points in image
(kps2, desc2) = descriptor.detectAndCompute(depth_maps[1], mask=None)

if desc1 is not None and desc2 is not None and len(desc1) >=2 and len(desc2) >= 2: # if there are key points in both images
    rawMatch = matcher.knnMatch(desc2, desc1, k=2) # match key points with k nearest neighbors algorithm
matches = []
# ensure the distance is within a certain ratio of each other
ratio = 0.75
for m in rawMatch:
    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
        matches.append((m[0].trainIdx, m[0].queryIdx))
        # trainIdx is keypoints of first image, queryIdx is keypoints of second image
        # matches holds all the tuples of corresponding keypoints

print(matches)
```

With these corresponding keypoints, I could theoretically calculate the rotation and transformation matricies required to stitch the images.

In the end, I think my inability to correctly filter the images held me back from starting on the other sections of the project, which disappointed me. I'm not sure what parameters I didn't tune correctly or what functions I could have missed from the documentation.

This solution ended up taking me around 20 hours to get to this point. The first few of these I created my disparity maps, and for the whole rest of the time, I was trying to simultaneously find a good way to stitch my images, and find a way to better filter my maps.

# other tries

I tried other approaches to see if anything would help create a cleaner disparity map. The first thing I tried was enhancing the images in an automated fashion by messing with their contrasts.

```
def enhanceImage(lab):
        temp_img = cv.cvtColor(lab, cv.COLOR_BGR2LAB)
        l_channel, a, b = cv.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

        return enhanced_im
```

LAB color space expresses color variations across three channels. One channel for brightness and two channels for color:

- L-channel: representing lightness in the image
- a-channel: representing change in color between red and green
- b-channel: representing change in color between yellow and blue

In the enhancement I performed [adaptive histogram equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#:%7E:text=Contrast%20Limited%20AHE%20(CLAHE)%20is,slope%20of%20the%20transformation%20function.) (also see [here](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)) on the L-channel and convert the resulting image back to BGR color space. Unfortunately, I didn't see much benefit to this in the disparity map quality (example below).

![Untitled](https://github.com/deanhunt7/deanhunt7.github.io/assets/83550862/3a4ae6d8-a387-4bad-ad32-b1fae8e496d9)

# reflections

When I first completed this, I felt really dejected. I wasn't able to complete even one of the three proposed tasks. Even after revisiting it, I've barely been able to make any progress. I dove really deep into the research regarding depth maps, and I simply can't understand what I'm missing.

It's clear to me why I'm getting poor results using the methods I am. Given that the tree images have multiple rows of trees contained in the same image, it's almost impossible to recognize which leaf corresponds to which other leaf. If there were only a single row of trees, it would be much easier to calculate, since we wouldn't worry too much about leaves at much different depths. It would be fine to be able to recognize all the leaves on the tree as being of the same depth, because at least then we would be able to begin calculating volumes of trees.

I'm assuming Orchard has been able to solve this problem, because they're still in business. I just can't figure out what I'm missing. I did briefly have the thought that this OA was a way of getting some free work out of prospective employees, but I don't really think that's true. I've set myself a reminder to check this after a long, long time. Maybe then I'll find the missing piece. Until then, this remains an incomplete problem in my mind.
