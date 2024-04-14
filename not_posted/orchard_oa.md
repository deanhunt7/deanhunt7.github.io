# Take-Home Challenge – Software Engineer Intern – Orchard Robotics

## Introduction

In this challenge you will build a 3D depth estimation and scene mapping solution to extract depth data from stereo camera images and create combined 3D maps of a row of apple trees in an orchard.

This challenge consists of three parts:

1. Creating good depth maps from input images

2. Combining consecutive depth maps into a 3D reconstruction of the scene of apple trees

3. (Bonus points!) Computing and extracting various statistics and insights from the 3D scene

This challenge is fairly challenging, so don’t worry if you are not able to complete all parts. Showing proficiency on even one or two parts is an accomplishment in and of itself, and will be looked upon highly during evaluation, given that good, scalable, maintainable code is written in a logical manner.

To achieve the best success with this challenge, sufficient understanding of a couple of fundamental robotics topics is ideal. These topics include:

- Stereo vision (see [this link](https://www.mathworks.com/discovery/stereo-vision.html), and [this](http://vision.stanford.edu/teaching/cs131_fall1415/lectures/lecture9_10_stereo_cs131.pdf))

- 3D reconstruction / mapping (see [this](https://en.wikipedia.org/wiki/3D_reconstruction))

- Basic open-source image processing / robotics libraries like [OpenCV](https://github.com/opencv/opencv)

### Input Data

The input data will consist of 20 different 3648 x 5472px images of apple trees in an orchard. These images are paired into 10 different “stereo pairs” of images taken synchronously by a pair of [stereo cameras](https://www.mathworks.com/discovery/stereo-vision.html). You can find all of the [input data in Google Drive here](https://drive.google.com/file/d/1IuvcGaiDjpFZpm0jzVxHwF2RNNY4HRB6/view?usp=sharing).

There also is one singular sample depth map image, to show an example of a good depth map that can be constructed from this data.

These stereo pairs of images were taken by cameras oriented top/down from each other, where one camera was offset 120 mm above the other. This number is the “stereo baseline” of the stereo setup. If you open these images side by side, you can see the slight perspective shift from the differing camera locations when viewing the images in unison.

The naming convention of these images is as follows:

[timestamp in milliseconds, to reference two images from the same stereo pair]-[camera number, either 0 or 1]-[identifier, either UD for undistorted image, PREDICTION, or DEPTH for the sample depth map].jpeg

These images have already been [undistorted and rectified](https://www.cs.cmu.edu/~16385/s17/Slides/13.1_Stereo_Rectification.pdf), so you will not have to worry about that when computing disparity/depth maps. In addition, there is about 70% overlap between any two consecutive stereo pairs, although this overlap is not necessarily consistent, due to random variations in the speed of the camera when taking these images. This overlap can be seen by opening any two consecutive images from the same camera, and comparing the portions of the scene that can be seen in both images.

Feel free to make any kind of automated edits to the images during pre-processing (i.e. You may write code that increases the brightness, contrast, etc of images, but any image edits should be done in an automated fashion. But no manual edits using Photoshop, etc).

### Tasks

1. Your main task is to [create depth maps](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html) from the stereo pair input images.
	- This can be done simply by computing stereo disparities from the image pairs, and converting them to depths.
	- Assume that the focal length of the cameras is not known, and therefore the depths are unit-less (however, relative “depths” can still be calculated from the disparity maps).


2. Based on these depth maps, you will need to find a way to stitch them together to create a [combined 3D reconstruction from the scene](https://rpg.ifi.uzh.ch/docs/teaching/2018/10_3D_reconstruction.pdf).
	- This means that you will have to go from multiple, discrete depth maps, into one combined “depth map”, or 3D reconstruction, that encompasses all the trees / scene covered in the 10 given stereo image pairs.
	- This is possible because the images contain a great amount of overlap, so the depths / features in one image pair can be matched up with the corresponding features in the previous / next image pair.
	- One simple way to approach this would be through the [iterative closest point algorithm](https://en.wikipedia.org/wiki/Iterative_closest_point), but there exist many other, better ways to approach this problem too.
	- After creating this combined 3D scene, create a visualization of the scene. (hint: this can be done in a “combined depth map” form, where the output is one extended “depth map” that includes all the trees from each of the 10 stereo images. Another more advanced possibility is a [true 3D visualization](https://i.ibb.co/grGhKSN/Trees-Volume-Sample.png) of the scene as a collection of 3d points, or as a mesh, using matplotlib / similar tools)

3. (Bonus Points!) Afterwards, given this holistic 3D reconstruction of all the apple trees in the scene, you will need to determine the total volume of all the trees.

	- You may assume that the trees are symmetrical, i.e. the volume of the tree on the opposite side of what the camera can see is symmetrical / identical to the volume of the tree that can be seen.
	- In addition, you should also try and determine the volume of each tree, although this will require devising a method to segment trees from one another (hint: you could create some volume threshold by scanning a vertical line across the scene to see where the tree canopy becomes less dense, as there is considerable empty space in between any two given trees).
	- Afterwards, you should also try and determine the average height of each tree from ground level.

*If you are using third-party code or libraries, please provide (brief) explanation of why you need that code and what that code does. We evaluate your submission based on the code you have written and if there is no such code, we won't be able to evaluate and proceed to the next stage.*

## Rules & Submission

- Feel free to use any open-source python libraries in your code, as long as you explain concisely what every imported function does (we recommend using OpenCV!)
- For task 3 you are not allowed to use third-party functions that readily solve those tasks, e.g. you are not allowed to use various `cv2` and `scikit-image` operators. We expect the algorithms to be based on points and geometry rather than full-image operations.
- You must send us only a single link to the Colab notebook with your solution and nothing else! We should be able to reproduce your results by running the notebook.
- Include your detailed comments and explanations on the decisions, rationale, and logic that you made to approach the problem. There is partial credit! – even if your solution for any part isn’t fully working, please make sure to still document your thought process / whatever code you attempted
- Make sure to include an estimate of approximately how much time it took you to get to the final solution.
