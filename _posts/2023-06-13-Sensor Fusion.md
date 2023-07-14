This blog post is mainly to help shore up my understanding of sensor fusion (the non-classified bits, anyway) and help get some things firmer in my own mind. I want to present the general sensor fusion pipeline, then take some time delving into each step to see the engine behind.

# The Elephant

A commonly used metaphor for describing sensor fusion is the old tale of the blind men from Indostan inspecting an elephant. 

![image](https://github.com/deanhunt7/deanhunt7.github.io/assets/83550862/a8ee0158-85da-4878-86c1-ef49b460d385)

The first man touches the side, and believes the elephant to be strong and sturdy, like a wall. The second touches the tusk, and believes the elephant to be very like a spear. This continues, with each man believing the elephant to be of a very different nature from the other men. If Indostan had developed sensor fusion, these men would recognize their problem. Alas, these men didn't have access to this technology, and remained ignorant about the true nature of an elephant.

If these men were to instead recognize their shortcomings (namely, that they were feeling different parts of the same elephant), they could begin to construct an emergent representation of the elephant based on their sensor readings. In this way, the men in the story are analogous to the senses in the human body, or the sensors in a jet. Each sensor reads different data, and sensor fusion (the brain in your body, computers on a jet) attempts to create a cohesive narrative based on these individual readings.

# The Jet

The F-35 has the most advanced sensor fusion technology on the planet at the moment. Sensor fusion on a jet is specifically designed for combat, where situational awareness (SA) is key to a victorious encounter with an enemy. The main priority of a fusion system is to locate enemy positions, track the enemies, and predict their future paths. The fusion process is generally broken down into a recursive cycle of decision processes, shown below.

![image](https://github.com/deanhunt7/deanhunt7.github.io/assets/83550862/d2227d6b-daef-4463-a085-ab2e4a6afcd8)


The first step of the loop is to gather observations from on-board and off-board sources. On board sources include radar, EW (electronic warfare) sensors, the DAS (Distributed Aperture System, basically a big array of cameras), and other sensors. Off-board, the F-35 has data links through which it gathers more data from other aircraft and ground units.

All this data goes into the VIM (virtual interface model), which essentially translates the data from each sensor into a usable form for the remainder of the process. Remember, the data picked up from the sensors is completely different from one another, so it's imperative to at least have some semblance of initial standardization before attempting to fuse the data into a coherent picture.

### Data Association

The first real step is the data association step. Imagine, as a computer, you've got a set of sensor data. You know that there are one or more enemy units somewhere within the data, but you've got a few problems. First, you need to decide what constitutes an enemy and what doesn't. This is a fairly well-defined process that uses unsupervised machine learning algorithms to take in a set of data and sort it into valid "sightings". Second, you'll need to decide which enemy is which. The computer sorts various entities into "tracks", representing a single element to track. The computer needs to determine the mapping between current readings and tracks, called data association.

There are a few ways of going about the mapping process. At the end of the day, the computer wants to have each observation mapped to a single track. First, simple gating happens to narrow down possible mappings, based on given errors in the sensors. These errors are added to create a spherical gate, and observations are pruned based on volumetric distances. 

Next, the mapping process. This can be done using a method called single hypothesis tracking (SHT). SHT is a one-to-one mapping between observations and tracks, creating new tracks for unpaired observations. SHT generally uses a GNN tracker (global nearest neighbor), which deterministically maps observations to tracks using a GNN algorithm. While this is a trivial process for single observations, mapping between multiple observations and tracks becomes an interesting optimization problem using the [Hungarian (Munkres) algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm). SHT generally isn't optimal though, for a few reasons. Oftentimes, one entity may produce multiple observations, which isn't recognized by SHT when creating the mappings. Furthermore, SHT can tend to produce locally optimal solutions, since GNN can make assignments that are locally optimal for those specific readings, but globally suboptimal given further information.

An alternative to this is multiple hypothesis tracking (MHT). MHT is kind of like a non-deterministic automata, in that there are threads of possibility kept alive until they eventually collapse into one association. With this algorithm, multiple possible hypotheses are considered until an optimal one is decided, leading to the global optimal mapping. (talk about JPDA collapsing it into one mapping) However, MHT is computationally intensive, due to the various possible associations that must be kept in context. Because of this, hypothesis pruning is essential to keep the mapping process lean and more efficient.

At the end of the day, the data association step outputs a mapping of observations to tracks. The data association step is by far the most computationally intensive part of the fusion process.

### State Estimation

Now that we've got the mappings, we can begin arguably the most important part of the fusion process: predicting the current (and future) location of the enemy. Each track has previous state data, which needs to be updated to real-time to support the pilot's SA. 

One of the wonders of kinematics are the tools it provides for describing the state of an object in space. Kinematic data is collected for each track (heading, velocity, az/el, acceleration) from sensors on the aircraft. The kinematics of the track can be extrapolated to predict the future kinematics of the aircraft (which, luckily, must continue to obey the laws of physics between observations). Obviously, this is insufficient; current heading, velocity, and position isn't indicative of future heading, velocity, and position. We need to integrate real measurements of the track to have an accurate prediction. The problem is that sensors, like kinematic extrapolations, aren't completely accurate. Sensors come with biases, both from innate uncertainties in the hardware and installation quirks. We'll end up with two different predictions for the current state of the track: one from kinematic extrapolation, and the other from our sensor measurement.

To resolve this, we use the trusty [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter). The Kalman filter calculates a gain, which I like to think of as the measure of your trust of the sensor. My linear algebra chops are pretty rusty, so I'm going to brush up by writing a Kalman filter post. Until then, just trust me when I say that the Kalman filter allows for more accurate state estimations. (In the meantime, a cool Kalman application I found [here](https://tng-daryl.medium.com/implementing-the-kalman-filter-on-stock-data-1dce3a192a93). It turns out there's an unexpectedly intuitive parallel between fighter jets and the stock market when it comes to state estimation). 

In the end, you wind up with a state estimation somewhere between your sensor measurement and your kinematic extrapolation, depending on the gain calculated by the Kalman filter. This state now represents the aircraft's best guess regarding the current location of the enemy.

### Sensor Scheduling

Finally, we need to collect more data, given that the states of the various tracks have changed while we've been calculating. Unfortunately, the plane only carries a finite amount of sensors, so we'll need to prioritize. Sensor scheduling is the process through which the computer prioritizes data collection for various tracks. I assumed this would be a big potential for some cool AI applications; but it turns out there isn't a lot of that (at least right now). When I talked to one engineer on the scheduling team, I learned there are a multitude of hard-baked rules for sensor scheduling that allows the computer to quickly schedule sensors and collect data. It makes sense; until we can get [good neural net chips](https://www.quantamagazine.org/a-brain-inspired-chip-can-run-ai-with-far-less-energy-20221110/), it'll be hard to bring along the compute necessary for that kind of application on the edge.

### ...and Recurse

The fusion system on an F-35 is completely self-sufficient, in that it requires no human input to continue developing and maintaining tracks. The recursive nature of the Kalman filter and state estimation allows for DP-like optimizations which improve the efficiency of the system.

And there you have it. This post is always a work in progress, so I'll keep adding to it as I learn more. Thanks for taking the time to read. I hope it was a useful post; I know it was definitely useful for me to write it.
