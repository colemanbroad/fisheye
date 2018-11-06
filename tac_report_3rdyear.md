---
title: "Third year TAC report"
author: [Coleman Broaddus]
date: "2018-11-06"
fontsize: 10pt
links-as-notes: true
---

# Title

Tracking cells during embryogenesis

# Abstract

Understanding when and where individual cells move, divide and die during embryogenesis is a fundamental part of modern developmental biology.
Current fluorescence microscopy allows us in principle to track the movement of cells throughout the entire embryo and across the full timespan of development for a variety of model species.
However, so far automated cell tracking algorithms still struggle to deal with tracking during development as it poses many qualitatively new challenges to the field of cell tracking.

During my PhD I have focused on new models and algorithms for cell tracking during development and the upstream tasks of cell detection and segmentation.

Specifically:

- I propose an extension to Tracking by Assignment (TbA) [@jug2014tracking] appropriate for cell tracking in data with large-scale tissue flows that obviates the need for additional computation of optical flow.
    This technique improves the quality of TbA in systems where cell motion is spatially correlated e.g. during *D. melanogaster* gastrulation and lesion regeneration in the flatworm *S. mediterranea*.
<!-- - I have implemented and applied convolutional neural network (CNN) models for pixel-wise classification on zebrafish retina data.  -->
- I have developed new models for nuclei detection using neural networks which enable the localization of small, densely packed nuclei in regions with poor image quality.
These models improve downstream segmentation and tracking and currently hold state of the art *C. elegans* nucleus segmentation scores in the [2018 ISBI Cell Segmentation Challenge](http://celltrackingchallenge.net).

# Introduction / Aim

Modern fluorescence microscopy allows live imaging of entire developing embryos with single cell resolution, yet automated methods for extracting information from these images still lag behind.
In particular analysis of cell location, shape, motion and lineages still depends largely on manual curation by human experts.
Automated analysis is difficult on this data for several reasons:

**The Problems of Cell Tracking During Embryogenesis**

1. large number of densely packed nuclei
1. variability of nucleus size and shape across across tissues, cell cycle and stages of development
1. variability of image quality across space and time: sample-induced distortions.
1. dense pixel-wise annotation is difficult in 3D data, which restricts many learned approaches
1. ambiguous data: multiple annotations may not agree...
<!-- - variability of image quality across space and time # (limited penetration depth, photobleaching, reflective tissue surfaces being created, etc...) -->
<!-- - anisotropic acquisition (voxel size and psf) -->

Learned approaches -- in particular convolutional neural networks (CNNs) -- have been shown to work well on many image analysis / computer vision tasks, achieving state of the art results across a wide variety of benchmark detection and segmentation challenges in BioImage analysis [@ronneberger2015u], natural images [@he2017mask] and medical images [@cciccek20163d].
Despite the dominance of learned methods for similar tasks current state of the art approaches to cell tracking during embryogenesis are entirely classical / unlearned [@amat2014fast], [@bao2006automated], [@mace2013high], [@magnusson2015global].
However these approaches suffer from limited accuracy and have many sample-specific and even acquisition specific hand-tuned parameters.
<!-- In the end manual curation of the initial automated results is still a major bottleneck. -->
This motivates the question:

*How can we harness the power of CNNs to help with detection and segmentation during embryogenesis?*

If our tracking result is imperfect it will require some amount of manual curation by an expert.
Tracking by Assignment (TbA) is a modeling framework for cell tracking that allows for highly flexible and accurate models, while providing the unique advantage that it is possible to include manual curation and annotations as constraints from which we can extrapolate our solution into uncurated regions.
When applying TbA to developmental data with large-scale cell flows it is common to additionally compute optical flow on the dataset and to use this information to extrapolate the expected positions of cells into future time-points.
Unfortunately, the computation of optical flow doesn't doesn't benefit from user curation, and can even mislead the subsequent tracking step if the many hand-tuned parameters are not properly fixed.
Thus, we ask:

*Is there a way of enhancing Tracking by Assignment for embryogenesis by more tightly incorporating cell flow?*

<!-- 
## delete

This begs the question:

*Why haven't convolutional neural networks taken over the world of embryogenesis and development?*

Some potential explanations:

1. There is a lag in technology transfer from the world of natural image processing (and even medical image processing) to development... we are a smaller community!
2. As mentioned above, part of the reason is - dense pixel-wise annotation is difficult in 4D, even in clean data, which slows and restricts many learned approaches.
3. In development the biologist is always trying to acquire an image of something *no one has seen before*. This means the data is always changing and you can't build up a large repo of GT annotations!!!!
4. Microscopes are changing too.
5. Some problems are dev-bio specific, like obtaining lineage trees. In most visual tracking problems objects don't divide in synchronous waves!
    Instance seg of a large number of densely packed, similar objects is also not found in the Medical imaging world!
    Medical images like CAT scans, Histology, Xrays, MRIs, Diffusion fMRIs, etc don't usually have TIME. Semantic Segmentation is a much more important problem than Instance Seg and Tracking.

The main problems involved in instance segmentation during development are:

- missing cells in regions of poor image quality
- merging multiple cells together when they are densely packed, or under-resolved (iso-z)
- not being able to accurately extract cell shape
- not being able to identify mitotic cells (inherent ambiguity as to 1-cell vs 2-cell when only Histone label is available.)

Main problems in tracking:

- cells appear and disappear. inherent all problems / errors from upstream detection / segmentation. full joint detection/segmentation doesn't solve the problem, and has significant complexity/overhead. (MLT, joint detection/tracking)
- large jumps during divisions. this sets lower bound on sampling frequency. can try to speed up acquisition just during divisions, but not well established. TbA and StarryNite incorporate explicit division costs. How much does this help?
- waves of division make problem harder.
- optical flow very important first step for large-scale tissue flow. not a lot of good optical flow for 3D data.
- but SEG scores always lower than TRA scores! in some sense segmentation is *harder*. (also unnecessary for good tracking?).
 -->

<!-- 
# Aim

During my PhD I have explored new models and algorithms for tasks that are important to the study of embryogenesis at the cellular level: detection, segmentation and tracking.
Specifically, in segmentation I aim to address the issue of 3D segmentation of nuclei in very dense tissue with minimal training data.
My work on detection and segmentation builds on the U-net CNN model, which has recently become ubiquitous in the field, having been used for everything from cell segmentation (Ronneberger, 3D Unet) to image restoration (CARE). By carefully constructing an output for the U-net based on user center-point annotation we can formulate center-point finding as an image-to-image learning task. I show that this formulation of the problem enables state of the art detection and segmentation of densely packed cell nuclei in 3D fluorescence microscopy datasets.
And in tracking I aim to address a problem of Optical Flow + Assignment Model by integrating the two steps together. This is both a simpler, more intuitive and more general way of introducing prior knowledge of coherent tissue flow, and enables a more powerful leveraged editing system that is not hostage to the mistakes of an initial optical flow step.
My work on tracking builds on top of discrete Assignment models as introduced by (Hamprecht) and pushed by (Jug, Andres, etc). ^[While DL approaches for cell tracking *do* exist, they are not very good. Current state of the art for tracking algs is classical maybe with a small amount of learning (structured learning Hamprecht). This may be an inherent property of tracking, where the relevant features for describing motion (differences in known location and appearance ) are simpler to describe by hand than it is to extract those features from the raw image. Basically, we know that the features should be as similar as possible across time, except for during divisions.]
 -->

<!-- All of  solutions are motivated by *embryogenesis specific* problems in bioimage analysis. -->

<!-- Each technique I demonstrate / validate on a variety of datasets.
I demonstrate / validate the detection/segmentation models on zebrafish, and c. elegans.
The tracking works on flies and planarians.
 -->

# Approach

## Tracking by Assignment with coherent flow prior

It is common for cell tracking methods to perform an initial optical flow step on raw image intensities before detecting or linking cells across time.
The optical flow is used a prediction for cell location in future time-points which incorporates the prior knowledge of 'flow', i.e. that nearby objects like to move in similar directions.

There are several problems with this approach:

- Optical flow uses *image similarity*, linkage uses *object similarity*.
- Parameterization of flow field requires arbitrary coordinate system decision (i.e. square grid). This breaks rotational and translational covariance of solution.
- Errors in flow propagate to tracking. We can't re-solve for flow using partially curated results.
<!-- - slow -->

<!-- Our proposed solution: -->
Our proposed solution includes flow information directly in the cell linkage step via *velocity gradient costs*.
We add a cost to the assignment model for each pair of neighboring cells which is low if the cells move in the same direction and high if they move in opposite directions and has a magnitude that decreases with the distance between cells.

<!-- 
- include flow information in the tracking model directly via "velocity gradient costs". essentially there should be a term in the energy/objective that penalizes for sharp spatial gradients in object velocity / displacement between time-points. This is easily theoretically motivated by the viscous flow term in Navier-Stokes equation!
- add velgrad term (and associated binary variable) between all pairs of prospective time-edges on (voronoi-)neighboring objects.
 -->

Benefits:

- Solve tracking and flow jointly. Not stuck with errors made by optical flow.
- We only have to describe object similarity.
- Natural vectorial representation. We avoid an arbitrary choice of coordinate system which gives us natural translational and rotational covariance. Same cost in 2D and 3D.
<!-- - fast -->

Downsides:

- We grow the size of the assignment model solved by $nk^2$, where $n$ is the number of edges in the cell neighborhood graph and $k$ is the number of edges for each cell into the next time-point.

## Tracking results

We demonstrate the merits of our approach on the flatworm *S. mediterranea*.

![(a) raw (b) nucleus classifier (c) raw + tracking (d) cost higher when neighboring displacements differ (e) cell neighborhood graph](/Users/broaddus/Desktop/slice1.png)

<!-- 
- picture of worm. 1) raw. 2) ilastik 3) velgrad neighbors 4) raw + arrows
- 1) raw + arrow w/out velgrad. 2) raw + arrows w velgrad
 --> 

The extreme light sensitivity of *S. mediterranea* induces muscular twitches leading to large "jumps" in nucleus position even at high framerate. 
TbA with velocity gradient costs is able to track nuclei even across these large displacements.
We find that including flow information via velocity gradient costs is more accurate, has fewer hyper-parameter, and runs faster than comparable optical flow techniques.
We have implemented a simple model for Tracking by Assignment with optional velocity gradient costs in Python and Gurobi which runs on the Furiosa cluster and is publicly available named [TrackTools](https://github.com/mpicbg-csbd/segtools).

## Center-point detection and segmentation from weak annotations

<!-- 
Segmentation of densely packed nuclei during embryogenesis is extremely challenging, (see CTC).
While learned approaches generally perform better on a given dataset, they require expensive/time consuming manual annotations for each new sample, protocol and imaging modality. The is a paucity of labeled examples available for most biological samples and imaging modalities. And no pretrained super-nets capable of generalizing across these data while maintaining high accuracy.
Typical learned approaches begin by training a pixel-wise classifier from image data where an expert has labeled (drawn on) a subset of the pixels corresponding to "foreground" (in our case, cell nuclei) and a second subset corresponding to "background" (non-nuclei). ^[
    This approach to labeling is independent of learning method.
    There are extensions that can improve segmentability or learnability, e.g. adding extra classes for explicit borders (DCAN,Stardist), instance labels for inidividual objects, labeling slices, etc.
    Also, one must choose which subset of the data to label. This is an especially complex choice for pixel-wise classification given that it is possible in the limit to label (draw on) every single pixel in the dataset! (or at least those pixels belonging to nuclei).
    And there is the inherent problem of ambiguous data which means multiple independent labelings disagree.
    Center-point annotations require only a single point / nucleus, while dense pixel-wise annotations require (in the limit) labeling every pixel within the volume of the nucleus.]
 -->

<!-- Typical learned approaches begin by training a pixel-wise classifier from image data where an expert has labeled (drawn on) a subset of the pixels corresponding to "foreground" (in our case, cell nuclei) and a second subset corresponding to "background" (non-nuclei).
After classifying each pixel the segmentation must then decide how to group the nucleus-class pixels into nucleus instances. or equiv it must find borders between nucleus instances.
This step poses most of the difficulty in the segmentation problem and is usually unlearned, most often being done by simple threshold + connected components.
 -->

We treat nucleus center-point detection as an image-to-image regression task. The input is the raw image and the target is simply a convolution of the hand-annotated center-points with a smooth Gaussian kernel. We train a U-net to perform this regression task and extract the final center-points as the peaks of the predicted Gaussian kernels.
This approach gives highly accurate center-points even in regions with very densely packed nuclei. We can then use these center-points as seeds in a subsequent watershed segmentation, with the watershed potential coming from a simple pixel-wise nucleus classifier.
This hybrid approach requires only very few pixel-wise annotations relative to an instance segmentation method based on pixel-wise classifiers without a separate detection step.

![(a) raw *C. elegans* ~150 cell stage (b) output of center-point network (c) center-points after peak detection (d) pixel-wise nuclei classifier (e) instance segmentation after watershed (f) plot of cell count over time for various center-point prediction models compared against true cell counts (GT). Insets show how nucleus size decreases with each wave of division.](tac3_figs/slice2.png)

<!-- An alternative approach to instance segmentation taken by (mask r-cnn, Nico Sherf) is to first detect individual object instances, giving an approximate location for each object *before* moving on to the subsequent task of providing of a precise pixel-level description of object shape.
    Stardist is one of very few approaches which actually merges both the detection and shape-description tasks into one single step.
    This approach makes more sense for small densely packed objects, where boundary pixels are often shared between objects, and shapes are relatively simple.
We take this approach, following similar prescription to (Nico Sherf, 3D Gaussian kernel prediction forest) in that we treat object localization as an image-to-image task of regressing a mixture of (unnormalized) Gaussians discretized on the pixel grid with a single kernel located at each nucleus center-point.
    Here we use a standard U-net to learn this regression task.
    The gaussian mixture target requires only single-click center-point annotations, (easily made w Fiji's multipoint selector) which are relatively cheap and widely available in comparison with full, dense pixel-wise labelings.
[diagram of process with closeup of nuclei, target and prediction.]
[second diagram showing full segmentation pipeline.]
The predicted nucleus center-points then serve as *seeds* in a subsequent watershed segmentation.
    The watershed potential can also be learned.
    In our c.elegans data we had access to a very small number (10) instance-labeled xy-slices from specific time-points and z locations. We explored a variety of methods of learning the watershed potential from these data, including training a small "miniature" (10k params) CNN w/out downsampling layers to regress distance-to-boundary for each nucleus. This distance-to-boundary prediction serves directly as watershed potential.
 -->

## Detection and Segmentation results

Our hybrid approach gives state of the art results on a challenging *C. elegans* benchmark dataset (segmentation (SEG) score of 0.52; see ISBI Cell Segmentation Challenge 2018), but has additionally been show to work on zebrafish retina where cells are even more densely packed, shapes are less spherical and fewer annotations exist.
Note that since submission our models have improved significantly as a result of changes to the network architecture and to construction of the target kernels in a size-adaptive fashion [@broaddus2019cell].

# Future Plans

There a number of simple extensions to both the tracking and detection/segmentation models that would be interesting.

Tracking:

- validate on benchmark datasets
- unification with Active Graph Matching [@kainmueller2014active] <!-- AGM also defines a neighborhood relation on atlas nuclei, and has a cost when the distance between nuclei in atlas differs from distance between them in the target. Is this cost exactly the same? Is AGM the same as ours, but with registration included? -->
- special cost for divisions. nuclei move rapidly in direction *opposite* that predicted by flow.

Detection/Segmentation:

- Add more information to the prediction target:
    - bootstrap nucleus size and shape predictions from initial watershed segmentation on GT center-points
    - confidence in prediction / likelihood that center-point is correct

<!-- - Segtools: Python library for curation, analysis and visualization of bioimage data
    + Visualizing raw images
        + a simple stack viewer for python
        + integrated slice and volume rendering with clicking
        + volume rendering command line control
        + (fancy) max projections
    + Visualizing segmentations
        + pallettes: segmentation and detection overviews for very large datasets
        + RGB images w outlines and recolorings. Entropy coloring. Matching and score coloring.
        + scatterplot objects by feature with dot as 2d max projection over bounding box
        + coloring: ramsey coloring, instance edge coloring, etc.
    + Scores and metrics for segmentation and detection
        + python library for cell segmentation and detection metrics
    + Annotation
        + center-point annotation combined with cell removal
        + curating nuclear detections in 3D viewer
    + sorting and displaying network predictions ordered by loss score?
    + ilastik segmentation on Drosophila melanogaster
 -->

<!-- - Segmentation and tracking in dorosphila early embryogenesis.
    5k cell stage. stage 6 - stage 11?
- Segmentation and tracking in Planarians (latin name?) with novel velgrad costs for tracking w large displacements.
- Segmentation and detection in zebrafish retina w very high cell density where previously only semiautomated methods existed.
- Detection, segmentation and tracking in benchmark c. elegans dataset with SoA SEG scores.
- Instance segmentation of zebrafish cells in 2D during early embryonic stages.
- 
 -->

# References

