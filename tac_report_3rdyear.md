---
title: "Third year tac report"
author: [Coleman Broaddus]
date: "2018-11-02"
---

keywords: [Markdown, Example]

# Title

3D cell tracking during embryogenesis

# Abstract

Understanding when and where individual cells move, divide and die during embryogenesis is a fundamental goal of modern developmental biology.
Modern fluorescence microscopy allows us in principle to track cells visually throughout the entire embryo and across the full timespan of development for a variety of model species.
However, so far automated cell tracking algorithms still strugle to deal with tracking during development as it poses many qualitatively new challenges to the field of visual cell tracking.

~~I am interested in developing new models and algorithms for cell segmentation and tracking with a focus on embryogenesis.~~

During my PhD I have focused on new models and algorithms for cell tracking and the upstream tasks of cell detection and segmentation.

Specifically:

- I propose an extension to Tracking by Assignment appropriate for cell tracking in data with large-scale tissue flows and obviates the need for a separate optical flow step during preprocessing which is currently best practice.
    This technique enables cell tracking for the first time on the Planarian worm *name* and improves the visual quality of tracking on *drosophila melanogaster*.
- I have implemented and applied convolutional neural network (CNN) models for pixelwise classification on zebrafish retina data. 
- I have developed new CNN-based models for nuclei detection which make it possible to localize even small, densely packed nuclei in poor quality images.
These models enable downstream segmentations which are currently the state of the art on the benchmark ISBI c. elegans dataset.

# Introduction

Modern fluorescene microscopy allows live imaging of entire developing embryos with single cell resolution, yet automated methods for extracting information from these images still lag behind.
In particular accurate detection, segmentation and tracking of cells (often via Histone labeled cell nuclei) still depends largely on manual curation by human experts.
Automated analysis is difficult on this data for several reasons:

- large number of densely packed nuclei
- variability of nucleus size and shape across across tissues, phases of cell cycle and stages of development
- variability of image quality across space and time, in particular as a function of depth into the organism from the surface (often the image `z`-axis)
- in poor quality / ambiguous data, truth can be unknowable! multiple annotations may not agree.
- dense pixelwise annotation is difficult in 3D+t, even in clean data, which slows and restricts many learned approaches
# - variability of image quality actoss space and time # (limited penetration depth, photobleaching, reflective tissue surfaces being created, etc...)
# - anisotropic aquisition (voxel size and psf)

Learned approaches -- in particular convolutional neural networks (CNNs) -- have been shown to work well on many image analysis / computer vision tasks, achieving state of the art results across a wide variety of benchmark challenges in BioImage Analysis (U-net), Natural Image Processing (Mask-RCNN) and medical imaging (DCAN).
Despite the dominance of learned approaches for these tasks current state of the art approaches to cell tracking during embryogenesis are classical / unlearned. [Amat, SALT and Klaus Magnussen]. However these approaches suffer from limited accuracy and have many sample-specific and even aquisition specific hand-tuned parameters. In the end manual curation of the initial automated results is still the bottleneck.
This begs the question:

*Why haven't convolutional neural networks taken over the world of embryogenesis and development?*

Some potential explanations:

1. There is a lag in technology transfer from the world of natural image processing (and even medical image processing) to development... we are a smaller community!
2. As mentioned above, part of the reason is - dense pixelwise annotation is difficult in 4D, even in clean data, which slows and restricts many learned approaches.
3. In development the biologist is always trying to aquire an image of something *noone has seen before*. This means the data is always changing and you can't build up a large repo of GT annotations!!!!
4. Microscopes are changing too.
5. Some problems are dev-bio specific, like obtaining lineage trees. In most visual tracking problems objects don't divide in synchronous waves!
    Instance seg of a large number of densely packed, similar objects is also not found in the Medical imaging world!
    Medical images like CAT scans, Histology, Xrays, MRIs, Diffustion fMRIs, etc don't usually have TIME. Semantic Segmentation is a much more important problem than Instance Seg and Tracking.

The main problems involved in instance segmentation during development are:

- missing cells in regions of poor image quality
- merging multiple cells together when they are densely packed, or underresolved (iso-z)
- not being able to accurately extract cell shape
- not being able to identify mitotic cells (inherent ambiguity as to 1-cell vs 2-cell when only Histone label is available.)

Main problems in tracking:

- cells appear and disappear. inheret all problems / errors from upstream detection / segmentation. full joint detection/segmentation doesn't solve the problem, and has significant complexity/overhead. (MLT, joint detection/tracking)
- large jumps during divisions. this sets lower bound on sampling frequency. can try to speed up aquisition just during divisions, but not well established. TbA and StarryNite incorporate explicit division costs. How much does this help?
- waves of division make problem harder.
- optical flow very important first step for large-scale tissue flow. not a lot of good optical flow for 3D data.
- but SEG scores always lower than TRA scores! in some sense segmentation is *harder*. (also unnecessary for good tracking?).

# Aim

During my PhD I have explored new models and algorithms for tasks that are important to the visual study of embryogenesis: cell detection, segmentation and tracking.
Specifically, in segmentation I aim to address the issue of 3D segmentation of nuclei in very dense tissue with minimal training data.
My work on detection and segmentation builds on the U-net CNN model, which has recently become ubiquitous in the field, having been used for everything from cell segmentation (Ronneberger, 3D Unet) to image restoration (CARE). By carefully constructing an output for the U-net based on user centerpoint annotation we can formulate centerpoint finding as an image-to-image learning task. I show that this formulation of the problem enables state of the art detection and segmentation of densely packed cell nuclei in 3D fluorescence microscopy datasets.
And in tracking I aim to address the 2-phase problem of Optical Flow + Assignment Model by integrating the two steps together. This is both a simpler, more intuitive and more general way of introducing prior knowledge of coherent tissue flow, and enables a more powerful leveraged editing system that is not hostage to the mistakes of an initial optical flow step.
My work on tracking builds on top of discrete Assignment models as introduced by (Hamprecht) and pushed by (Jug, Andres, etc). footnote: While DL approaches for cell tracking *do* exist, they are not very good. Current state of the art for tracking algs is classical maybe with a small amount of learning (structured learning Hamprecht). This may be an inherent property of tracking, where the relevant features for describing motion (differences in known location and appearance ) are simpler to describe by hand than it is to extract those features from the raw image. Basically, we know that the features should be as similar as possible across time, except for during divisions.

Both solutions are motivated by *embryogenesis specific* problems in bioimage analysis.

Each technique I demonstrate / validate on a variety of datasets.
I demonstrate / validate the detection/segmentation models on zfish, and c. elegans.
The tracking works on flies and planarians.

# Approach

## Tracking by Assignment with coherent flow prior

Typical pipeline for tracking (not just TbA! also Amat's tracking! Corinna's flywing. Florian's tr2d?) does an initial optical flow step on pairs of images and includes that info in the tracker.
The idea of this two-step approach is to get a rough map of the trajectories of objects that includes the prior knowledge of 'flow', i.e. that nearby objects like to move in similar directions. mathematically: OF techniques allow us to incorporate a smoothness prior on the velocity field.

The problem with this approach are manifold:

- OF uses image similarity, tracking uses Object similarty.
- parameterization of flow field requires arbitrary coordinate system decisions: usually taken to be a grid. solution to OF not rotationally or translationally covariant. 
- often very slow
- two-step approach: errors in flow propagate to tracking. especially a problem if we want to curate the tracking solution! we can't resolve the flow using the information from our tracking corrections. they only know about images!

Our proposed solution:

- include flow information in the tracking model directly via "velocity gradient costs". essentially there should be a term in the energy/objective that penalizes for sharp spatial gradients in object velocity / displacement between timepoints. This is easily theoretically motivated by the viscous flow term in Navier-Stokes equation!
- add velgrad term (and associated binary variable) between all pairs of prospective time-edges on (voronoi-)neighboring objects.

Benefits are:



- we only have to describe object similartiy.
- natural vectorial representation. no arbitrary parameterization choices. natural translational and rotational covariance.
- solve tracking and flow jointly. no two-step problems.
- fast

## Tracking results

We demonstrate the merits of our approach on the planarian worm *worm*.
[picture of worm. 1) raw. 2) ilastik 3) velgrad neighbors 4) raw + arrows]
[1) raw + arrow w/out velgrad. 2) raw + arrows w velgrad]
Light-induced muscular twitches create large "jumps" in object positions between timepoints even at highframerate.
Object motion can be described by large, long-wavelength movements + short, short-wavelength, random movements. (!!! Can we actually show this decomposition experimentally??) Like Viscous flow of diffusing particles?
The velgrad cost model is able to track particles across these large gaps. Works where Fiji optical flow fails. Works as well / better then deepflow in fraction of the time with fewer params to tune.
Also Works in 3D on the fly...  Totally general 2D/3D. [show improved tracking via visual analysis.]
Available along with ILP solving framework right now on cluster with TrackTools module.

## Centerpoint detection and segmentation from weak annotations

Segmentation of densely packed nuclei during embryogenesis is extremely challenging, (see CTC).
While learned approaches generally perform better on a given dataset, they require expensive/time consuming manual annotations for each new sample, protocol and imaging modality. The is a paucity of labeled examples available for most biological samples and imaging modalities. And no pretrained super-nets capable of generalizing across these data while maintaining high accuracy.
Typical learned approaches begin by training a pixelwise classifier from image data where an expert has labeled (drawn on) a subset of the pixels corresponding to "foreground" (in our case, cell nuclei) and a second subset corresponding to "background" (non-nuclei). ^[
    This approach to labeling is independent of learning method.
    There are extensions that can improve segmentability or learnability, e.g. adding extra classes for explicit borders (DCAN,Stardist), instance labels for inidividual objects, labeling slices, etc.
    Also, one must choose which subset of the data to label. This is an especially complex choice for pixelwise classification given that it is possible in the limit to label (draw on) every single pixel in the dataset! (or at least those pixels belonging to nuclei).
    And there is the inherent problem of ambiguous data which means multiple independent labelings disagree.
    Centerpoint annotations require only a single point / nucleus, while dense pixelwise annotations require (in the limit) labeling every pixel within the volume of the nucleus.]

After classifying each pixel the segmentation must then decide how to group the nucleus-class pixels into nucleus instances, or equiv it must find borders between nucleus instances.
This step poses most of the difficulty in the segmentation problem and is usually unlearned, most often being done by simple threshold + connected components.

An alternative approach to instance segmentation taken by (mask r-cnn, Nico Sherf) is to first detect individual object instances, giving an approximate location for each object *before* moving on to the subsequent task of providing of a precise pixel-level description of object shape.
    Stardist is one of very few approaches which actually merges both the detection and shape-description tasks into one single step.
    This approach makes more sense for small densely packed objects, where boundary pixels are often shared between objects, and shapes are relatively simple.
We take this approach, following similar prescription to (Nico Sherf, 3D Gaussian kernel prediction forest) in that we treat object localization as an image-to-image task of regressing a mixture of (unnormalized) Gaussians discretized on the pixel grid with a single kernel located at each nucleus centerpoint.
    Here we use a standard U-net to learn this regression task.
    The gaussian mixture target requires only single-click centerpoint annotations, (easily made w Fiji's multipoint selector) which are relatively cheap and widely available in comparison with full, dense pixelwise labelings.
[diagram of process with closeup of nuclei, target and prediction.]
[second diagram showing full segmentation pipeline.]
The predicted nucleus centerpoints then serve as *seeds* in a subsequent watershed segmentation.
    The watershed potential can also be learned.
    In our c.elegans data we had access to a very small number (10) instance-labeled xy-slices from specific timepoints and z locations. We explored a variety of methods of learning the watershed potential from these data, including training a small "miniature" (10k params) CNN w/out downsampling layers to regress distance-to-boundary for each nucleus. This distance-to-boundary prediction serves directly as watershed potential.

## Detection and Segmentation results

We find that this approach gives state of the art results on a challenging C. elegans benchmark dataset (see ISBI Cell Segmentation Challenge 2018), as well as strong results on zebrafish retina, which promises to drastically reduce curation time.
[show pallette on c. elegans and zebrafish retina?]
We submitted a preliminary version of this model to the official ISBI CTC and ended up with segmentation (SEG) score of 0.52 which is state of the art on this dataset (see AC(4) on celltrackingchallenge.com), and a detection (DET) score of 0.89, which is 2nd best. Further refinements to this model have been made which significanly improve both detection and segmentation on our training set, but have yet to be evaluated by the challenge organizers.

# Future Plans

There a number of simple extensions to both the tracking and detection/segmentation models that would be interesting extensions.
Tracking:
    - Validate on benchmark datasets.
    - costs for springy divisions. The most obvious time when very close particles move in opposite directions! In stark contrast to the idea of flow.
    - unification with Active Graph Matching (AGM). AGM also defines a neighborhood relation on atals nuclei, and has a cost when the distance between nuclei in atlas differs from distance between them in the target. Is this cost exactly the same? Is AGM the same as ours, but with registration included?
Detection/Segmentation:
    - Add more information to the prediction target! More than just centerpoint info, but also include size/rough shape from bootstrapped inst seg, and prediction confidence from detection analysis.
    - 



PhD Thesis: Table of Contents
=============================

- Methods:
    + Tracking by Assignment with flow
        * Planarian worm
        * Drosophila melanogaster
        * (potentially) Zebrafish retina
    + Nuclei detection and segmentation from weak annotations
        * ISBI CTC: c. elegans
        * drosophila melanogaster
- Applications:
    + In house data:
        + Planarian: Detection, segmentation and tracking
        + Zebrafish Retina: Detection, segmentation and tracking
        + Drosophila: Detection, segmentation and tracking
        + 
    + Benchmark data:
- Segtools: Python library for Curation, Analysis and Visualization of bioimage data (3D images of fluorescent cells?)
    + Visualizing raw images
        + a simple stack viewer for python
        + integrated slice and volume rendering with clicking
        + volume rendering command line control
        + (fancy) max projections
    + Visualizing segmentations
        + pallettes. segmentation and detection overviews of very large datasets
        + RGB images w outlines and recolorings. Entropy coloring. Matching and score coloring.
        + scatterplot objects by feature with dot as 2d max projection over bounding box
        + coloring: ramsey coloring, instance edge coloring, etc.
    + Scores and metrics for segmentation and detection
        + python library for cell segmentation and detection metrics
    + Annotation
        + centerpoint annotation combined with cell removal
        + curating nuclear detections in 3D viewer

    + sorting and displaying network predictions ordered by loss score?
    + ilastik segmentation on Drosophila melanogaster


- Segmentation and tracking in dorosphila early embryogenesis.
    5k cell stage. stage 6 - stage 11?
- Segmentation and tracking in Planarians (latin name?) with novel velgrad costs for tracking w large displacements.
- Segmentation and detection in zebrafish retina w very high cell density where previously only semiautomated methods existed.
- Detection, segmentation and tracking in benchmark c. elegans dataset with SoA SEG scores.
- Instance segmentation of zebrafish cells in 2D during early embryonic stages.
- 





