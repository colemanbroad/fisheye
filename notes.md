# images

`img1.tif` from  `wholeeye_input&segmentation.tif`
    - The channels are organised as follows:
    C1 - nuclei
    C2 - nuclei segmentation
    C3 - segmentation dividing cells
    C4 - nuclear envelope
    C5 - segmentation nuclear envelope 
    - has no time. just a single confocal stack
`img2.tif` from `xwingdatafish_forcoleman.tif`
    - blurry, untrackable Xwing data with x,y,z,t,c=1
`img3.tif`
    blurry 3D stack of nuclei. Can't resolve fine structure. Mosaic labeling. Very sparse. no time.
`img4.tif`
    relatively dense labeling. SPIM data. can't resolve fine structure. just nuclear marker. no time.
`img5.tif` from `20150611_injHSLAP_HSH2B_28-30hpfOn_5min_ROI1_REG-1.tif`
    blurry, sparse, with time, SPIM, nuclear stain.
`img6.tif` from 
    `20_12_17_multiview H2B_RFP&BFP_Lap2bGFP_fish6_Multiview_RIF_Subset.czi`
    XYCZT. nuclear envelope and nuclear volume markers. Confocal w 5min time res. 400x400 crop from x,y = 600,900 (or 600,950?) Saving to tif reshapes to TZCYX.

# ilp files

`img006.ilp` - obviously matches with the img006 input data...

# Which combination of markers is best?

We're trying to segment and track cells in the retina in 3D. The retina is densely packed with cells, and we would like to track as many as possible. We have two different potential markers:

1. H2B - RFP|GFP (nucleus)
2. Lap2b - GFP (nuclear envelope)

We can combine these markers in different ways. The images are recorded on a confocal microscope which can image three channels simultaneously, although currently we only use two.

We took a crop out of `fish6_ch1_40-80_800_800.tif` of channel 1. Just showing the nuclear envelope makes it hard for the eye to make out the shape of cells.

# Segmentation and Tracking with membrane AND nuclear markers

How do we use both channels together to jointly make tracking easier? Using both in a hand-designed model may be quite tricky, but in a learned pixelwise classification approach, it's not so hard at all!

[idea] Does ilastik use temporal features? Do convolutional kernels work:

1. Across channels or across time?
2. ~~Predict for a voxel using features computed for other channels (at same point in space) and/or nearby timepoints?~~ We are trying to predict a label at each voxel. This means we have to use features from each channel together, because they are associated with the same voxel and thus the same label. We could also use features calculated from neighboring voxels in space and time as input to the classifier. We don't do this for space, because we don't expect the features to change very much. But it might make sense for time. 

[gscholar] segmentation and tracking with nuclear and membrane channels

[@mosaliganti2012acme] Only shows images of membranes... Uses watershed segmentation.

[@khan2014quantitative] Has membrane and nuclear channel, seems to segment both. They must use joint info somehow. It's 3D cell segmentation, it looks like they use the nuclear channel to identify cell centers and number, then get full cell segmentations using the nuclei as seeds. They cite [@mosaliganti2012acme] and compare against it. 

[gscholar] acme automated cell morphology

[@stegmaier2016real] and [@de2015morphographx] are listed as similar.
[@santella2015wormguides]: here they talk about the cell *atlas*!

# ilastik 2: dense labeling confocal

After applying a simple pixelwise classifier from ilastik using just the nuclei channel we can look at the segmentability with simple thresholding of the prob map. The results at 0.95 `results/res001.png` and 0.99 `results/res002.png` show that nuclear volumes are difficult to separate, even at very high threshold values. We can either try using a segmentation model with more prior knowledge (watershed or d.t. watershed) or we can try to include the border information that we have from the nuclear envelope channel... Is this easy to do in ilastik?

NO. its not. and ilastik keeps freezing because my disk space is running low. I need a better solution...

Here's the way forward... I want to be able to bootstrap myself into really powerful models. I need to use hand-designed features to get there... Also, from the purely practical point of view, I need a better way of using the powerful hardware I have at my disposal. I would like to be able to parallelize the running of 
# ilastik 1: mosaic labeling spim

`/Users/colemanbroaddus/Desktop/Projects/fisheye/fisheye_mosaic_iso.ilp` is the project. The labeling is quite sparse. 
# ilastik 3: `img006.ilp`

How do we know when the classifier is good enough? When is it ready? How do we include info from both channels in the segmentation?

1. The nuc env channel is incorporated in the nuclear channel, so a simple segmentation of nuclear channel is already useful.
2. The classifier is good enough when the segmentations are good enough. But we should also keep track of classifier accuracy as we add more labels.

The nuclear channel in `results/res003.png` shows pmaps after the first round of labeling. The associated nuclear segmentation at th=0.75 is `results/res004.png`. This 3D stack has ≈2700 nuclei! This is too many. The situation is fixed if we first apply a 2px blur then we get the following with only 122 objects at th=0.75, see `results/res005.mov`. And in 3D it also looks pretty reasonable.

Now, after adding many more labels, we've moved on to the second phase of the classifier... This changes the cell size distribution. As a function of threshold we have the cell count (`results/res006.png`) and the cell size (log2) distribution (`results/res007.png`). There is a plateau of cells that emerges in the 4k-32k range as thresh → 0.95. Judging by eye I would expect cells to be in the 100x20x30 range ≈ 60k, which is within a factor of 2, and probably an overestimate. I bet the plateu that emerges with high threshold is valid. Also, the negative curavture on n_cells(thresh) hints at an impending asymptote. This is also closer to the threshold values that worked for fly and trib probability maps. I guess this is a property both of our labeling and of the objects images themselves which tend to have instance boundaries within semantically identical pixels. (Although this is what the nuclear envelope was supposed to be for!)

There is still lots of information we're not using, i.e. constraints on size, and priors on shape, continuity in x and y, and also z and t. Flow. It's a difficult task, and I'm not sure that it's possible to do accurately by eye. The jumps in time are long enough that it's not always easy...

Fist, let's explore the higher end of the threshold spectrum. We can see from `results/res008.png` and `results/res009.png` that the cell count plateaus between 0.95 and 0.966 before taking off again, and that the plateu in the size distribution near the desired size of 2**14 .. 2**16 is diminished as we increase the threshold. This suggests 0.95 - 0.96 as the optimal threshold region. Here is the probability maps and the segments at at 0.95 cutoff: `results/`. 

Now how do we know if/how many of these cells are correct? Should we try correcting time point 1 before moving on to the rest of the series?

---

I want to be able to quantify the accuracy of my ilastik classifiers. Categorical cross entropy, prececion, accuracy etc are all common metrics for classification problems which should be readily available. Also, it makes sense to replace ilastik with something custom that will let us transition smoothly between RFs, NNs and other learners, etc, and script them for easier analysis. We could even use the first couple layers of a pre-trained NN to do the classification for us!

Now we have to figure out how to train a net using just a very small number of annotations... This is a new problem.

# replace ilastik with tiny nets

let's see if we can use very small neural nets to replace ilastik even with a small number of training labels.

*How many labels do we currently have?*

12k labels! This is waaaay more than I thought!
And all these labels were made in just a single afternoon.

*How should we adjust the loss to reflect the missing/unlabeled regions?*

Just change the pixelwise weight in the loss function to be zero in places where there is no label. AND we could potentially include the predictions of a random forest, which gives us a probabilistic output across labels that we can use directly in the categorical cross-entropy loss, OR we could use the argmax of this distribution, but weight the pixel's term in the loss by some function of it's confidence score...

*How should we arrange the infrastructure for training and prediction?*

We want to make labels and view solutions on our machine, but we want training and testing to be done on the cluster. We can use the same setup we had available for the zebrafish membrane task, including saving high-level results to a pandas dataframe...

*Will this approach translate to the fly?*

Assuming that our networks actually do a better job at prediction than the RFs did... Then yes. This project will be our exploration into bootstrapping powerful models from weak models, 3D from 2D, etc.

*How does the paradigm of Active Learning apply in this context?*

[gscholar] active learning in bioimage analysis
[@sommer2011ilastik] clearly falls within the active learning framework.
[@kutsuna2012active] is also active learning, but without the power of deep neural networks
[gscholar] "active learning" in bioimage analysis
[@arganda2017trainable] also can be used in an active-learning way, although it lacks the immediate feedback of ilastik

https://en.wikipedia.org/wiki/Active_learning_(machine_learning) is a surprisingly useful summary of the topic.

> Active learning is a special case of semi-supervised machine learning in which a learning algorithm is able to interactively query the user (or some other information source) to obtain the desired outputs at new data points. There are situations in which unlabeled data is abundant but manually labeling is expensive. In such a scenario, learning algorithms can actively query the user/teacher for labels. This type of iterative supervised learning is called active learning. Since the learner chooses the examples, the number of examples to learn a concept can often be much lower than the number required in normal supervised learning.






# [idea] view image statistics. SUPER-histograms.

Histograms and line plots are a good way of visualizing the statistics of your image. We should be able to view line plots, histograms, autocorrelation, etc for lines going through the image and across time. I think the autocorrelation decay length tells recommends a sigma width for convolutions....

Don't work too much on ImStackShow, just save and open things in Fiji. It's easier now that you know the correct dim ordering! Also, maybe give dense instance annotations and learn a very small, but still deep representation on a downscaled image. The benefit of bootstrapping is you only have to fix mistakes as you scale up, you don't have to annotate tons of dumb pixels. This is the idea behind ilastik. Only curate areas of uncertainty. But eventually performance levels off, as a simple consequence of the underpowered RF models relative to Deep ones. 

# [idea] replace ilastik

Pros
- Custom kernels and features (eg features going across time)
- Custom feature vectors (eg using features from neighboring time points)
- Implementations of feature makers on GPU
- Closer integration with 3D viewer
- Better slice annotation by using Fiji
- Potential Deep Learning Backend
- Use cascaded RFs, Grad Boosted Trees, etc not available in ilastik
- Run prediction on cluster
- No need to constantly import/export
- Everything scriptable. No GUI necessary except for annotation.
- Use arbitrary brushes in Fiji!
- Potential for innovation
- Can do Cross Validation

Cons
- Many more moving pieces you have to organize
- Probably wont be able to do live mode prediction
- ilastik is getting GPU/3D viewer/deep learning Anyways!
- won't be able to use vigra RFs (which might be better than scikit?)

First step:
Recreate the features using slow scikit. What are the features?

3D Features:
- RAW
- Gaussian Smoothing
- Laplacian of Gaussian 
- Gaussian Gradient Magnitude
- Diff of Gaussians
- Structure Tensor Eigenvalues
- Hessian of Gaussian Eigenvalues





# [problem] working with large, remote czi images

It's a problem because of the time and space required to move 14 GB of data to my disk, rearrange it, crop it, store it, etc. You just crop a small region of the image that looks interesting.

The Crop tool is Broken in Fiji! When I try to crop an 5D image it copies the first frame to EVERY OTHER FRAME. WTF. It keeps X,Y, and C, but every other Z, and T is just a copy of the first one...