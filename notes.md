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
`img6_t0_zup.npy` is img6, but only the first timepoint and with the z axis cubically interpolated to scale the axis by a factor of ≈5.
`res042`, `res043` we see the cell membrane even during Meta/Ana - phase?
`res000`: movie across time of original `img6` data. single slice z=60.
    size in MB: gif: 27, tif:92, avi(jpeg): 7, png(one time point) 5

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

*How do we know when the classifier is good enough? When is it ready? How do we include info from both channels in the segmentation?*

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

*How accurate is our existing annotation?*

I can't tell. When I view the segmentations inside of the 3D viewer I get the sense that some of the larger nuclei are severley undersegmented, but coming up with a numerical score is difficult. We can use nuclear size distribution as a proxy for accuracy, but we don't a real gold standard annotation, or even a cell centerpoint annotation.

# [idea] Use cell centerpoint annotations as additional info

We've generated cell centerpoint annotation manually using the multi-point tool in Fiji. We should fix this tool! It would be easy to build a tool that segmented and removed the cell as soon as it's centerpoint is annotated.

If we use these the centerpoint annotations to seed a watershed segmentation we get reasonable looking segmentations! Here we've created a version of the original fluorescence image where boundaries between instances are identified (in x&y only?) and colored brightly. This allows us to see the segmentation on top of the original image clearly by scanning through the z stack. See
`results/res011.mov`. This images has 114 cells in accordance with our manual centerpoint labeling, And it appears to do a reasonable job of identifying instance boundaries, although certainly far from perfect. The cell size distribution (sorted plot of log2 size) is shown in `results/res012.png`. This provides an excellent validation of our earlier idea that the more correct segmentations are the ones that feature this plateau in the cell size distribution! 

But this result came at the cost of a lot of difficult manual centerpoint detection. And there are still mistakes! We're probably missing cells that escaped our notice during centerpoint detection, and many of the segmentations for cells are significantly off or missing. We can improve this segmentation in many ways.

1. Adjust parameters in the watershed to improve visual quality of result.
2. Use 3D view and nhl features to identify outliers and add new centerpoints.
3. Add new annotations to the pixelwise classifier data and retrain.
4. Including the membrane channel in the segmentation. At the membrane (nuclear envelope) signal is only used as input to the classifier, however if a pixel is 90% nuclei, it matters whether the remaining 10% is background vs nuclear envelope!

First, let's visualize the original two-channel image + the envelope prediction as a 3rd channel in color... This looks really great. It's clear that the pixelwise classifier does a good job of identifying the membrane class, and that the membrane class is a good marker for nuclear boundaries. Let's try to resegment but now with explicitly including the membrane probability map in the segmentation... We can immediately see from the size distribution that including the membrane probability in the watershed potential produces a better segmentation! NOPE. After further review it looks like simply NOT blurring the probability maps with a 1px blur before watershed results in the nicer cell size distribution and the difference between including the membrane channel and not is very small. See `results/res013.png`. Blue is size dist with normal watershed potential, no blur, mask<0.5. Orange is same thing but with adding the membrane prob map to the potential.

Despite the nice size distributions, we can still identify small numbers of cells which are either over or under segmented. We can see them in the size distribution as well as in the 3D view. To try and fix the few remaining mistakes I tried using the 3D viewer to annotate the largest nuclei (undersegmentations), followed by fitting the appropriate number of Gaussians in a mixture model. The Gaussian mixtures didn't fit well, and ended up cutting cells in half, see `results/res014.png`. Alternatively we can use the slice viewer to manually identify the correct cell centerpoints for undersegmented cells and add these points to our pointset.

[idea] Can we improve the cell centerpoint annotation workflow by segmenting and removing cells from the image continuously as we mark them? 

Ideally, we would click on a cell centerpoint in 3D, then that cell would be segmented from the underlying probability maps, highlighted in the 3D view and/or removed from the slice and 3D view to allow us to see the remaining cells. And there should be some way for us to control the shape / threshold level of the cell via a slider or keyboard before removing it from view.

We can also make mean or max projections of a non-overlapping (in x,y) subset of cells that are likely undersegmentations, highlight their borders and allow the user to re-assign centerpoints based on clicks. See example img `results/res015.png`.

[question] How will this manual curation approach fare against simply drawing the borders of cells for an instance segmentation?

1. The best way to mark a 2D slice for instance segmentation is to draw cell borders with e.g. value 1, then put marks in the connected component background regions with a different mark, eg 2. And we must be sure that the borders for a single cell are fully 8-connected s.t. cell internals are 4-connected. Drawing densely in 3D is just too time consuming to be worthwhile, but individual slice drawing is possible. This way, as long as the image is predominantly background and cell internals, you only have to mark a small fraction of the available pixels.
2. We want to compare this approach to one in which random individual pixels are marked for semantic segmentation training, followed by cell centerpoint marking. This makes it relatively easy to get a dense segmentation from labeling a small number of points. 

After adding centerpoint annotation to ImShowStack and annotating a few cell centerpoints we can replace our old centerpoint set with the new one and resegment. In addition to enabling cell centerpoint annotation I've connected the ImShowStack to the spimagine volume renderer s.t. clicking on cells in the stack will identify the segment and highlight the volume inside the volume renderer. It looks like this: `results/res016.mov`. This is a lot of fun, and really helps to give you a sense of your 3D segmentations. Through playing around with this system I've noticed the main remaining problem with the workflow is the poor semantic segmentation map. If I place a cell centerpoint inside of a cell, but outside of the watershed mask then our new cell is just a single voxel segmentation! We can fix this either by increasing the size of the watershed masked region, or by improving the pixelwise classifiers.

[idea] Turn our CARE networks into classifiers by encouraging pixel values to fall into a predetermined number of tight value intervals. Put this term in the loss function in addition to the terms which encourage restoration.

Let's try to improve the pixelwise classifier available by making use of neural networks. Let's see if the amount of training data available is sufficient for our purposes. If Not, then we can add hints from the output of the ilastik RFs...



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
# [problem] unet can't learn

After training our first unet on the small number (12k) individual labels from the ilastik dataset we see that the net is incapable of learning anything. All pixels are predicted as class 3. We can proceed in several ways:

1. add more training data
2. change the net architecture (make it smaller?)
3. change the loss (we already set the class weights for unknown pixels to zero!)
4. or include the previous RF predictions in the prediction target.

No. 4 seems like the most likely candidate. It would certainly give us enough data. It requires a change in the loss as well. How do we balance real labels and RF predictions? What loss do we use on the RF probabilities? L2? or the exact same binary crossentropy as before? We can use the exact same crossentropy as before! This is just the expected value of the CE for each of the different classes. Another possibility would be to use the Kullback-Leibler divergence to minimize the difference between the distributions.

Now I want to be able to see my training as it's happening. This requires Tensorboard. My validation loss doesn't seem to decrease. And my accuracy has also been constant, despite train loss going from 2.0 to 0.36... What CE loss should we expect for random guessing from three classes with balanced data?

I wonder if my lack of training examples causes the problem? I'm training only on the first time point (where all the pixelwise annotations are). I use 71 samples == z-slices and batch size = 1. And my loss gets stuck at 0.35 very quickly. If i reshape my data s.t. I have 16x100x100 instead of 400x400 sized images then this will help my loss explore and take gradient steps in new directions. Loss is stuck at 3.5. Increase batch size to 10. OK. I realize now I should probably normalize s.t. my pixel values aren't in the 1000's... Let's first try normalizing the means to 1... This definitely helps. The loss goes immediately down to the 0.2x range and the accuracy keeps rising up to about 80%... Now let's try reducing the importance of the non-hand-labeled pixels by simply multiplying all the probabilities across the three classes by a factor 0.10. This should work the same as a weight/mask without the requirement that we change the loss or the shape of ys_. But this changes the absolute value of the CE loss! Now it's 0.015 instead of 0.20. But if instead of multiplying by a constant factor we keep the sum of y_true normalized then we should be fine. But the accuracy still seems to be working. We're now up to 91.3% accuracy. See `weights/w002.h5`.

Now the problem is the probability maps just aren't that good looking. We need to compare them against ilastik through accuracy, CE, and subjective visuals, then we either need to make the Unet 3D or continue playing with the ratio of labels to RF predictions. If we normalize the ys weights we can keep the CE loss consistent in terms of absolute value, and we can reweight the labels s.t. the hand-labels count for the same total amount as RF predictions. This allows us to retrain and bring the loss down to around 0.1 with val_acc > 90%. See `training/t001/`. 

*how should we compare the quality of the RF vs the Unet?*

We can compare the predictions of both the the dense ground truth we have from manual nucleus labeling. Can I access the myers_spim_data from my scripts in my projects space? hmmmm.... 

spimdir="/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/"
greencarpet="/net/fileserver-nfs/stornext/snfs4/projects/green-carpet/"

Easy peasy.

# Densely labeled training data

Using the new, densely labeled training data we can re-attempt semantic segmentation... We're starting off from the previous best unet weights, `t001`.

The results are clear. The dense labeling is by far the best one. Even though we've only labeled a mere 10 z slices! The membrane fits nicely to the edge of the nuclei! Just like we demonstrated in our labelings... The only difference between the nets was the training data. NO difference in network parameters! See `results/res017.png` and `t002`. Is the mismatch between nuclear edge and envelope label the result of chromatic aberration? Are we correcting CA with dense net?

This is a large improvement over the sparse annotations and should improve the segmentations. Let's look at cell size distribution using watershed and potentials from the three different pixelwise classifiers... In `res018.png` we can see how the size distribution improves dramatically as we go from RF to NET and continues to improve if we replace the sparse labels with dense ones. This plot allows us to say a few nice things.

1. Nets are not better than RF solely as a result of being able to soak up more data. They are better even on the same data. [^1]. 
2. Nets have the ability to learn, even from relatively small annotation data. 12k individual pixels (30 mins spent labeling in ilastik) are sufficient to train a Unet and surpass ilastik performance.
3. Dense annotations are superior to sparse, given the same amount of time spent labeling. )[^2]

We should also do this comparison on the actual pixelwise prediction accuracy against the ground truth. What is the CE score for each? On each of the datasets? Let's test this. See `scripts/sc001.py`. [^3] 

| Model       | Sparse Accuracy | Sparse CE | Dense Accuracy | Dense CE |
| RF Sparse   | 1.00,1.00,1.00  | 0.0052972 | .335,.916,.807 | 0.22041  |
| Unet sparse | .812,.984,.957  | 0.0559251 | .375,.937,.775 | 0.21736  |
| Unet Dense  | .665,.961,.992  | 0.0785537 | .379,.894,.930 | 0.11159  |
| RF Dense?   |                 |           |                |          |

[table001]

These numbers are very interesting:
- The cross entropy score for the sparse RF predicting on it's own training data is *very* low, and the accuracy is perfect! This must be overfitting in the extreme. [^4]
- The dense unet has lower CE on the sparse data than on it's own training data! What does this mean about the quality of the sparse data?
- The dense accuracy tells part of the story. The hand-picked labels are roughly evenly balanced between classes, and the models trained on sparse data are biased away from the background, which takes up the majority of the real data. Dense labelings have no class bias! This gives the dense Unet a lower accuracy on nuclei, but a much higher accuracy on background! This is where the bulk of the improvement comes. Surprisingly, it also has a higher accuracy on membrane!

# ISONET - style 3D pixelwise classification

See `training/t003/t003.py`. The goal is to arrive at a better 3D pixelwise segmentation by applying the ISONET technique of restoring image quality in XY, YZ and XZ views independently, all from XY annotations. To make this work we need to 

1. blur the xy slices independently along the axis to be rescaled.
2. downsample along that axis by taking every nth slice (n=5?). The XY slices should at this point look qualitatively like XZ or YZ slices of the data.
3. upscale the slices back to their original size with linear (cubic?) interpolation
4. Train the network to restore the labels. (labels don't change).

When predicting, feed XZ and YZ slices (indpendently) into the network for prediction.
1. upscale Z by n=5 using linear (cubic?) interpolation
2. feed XZ and YZ slices into the network independently
3. Average together (pixelwise) the XZ and YZ stacks. *should we also average them with cubically upscaled versions of the XY predictions?* We could do the following:

1. Train the both an normal and an ISONET style network in the way previously mentioned.
2. Upscale the stack cubically in z. Feed in the XZ and YZ slices into the ISONET network and the normal XY slices (but now 5x as many slices because of the interpolation!) into the normal network.
3. Then at the end average all three stacks together!

Actually, the simplest thing would just be to not even train another network! Just try predicting on the upscaled XZ and YZ stacks with the normal net and average them at the end. Maybe it will suck who knows.... Importantly, 
*training* in that case doesn't even need to use down-uped stacks!

Let's look at these results with our current net... See `training/t004/t004.py`. You can see from `res019` the the XZ view of a net predicting on normal XY slices shows that the envelope region is missing from the Top and Bottom of cells (z is vertical axis in this view). Compare this with the normal XY view of the same data in `res020`. This is also obvious from the 3D view, see 
`res022` (xy view) and `res023` (side view). This is a consequence of the anisotropy inherent in the training data and in our labeling scheme. We seek to remedy this anisotropy by allowing the net to predict on XZ and YZ views of the same data, hopefully restoring the Tops and Bottoms of our nuclear envelope. In `res021` we can see the result of predicting on transposed, XZ slices (vertical/horizontal are z/x). The cell shape is mostly destroyed and the envelope is a gone in many regions, although we can find a small amount of membrane on the tops/bottoms of cells which is promising!

This tells us that we can't simply send 5x upscaled XZ data through the network trained on normal XY slices... We need to artificially blur and downsample in order to better teach the network how to restore XZ (and YZ) slices. Let's try training a network to do exactly this in `t005`. 

The network gets the loss down to .134, but it is not as good as in `t002` where we had 0.111 CE loss on the same data but without downsampling and upscaling the xs.

loss: 0.1343 - acc: 0.8304 - val_loss: 0.1582 - val_acc: 0.8259

*What sort of a price do we pay for the averaging?*

Or does the averaging actually help up improve the metrics in `table001`? 

new CE = 0.14754564981272042 on dense data if we take every 5th z slice for comparison with the dense labels. The class accuracy is (.208, .945, .890). So the accuracy for nuclei actually increases! 

Now we compare unet (`t002` on left) vs ISONET unet (`t005` on right) in terms of general image quality (see `sc003`): 

- `res024` shows XY slices. `t002` looks more cloudy and fluffy and has fewer meaningful gradients in nuclei probability. `t005` has clear orthogonal streaky artifacts, but it also shows us the top and bottom of cells in z including their red nuclear membranes, which elongates them in z (see bottom left). Some cell shapes look more complete in `t005`. Some membranes are more solid in `t002`.
- `res025` shows the XZ slice view of the same data. On the left we've simply used cubic interpolation to upscale Z for comparison. Again we see entire pieces of nuclei missing on the left (top right), and the same structure of artifacts. Notably, many of the cells in `t002` are missing the nuclear membrane on their top and bottom, which can be partially restored using ISO but the signal is weak. 
- `res026` is XZ view and shows (top left) a pair of nuclei with almost no envelope border, but that the ISO nevertheless manages to put a gap between them!

Let's look at cell size distributions... See `res029` and `sc004`. It's clear that the nets all do better. The net trained on simple, dense XY data with no borderweights has the cleanest looking prob maps, and doesn't have any under-sized cells, while the net with border weights does the best job preventing undersegmentation of large cells but is very simliar to iso_avgd.

*What if we don't use the seeds?*

We get nonsense... unless we apply a σ=2px blur to the nuclear channel and retry. Then we see the ISO and border-weighted models perform well! See 
`res030`.

*How can we improve?*

1. Add XZ YZ slice annotations.
2. Blur z before downsampling & upscaling
3. Combine stacks without simply pixelwise average. Maybe even blur the stacks or combine nearby pixels [^5]. 
4. We can add the pixel weight scheme that puts extra importance on membranes and boundaries.
5. Change our labeling technique to include the dense nuclear envelope on the tops and bottoms of cells.
6. Make the Unet 3D/4D by adding z and t ±1 to channels dimension.

Let's begin with #4 by retraining `t002` but with extra boundary weights. See 
`t006`. We have to pick the (arbitrary) expoential decay length! We chose 10. This is very important! OK the results look nasty. There is red envelope everywhere. We expect it to be over-represented, but it doesn't look nice. There are however some regions where the weighted network (right panel) does a better job of separating touching objects! See `res027` and `res028`. The loss is:

loss: 0.1530 - acc: 0.8396 - val_loss: 0.1746 - val_acc: 0.8298

We need perfect ground truth in this data so we can compare techniques accurately. Let's add annotations to XZ and YZ slices, then replace the ISONET technique with simple upscaling and prediction from the new annotations.

Let's make a new workflow `sc005` for cleaning up the hand-made labelings and network training data from annotated slices in 3D stacks.

Now we can train a network for XZ slices and we can include the new boundary weights technique. See `t007`. Final stats:

loss: 0.1565 - acc: 0.8272 - val_loss: 0.2531 - val_acc: 0.6874

The qualitative differences between this run and ISONET results from `t005` are very slight. After averaging XY,YZ,XZ views the only consistent qualitative difference I find is the z-width of membranes in `t007` (right panel) is a little more thin, see `res031`, and this may be the result of the border weights.

Note that this required 9 additional XZ slice annotations, which were used to train the new model. 

- We could combine this new data with the artificial XZ data and retrain?
- We could add YZ slice annotations
- We could settle for what we have and try working on the post-processing / segmentation? Guassian fitting? Level-set approach? Multicut? Centerpoints?
- Do we try to work on visualization?

Let's rerun `t007` with both artificial and real XZ data and see if we can improve. We can. The numbers are now:

loss: 0.1327 - acc: 0.8472 - val_loss: 0.1505 - val_acc: 0.8383

which are better than `t005` `t006` or `t007`(XZ only). And we can see the improvement in `res032` where the network now knows to put membrane in between touching nuclei and not just background as in `t005` and the image is generally cleaner. The size distribution using watershed w manual seed points is very similar to `t005` and `t006` (`res033`). [^6] But without manual seeds (see `res034`) we find roughly 25 more cells in `t007` than in `t006` and 50 more than in `t005`! Note that `t007` (the brown curve) has the clearest sharp upturn in size at the end of the distribution. These are clear undersegmentations; fusing of already *large* nuclei. 

- Can we make the data more clear by measuring surface area of cells and plotting it with volume simultaneously?
- can we get a sense for segmentation quality by looking at boder-highlighted cells?

After studying the segmentation borders from two_level_thresholding on (from left to right) the random forest, `t001` and `t007` it's amazing to see how similar they are in this view (`res035`, `res036`, `res037`). See `sc004`. 

*How do we know which is correct?*

We need dense Gt! Also, let's look at all the cells proposed by the various out in a grid...

# visualizing and correcting results

New script for interactive exploration. Interactive cell display and selection. `sc006`. Similar to `sc005`, but the purpose is analysis of a segmentation. Not curation of hand-drawn labels.

Now we can easily select nuclei from scatterplots and highlight them inside of the stack or 3D viewer, making them either *brighter* or *darker* (to reveal dim background). This works as a substitute for calculating pixel borders in the stack as well.

After removing the nuclei touching the image boundaries we get a nice separation into four distinct classes if we scatterplot (x = log2 size, y = avg brightness):

1. majority of well segmented nuclei of similar size: `res038`
2. undersegmented nuclei in a clump: `res039`
    - note that the long axis of both undersegmentation is in the z-direction, just as it was for drosophila! This tells us 
3. small, bright nuclei in metaphase, just after division: `res040`
4. pieces of other cells that were oversegmented: `res041`

And here's the update process in action: `res044`, `res045`.

*How do we fix the small number of over and under segmentation?*

- Continue to add labels, especially in XZ or YZ, until it looks good!
- Heuristic fixes during post processing
- Change the way we build watershed potential
- Predict distance to membrane?
- Change Unet (instance, etc)



# Analyze segmentation across time. Does our model generalize well?

See `sc007`. All images have been bicubic upscaled. Now we want to see if the segmenations look reasonable for all times and judge whether the objects are trackable.

By eye the probability maps look to be of consistent quality across time... 

Todo:
- plot size and brightness dist in time-adjacent pairs
- separate hypotheses into the minimal number of disjoint sets s.t. cells within the same set share no borders. (essentially the graph coloring problem). Show borders around segments only one set at a time so the borders are unambiguous. Slide through the volume and mark centers of cells with bad borders, or fix borders directly. Repeat with next set.
- view all cells in max projections flatly arranged in a grid. maybe with rotation?

---

The distribution of brightness vs size appears to shift.

# Can we improve the classifier by adding Z and T information?

Let's try this by adding nearby Z and T slices to the channels dimension in the model. Hopefully the model will figure out what to do with the new information!

ALSO we should be augmenting our data in simple ways.

Should we change the remaining structure of the network in any way? Do we need to add many more features to the next layer so we can absorb all the new information? 

Let's try retraining the XY model `t002` with these features:
- Augmentation
    + 90 rotation
    + Flips
    + Noise
    + Warp?
- Boder Weights
- Z & T slices added to channels
- ~~Mauricio's new annotations???~~

`t008` = `t006` + augmentation?

loss: 0.1178 - acc: 0.8712 - val_loss: 0.1607 - val_acc: 0.8164




# t009

- Add new classes for "unknown" and "dividing".
`res044` shows we're not able to learn very much yet. The loss plateaus at 0.226 :/
- There was a bug! In how I worked with channels and reshaping, etc.
- Fixed and now rerunning... loss is down at .2 by end of 2nd epoch. accuracy .79 by end of 2nd epoch. 35 sec / epoch. This is with only 3 classes though! at least things are back to working. moral of the story is be careful with your channel order!
- Re-ran using the GPU instance given by jupyter and all of a sudder it's down to 1sec / epoch... wtf.

# jupyter notebook

We've finally got the notebook running smoothly. I can train on the cluster GPU from home and I can even display and check my work as I go! Here's the new way of visualizing things `res045`.
First good model trained to:
loss: 0.0918 - acc: 0.8782 - val_loss: 0.1321 - val_acc: 0.8376

this is 3-class without augmentation.
with 5-classes (4 + 1 ignore class) we get pixelwise predictions for divisons.
This is very interesting, and despite the lack of labels we get reasonable predictions across many
slices. The net picks up on the coliflour texture of the chromatin in metaphase and the small and round features. We plot the 100 slices with the highest maximum value for divison in `res046` and 
`res047`.

NOTE: having a class you ignore in the loss doesn't mean it's ignored in the accuracy calculation afterwards! This means we have to write our own accuracy function that know which pixels to ignore.

Now with simple 8-fold flip and rotate we get...
loss: 0.0692 - acc: 0.8620 - val_loss: 0.0806 - val_acc: 0.8482
And most impressively, the top-rated divisons look very convincing! `res048` (top 64) `res049`.


# good ideas

For segmentation based on watershed we have a few options.
U-net
U-net w explicit boundary class
U-net w explicit boundary class and distance to nearest boundary!

Distance to boundary is common.
Should we use that for seeds in watershed?
But use the nuclei and boundary channels to define the watershed potential?




# Footnotes

[^1]: But we have to make sure that we're not comparing a unet vs an *oversaturated* RF, i.e. one whose marginal learning rate is low. We expect that the RF was not oversaturated, because it was still updating it's prediction as I added pixels in the viewer.

[^2]: Is this just because *more* is better than *fewer*? Is it just a fact about the efficiency of the dense annotation technique vs the sparse one? 

[^3]: Note that we're testing against the full set of labeled data, not just the validation data! So this includes the training data for the dense unet. The dense unet validation cross entropy loss was val_loss: 0.1264. The sparse labels used in training the other models ALSO come from the same timepoint and z slices, so the comparison is not unfair.

[^4]: I thought that RFs were supposed to be immune to overfitting?

[^5]: What about doing a max over channel 0 (nuc envelope) followed by averaging the remaining two channels and rescaling s.t. all three channels sum to one? This would give a preference to nuc envelope and help identify object boundaries.
[^6]: We need a stronger test to distinguish segmentation quality. We need unseeded tests and manual, pixelwise GT! 
