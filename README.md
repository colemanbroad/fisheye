Segmentation and tracking inside the zebrafish retina with deep nets and assignment models.

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

`img007` was `141029Ath5LAP2b-GFP_bactin-rasmKate2_view1_Subset_cell1-1.tif`

`img008` was `MAX_141029Ath5LAP2b-GFP_bactin-rasmKate2_view1_Subset_cell1.tif`

`img006_noconv` was

```python
raw = imread("/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/Mauricio/for_coleman/ath5lap2b_zflinedata_training/20_12_17_multiview H2B_RFP&BFP_Lap2bGFP_fish6.czi")
img006_noconv = raw[0,0,:,:,49:49+71,950:950+400,1147:1147+400,0]
img006_noconv = np.moveaxis(img006_noconv, 0,2)
```

`division.npz`
shape (11, 10, 2, 189, 216)
No labels. Just a single division cropped out of img006.

## labels

`labels006` the wavy hand-traced sparse labels for img006 created in ilastik.
Only exists for t=0.

`labels_lut.tif` 31 fully-annotated xy slices from various t and z, but almost all consecutive from t==0 and z=0.

`labels_iso_t0.npy` are the labels for time=0 on an isotropic resolution image. It's basically the same labels as labels_lut, but properly scaled in z.
(Actually... maybe not perfectly properly scaled in z. Should the labels stretch out with the pixels around them or stay const z-width?)

`img006+lab` normal img006 but with `labels006` labels in the first time point 
in the third channel.

`img006_labels_000.tif` just the 31 labeled xy slices from labels_lut, but all concatenated together in a row. 3 channel. labels are first channel.

`img006_borderlabels` boders around cells in first 10 xy slices in t=0 are drawn. Then each of these 2D cells has a centerpoint annotation. 

## anno

`anno000.npy`
shape (130,2)
2D annotations from 3D annotation tool.
labels cells as correct '1', underseg '2', etc... pairs with `lab000.npy`

`lab000.npy`
shape (355,400,400)
label image
automated segmentation of upscaled image.
The labels don't change smoothly as we travel through z. sometimes no change at all.

`point_anno.csv`
csv with header: Area,Mean,Min,Max,X,Y,Ch,Slice,Frame,Counter,Count
cell centerpoint annotations for img006, t=1 (2nd timepoint)
extract them with `lib.mkpoints`
```
points = lib.mkpoints()
cen = np.zeros((71,400,400))
cen[list(points.T)] = 1
cen2 = convolve(cen, np.ones((5,)*3)))
iss = Stack(perm(np.concatenate([img[1], cen2[:,A]],1), 'zcyx','zyxc'))
```
 
# ilp files

`img006.ilp` - obviously matches with the img006 input data...

