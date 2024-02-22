Part of a data set acquired in MST as described in a [previous study](https://onlinelibrary.wiley.com/doi/10.1111/ane.13518).
This data set was used and transferred in the context of the Brain-SHIFT project funded by PIHC 2022.

This data set is a pseudo-BIDS, and cannot follow exactly BIDS specs as CT and annotations were not yet included
in the latest BIDS version (1.8.0).

More precisely 2 folders include NifTi folders:
- `ct` includes the original images as acquired in MST. Three different series were extracted:
  - `slicethickness-small_registered-false` is the original image (noisy but with small z resolution),
  - `slicethickness-large_registered-false` is a processed version in which slices were averaged to improve the SNR,
  - `slicethickness-large_registered-true` is a processed version spatially registered to a standard space in which slices were averaged to improve the SNR,
- `annotation` includes the mask of the hematoma (1), right (2) and left (3) ventricles manually 
annotated by 2 Technical medicine students using XNAT OHIF platform. The description allows to find to which image
the annotation mask can be superimposed. The annotation was performed on only one space and eventually interpolated
to another one.