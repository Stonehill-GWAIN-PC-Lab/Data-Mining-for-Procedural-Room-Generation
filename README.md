# Data-Mining-for-Procedural-Room-Generation
This repository holds several different methods to data mine rooms stored as json files. This project was part of a post-doctoral project on generating 3-D virtual environments. At its core, it takes my interpretation of several publications to determine relationships between objects in an environment, and provides back those patterns. This project attempts to be data-set agnostic.

## Prerequisites
Requires Python 3.x
Requires numpy,scipy,h5py,opencv,python json 

## Data sets supported
SUNRGBD
SUNCG (Legacy, no longer available)
Matterport3D
Unreal Engine Json Generator
##Methods supported
At this time, I've fully build out the methods for:

Z. S. Kermani, Z. Liao, P. Tan, and H. Zhang, “Learning 3D Scene Synthesis from Annotated RGB‐D Images,” Computer Graphics Forum, vol. 35, no. 5, pp. 197–206, 2016, doi: 10.1111/cgf.12976.


I've also added some of the metrics described in:
M. Fisher, D. Ritchie, M. Savva, T. Funkhouser, and P. Hanrahan, “Example-based synthesis of 3D object arrangements,” ACM Transactions on Graphics (TOG), vol. 31, no. 6, p. 135, 2012.

Q. Fu, X. Chen, X. Wang, S. Wen, B. Zhou, and F. Hongbo, “Adaptive synthesis of indoor scenes via activity-associated object relation graphs,” ACM Transactions on Graphics, vol. 36, no. 6, p. 13, 2017.


For those metrics, I have not fully written out those methods on this repository.
##Citations
If you find this work useful, then you should cite the following papers:

Z. S. Kermani, Z. Liao, P. Tan, and H. Zhang, “Learning 3D Scene Synthesis from Annotated RGB‐D Images,” Computer Graphics Forum, vol. 35, no. 5, pp. 197–206, 2016, doi: 10.1111/cgf.12976.

A. X. Chang, M. Eric, M. Savva, and C. D. Manning, “SceneSeer: 3D Scene Design with Natural Language,” arXiv preprint arXiv:1703.00050, p. 10, 2017.

M. Fisher, D. Ritchie, M. Savva, T. Funkhouser, and P. Hanrahan, “Example-based synthesis of 3D object arrangements,” ACM Transactions on Graphics (TOG), vol. 31, no. 6, p. 135, 2012.

Q. Fu, X. Chen, X. Wang, S. Wen, B. Zhou, and F. Hongbo, “Adaptive synthesis of indoor scenes via activity-associated object relation graphs,” ACM Transactions on Graphics, vol. 36, no. 6, p. 13, 2017.

#### Dataset citations
S. Song, S. P. Lichtenberg, and J. Xiao, “Sun rgb-d: A rgb-d scene understanding benchmark suite,” in the IEEE conference on computer vision and pattern recognition, 2015, pp. 567–576.

N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, “Indoor segmentation and support inference from rgbd images,” Computer Vision–ECCV 2012, pp. 746–760, 2012.

J. Xiao, A. Owens, and A. Torralba, “Sun3d: A database of big spaces reconstructed using sfm and object labels,” 2013, pp. 1625–1632.

A. Chang et al., “Matterport3D: Learning from RGB-D Data in Indoor Environments,” International Conference on 3D Vision (3DV), 2017.

A. Janoch et al., “A category-level 3d object dataset: Putting the kinect to work,” in Consumer depth cameras for computer vision, Springer, 2013, pp. 141–165.

J. T. Balint and R. Bidarra, “Mined Object and Relational Data for Sets of Locations.” 4TU.Centre for Research Data. Dataset, Feb. 13, 2019. [Online]. Available: https://doi.org/10.4121/uuid:1fbfd4a0-1b7f-4dec-8097-617fea87cde5

You can also cite the repository directory as a library.
## TODO:
There is a plethora of techniques to mine locations. It would be great to add some of those to this repository.