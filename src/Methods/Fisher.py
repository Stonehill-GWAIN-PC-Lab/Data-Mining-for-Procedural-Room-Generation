# Learning implementation of methods described in:
# M. Fisher, D. Ritchie, M. Savva, T. Funkhouser, and P. Hanrahan, “Example-based synthesis of 3D object arrangements,” ACM Transactions on Graphics (TOG), vol. 31, no. 6, p. 135, 2012.

from .. import ObjectMetrics
import numpy as np
import pandas as pd
import math

import time
from collections import Counter,defaultdict
from  .SceneSuggest import parentScene,supportSurfaceCounts,size_sort
import subprocess
import os
import re

from multiprocessing import Pool
from functools import partial,reduce
import os
NUM_THREADS = 8

path_to_data = "../../../../projectnb/ct-shml/"

def getObjects(frames):
    object_labels = []
    for frame in frames:
        for scene in frames[frame]:
            object_labels.extend([obj.label for obj in scene.annotation3D])
    return set(object_labels)

# probabilistic model for scenes

# occurnence model
    # O(S) takes a scene S as input and returns a probability for the static sup- port hierarchy of the objects in the scene
    # (describes what objects can be in synthesized scenes)
def occurenceModel(scene):
    # use a Bayesian network B(S) to model the distribution over the set of objects that occur in a scene. 
    print('TODO')

    # given a fixed set of objects we use a simple parent probability table to define a function T (S) that gives the probability of the parent-child connections between objects in a scene.

    
#arrangement model
    # A(o, S ) is a function which takes an object o positioned within a scene S and returns an unnormalized probability of its current placement and orientation.
    # (describes where those objects can be placed)

def sunRGBDDataMiningFisher():
    '''max_amount controls the maximum number of rooms we look out, which helps us bound the problem
       starting_location tells us that there are rooms in the beginning that we can skip, most likely because we've already looked at them
       location_amount tells us to only look at a certain number of rooms, again so that we can run this in stages
       removed_rooms controls for our noisier rooms which may not give us good data. As an idea, if I was copying Kermani et al., I would remove all rooms except bedroom. Deep convo priors would be all rooms but living, bedroom, and office, etc.
       min_amount is another room remover. We determine a room threshold where we believe that we cannot get good data under the threshold
       write_type tells us if we are overwriting the file or appending.
    '''
    import src.Parse.SUNRGBD as SUNRGBD
    import src.Parse.NYUSupport as NYUSupport

    NYUSupport.mineNYU2Data(path_to_data,"support_mining.csv")
    frames = defaultdict(list)
    a = path_to_data+"sunrgbd/"
    direct = ["kv2/align_kv2/","kv2/kinect2data/","kv1/b3dodata/","kv1/NYUdata/",
              "realsense/lg/","realsense/sa/","realsense/sh/","realsense/shr/",
              "xtion/xtion_align_data"] #"xtion/sun3ddata", has weird paths

    direct = [a+d for d in direct] #Append the relative
    for d in direct:
        paths = [f for f in os.listdir(d) if not os.path.isfile(os.path.join(d,f))]
        frames = SUNRGBD.getFrames(d,paths,frames)
    #This combines our similar rooms from the pattern analysis
    print ("Total Objects",len(getObjects(frames)))

if __name__ == "__main__":
    import os
    same_rooms   = {"idk":"corridor","recreation_room":"rest_space"}
    same_objects = {"fridge":"refridgerator","bathroomvanity":"bathroom_vanity","toyhouse":"toy_house","bookshelf":"book_shelf","tissuebox":"tissue_box"}
    removed_rooms = ["Dining_Room_Garage_Gym","Dining_Room_Kitchen_Office_Garage","Room","Living_Room_Dining_Room_Kitchen_Garage"]
    support = (20,1000)
    sunRGBDDataMiningFisher()