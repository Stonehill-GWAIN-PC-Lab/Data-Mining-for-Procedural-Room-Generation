# Learning implementation of methods described in:
# M. Fisher, D. Ritchie, M. Savva, T. Funkhouser, and P. Hanrahan, “Example-based synthesis of 3D object arrangements,” ACM Transactions on Graphics (TOG), vol. 31, no. 6, p. 135, 2012.

from src.Methods.Kermani import frequencyFind, graphRelationHelper, writeTestFile, minSpanningGraph
from .. import ObjectMetrics
import numpy as np
import pandas as pd
import math
import scipy.stats as st

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

#Assumes current dir = /project/ct-shml/SUNRGBD/Data-Mining-for-Procedural-Room-Generation
path_to_data = "../../../../projectnb/ct-shml/"
kitchen_dataframe = pd.DataFrame()

def getObjects(frames):
    object_labels = []
    for frame in frames:
        for scene in frames[frame]:
            object_labels.extend([obj.label for obj in scene.annotation3D])
    return set(object_labels)

def FisherRelationships(frame,data,fi,prox_list,debug = False):
    '''Mines our different relationships by grouping objects using the similarity of their neighborhoods(Fisher et al. Section 4)'''
    #Process our data: list of FrameData objects
    print(subprocessGraphRelations(data,0.05,ObjectMetrics.proximityCenter,minSpanningGraph))


def subprocessGraphRelations(scenes,percent_threshold,graph_func,graph_type):
    '''Modified function from Kermani.py that does not use Gbolt dependency'''
    min_gap = math.ceil(len(scenes) * percent_threshold) #If we dont' have our single object at the minimum threshold, we won't have any larger support
    good_objects = frequencyFind(scenes,min_gap)
    func = partial(graphRelationHelper,graph_type,graph_func,good_objects)
    if(len(scenes) > NUM_THREADS * 10): #Makes it worth our while. This number can be played with a bit, if desired
        p = Pool(NUM_THREADS)
        parsed_graphs = p.map(func,scenes)
        p.close()
        p.join()
        parsed_graphs = [p for p in parsed_graphs if p is not None]
    else:
        parsed_graphs = [p for p in map(func,scenes) if p is not None]
    label_dict = writeTestFile(parsed_graphs)#Writes out the file to read
    print("label dict")
    print (label_dict)
    df = return_data_frame(label_dict) #Reads back in the file as a pandas dataframe
    createVersusDataFrame(df)
    if df is None or 'verts' not in df.columns.values:
        print('empty df')
        return None
    print("good df")
    return df

def createVersusDataFrame(df):
    kitchen_dataframe = df
    ref_col = (df["dict obj ref"])
    object_list = []
    for value in ref_col:
        if value != "null":
            if !(inList(object_list, value)):
                    object_list.append(value)
    print(object_list)

def inList(object_list, value):
    for x in object_list:
        if x == value:
            return True
    return False 

def return_data_frame(label_dict):
    '''Modified function from Kermani.py with different paths'''
    r_label_dict = {}
    for key in label_dict:
        r_label_dict[label_dict[key]] = key
    print(r_label_dict)
    #paths = [f for f in os.listdir(".") if os.path.isfile(f) and "out.txt" in f]
    report_df = {}
    keys = ["support","verts","dict obj ref","edge 0","edge 1", "index"]#,"num_vert"
    for key in keys:
        report_df[key]  = []
    i='0'
    try: #We wrap this in a try block because it is always possible that gbolt ran out of memory and when it does we need to say that our gspan failed
        vert = []
        edge = []
        with open('input.txt','r') as fi:
            lines = fi.readlines()
            for line in lines:
                if len(line) == 1:
                    vert  = []
                    edge  = []
                else:
                    data = line.strip().split(" ")
                    print('data:',data)
                    if data[0] == "t":
                        report_df["index"].append(data[2])
                        i=data[2]
                        report_df["support"].append("null") #unused col
                        report_df["verts"].append("null")
                        report_df["dict obj ref"].append("null")
                        report_df["edge 0"].append("null")
                        report_df["edge 1"].append("null")                        
                    elif data[0] == "v":
                        report_df["verts"].append(data[1])
                        report_df["dict obj ref"].append(r_label_dict[data[2]])
                        report_df["edge 0"].append("null")
                        report_df["edge 1"].append("null")        
                        report_df["index"].append(i)
                        report_df["support"].append("null")
                    elif data[0] == "e":
                        report_df["edge 0"].append(data[1])
                        report_df["edge 1"].append(data[2])   
                        report_df["index"].append(i)
                        report_df["support"].append("null")
                        report_df["verts"].append("null")
                        report_df["dict obj ref"].append("null")
                    else:
                        pass #We skip the x
        df = pd.DataFrame(report_df)
    except:
        print("Exception")
        return None
    df.set_index('index',inplace = True)
    print(df)
    return df

def runOccurenceModel():
    #read mining.csv, create train and test dataset
    #create occurence model and train on train data
    #test against test data
    print('TODO')

def occurenceModel(scenes):
    '''Function takes an input scene and returns a probability for the static support hierarchy of the objects in the scene.
    The Occurence Model (Fisher et al. Section 6) describes what objects can be in synthesized scenes'''
    # use a Bayesian network B(S) to model the distribution over the set of objects that occur in a scene. 
    print('TODO')
    # given a fixed set of objects we use a simple parent probability table to define a function T (S) that gives the probability of the parent-child connections between objects in a scene.
    print('TODO')
    
def arrangementModel(object, scene):
    '''Function takes an object o positioned within a scene S and returns an unnormalized probability of its current placement and orientation.
    The Arrangement Model (Fisher et al. Section 7) describes where scene objects can be placed.'''
    print('TODO')
    
def twoObjectRelationshipProbability(object1_pos, object2_pos):
    #basically finding the zscore of the relationship
    std=15.0
    mean=90.0
    distance = abs(object1_pos - object2_pos)
    #print(distance)
    z_score = (distance-mean)/std
    #print(z_score)
    probability = round(st.norm.cdf(z_score), 5)
    #print(probability)
    if(probability>.5):
        return 1- ( (probability-.5) /.5)
    if(probability<.5):
        return probability/.5
    return 1

def sunRGBDDataMiningFisher(starting_location = None,data_cleanup = None, write_type = 'w'):
    '''max_amount controls the maximum number of rooms we look out, which helps us bound the problem
       starting_location tells us that there are rooms in the beginning that we can skip, most likely because we've already looked at them
       location_amount tells us to only look at a certain number of rooms, again so that we can run this in stages
       removed_rooms controls for our noisier rooms which may not give us good data. As an idea, if I was copying Kermani et al., I would remove all rooms except bedroom. Deep convo priors would be all rooms but living, bedroom, and office, etc.
       min_amount is another room remover. We determine a room threshold where we believe that we cannot get good data under the threshold
       write_type tells us if we are overwriting the file or appending.
    '''
    import src.Parse.SUNRGBD as SUNRGBD
    frames = defaultdict(list)
    a = path_to_data+"SUNRGBD/"
    direct = ["kv2/align_kv2/","kv2/kinect2data/","kv1/b3dodata/","kv1/NYUdata/",
              "realsense/lg/","realsense/sa/","realsense/sh/","realsense/shr/",
              "xtion/xtion_align_data"] #"xtion/sun3ddata", has weird paths

    direct = [a+d for d in direct] #Append the relative
    for d in direct:
        paths = [f for f in os.listdir(d) if not os.path.isfile(os.path.join(d,f))]
        frames = SUNRGBD.getFrames(d,paths,frames)
    #This combines our similar rooms from the pattern analysis
    print ("Total Objects",len(getObjects(frames)))
    if data_cleanup is not None:
        keys = data_cleanup.cleanupRooms(frames)
        data_cleanup.cleanupObjects(frames)
    else:
        keys = [k for k in frames]
    from functools import reduce
    total_frames = reduce(lambda x,y: x+y,[len(frames[frame]) for frame in keys])
    print ("Finished sorting the file paths:",total_frames)
    with open(r'/project/ct-shml/Jimmy-SUNRGBD/outputs/mining.csv',write_type) as fi:
        #TODO: Make every connection discovered by subgraph pattern mining
        for frame in keys:
            data = frames[frame]
            fi.write(frame+','+str(len(data))+'\n')
            if(frame=="kitchen"): #We are only learning on kitchens
                print('kitchen example')
                FisherRelationships(frame,data,fi,[],True)
                del data #Clean up our messes
    print("Finished running file")

if __name__ == "__main__":
    import os
    same_rooms   = {"idk":"corridor","recreation_room":"rest_space"}
    same_objects = {"fridge":"refridgerator","bathroomvanity":"bathroom_vanity","toyhouse":"toy_house","bookshelf":"book_shelf","tissuebox":"tissue_box"}
    removed_rooms = ["Dining_Room_Garage_Gym","Dining_Room_Kitchen_Office_Garage","Room","Living_Room_Dining_Room_Kitchen_Garage"]
    support = (20,1000)
    sunRGBDDataMiningFisher()
    #print(twoObjectRelationshipProbability(90, 165))
    #print(twoObjectRelationshipProbability(90, 15))
    #runOccurenceModel()
