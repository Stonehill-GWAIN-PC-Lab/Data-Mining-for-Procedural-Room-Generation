# Learning implementation of methods described in:
# M. Fisher, D. Ritchie, M. Savva, T. Funkhouser, and P. Hanrahan, “Example-based synthesis of 3D object arrangements,” ACM Transactions on Graphics (TOG), vol. 31, no. 6, p. 135, 2012.

from src.Methods.Kermani import frequencyFind, testInsert
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

class SceneGraph:
    '''SceneGraph is used to store the vertices and edges that we build the furniture graphs from (for graph mining)'''
    def __init__(self,vertices = [], edges = [], edgeCosts = []):
        self.vertices = vertices
        self.edges  = set(edges)
        self.edgeCosts = edgeCosts
    def addVertex(self,vertex):
        self.vertices.add(vertex)
    def addEdge(self,edge, cost):
        print('addEdge()',edge,cost)
        if edge not in self.edges:
            print("if")
            self.edges.add(edge)
            self.edgeCosts.append(cost)
        for vert in edge:
            if self.vertices[vert] not in self.vertices:
                self.vertices.add(vert)

def getObjects(frames):
    object_labels = []
    for frame in frames:
        for scene in frames[frame]:
            object_labels.extend([obj.label for obj in scene.annotation3D])
    return set(object_labels)

def FisherRelationships(frame,data,fi,prox_list,debug = False):
    '''Mines our different relationships by grouping objects using the similarity of their neighborhoods(Fisher et al. Section 4)'''
    #Process our data: list of FrameData objects
    global all_distances
    all_distances = []
    print(subprocessGraphRelations(data,0.05,twoObjectRelationshipProbability,minSpanningGraph))
    #trying to figure out the new mean and std based on values from all_distances
    print("Total number of distances found:"+str(len(all_distances)))
    print(all_distances)

def subprocessGraphRelations(scenes,percent_threshold,graph_func,graph_type):
    '''Modified function from Kermani.py that does not use Gbolt dependency'''
    min_gap = math.ceil(len(scenes) * percent_threshold) #If we dont' have our single object at the minimum threshold, we won't have any larger support
    good_objects = frequencyFind(scenes,min_gap)
    func = partial(graphRelationHelper,graph_type,graph_func,good_objects)
    if(len(scenes) > NUM_THREADS * 10): #Makes it worth our while. This number can be played with a bit, if desired
        p = Pool(NUM_THREADS)
        parsed_graphs = p.map(func,scenes) #create minSpanningGraph(objects) and apply twoObjectRelationshipProbabiltiy function to each relationship in the graph
        p.close()
        p.join()
        parsed_graphs = [p for p in parsed_graphs if p is not None]
    else:
        parsed_graphs = [p for p in map(func,scenes) if p is not None]
    label_dict = writeTestFile(parsed_graphs)#Writes out the file to read
    df = return_data_frame(label_dict) #Reads back in the file as a pandas dataframe
    print(df)
    createVersusDataFrame(label_dict)
    if df is None or 'verts' not in df.columns.values:
        return None
    return df

def createVersusDataFrame(ld):
    #the purpose is to get the cross relation with every two items
    #kitchen_dataframe
    print("in createVersusDataFrame")
    #getting all the objects
    object_list = list(ld)
    #find total relations and save them in a list in alphabetical order
    total_combos = []
    for i in range(0,len(object_list),1):
        for j in range(i,len(object_list),1):
            if(i!=j):
                temp = ""
                if object_list[i] < object_list[j]:
                    temp = str(object_list[i]) + " v " + str(object_list[j])
                else:
                    temp = str(object_list[j]) + " v " + str(object_list[i])
                total_combos.append(temp)
    total_combos.sort()
    list_of_zeros = []
    for x in total_combos:
        list_of_zeros.append(0)
    #making the dataframe with the correct rows and columns
    relation_dataframe = pd.DataFrame(list(zip(total_combos, list_of_zeros, list_of_zeros)),columns=["relation","neighborhood_avg","total_appearance"])
    #print(relation_dataframe)
    #findRow(relation_dataframe,"bottle","garbage_bin")
    return relation_dataframe

def findRow(relation_dataframe, object1, object2):
    #returns the row of the relation in the dataframe
    mask = []
    if object1 < object2:
        mask = relation_dataframe["relation"] == str(str(object1) + " v " + str(object2))
    else:
        mask = relation_dataframe["relation"] == str(str(object2) + " v " + str(object1))
    #print(mask)
    for i in range(0,len(relation_dataframe),1):
        if mask[i]==True:
            return(i)
    #returns -1 if error
    return -1

def inList(object_list, value):
    for x in object_list:
        if x == value:
            return True
    return False 

def minSpanningGraph(objects,c_func,value_array = None):
    '''Modified function from Kermani.py that creates a minSpanningGraph from our room objects.
    Modified to use Fisher Cost function instead of just proximity'''
    if len(objects) == 1:#Corner case (happens more than it should) a min-span tree from a graph of 1 is done
        return SceneGraph(objects)
    V = [o for o in objects]
    E = []
    #print ([str(v) for v in V])
    for i in range(len(V)):
        for j in range(i+1,len(V)): #These numbers link back to the objects themselves
            c = c_func(V[i],V[j],value_array)
            if c is not None:
                E.append(((i,j),c)) #This has our cost function
    E=sorted(E,key = lambda a:a[1])
    print("E:")
    print(E)
    T = SceneGraph(V) #Edges for minimum spanning tree
    T.edgeCosts=[] #reset edges
    print('pre function:')
    print(T.vertices)
    print(T.edges)
    print(T.edgeCosts)
    while len(E) > 0:
        edge = E.pop(0)
        print("Edge:")
        print(edge[0])
        print(edge[1])
        if testInsert(edge[0],T):
            T.addEdge(edge[0],edge[1]) #Add edge and the edges cost
    print("printing all edges and their cost in our graph")
    print("edges:",T.edges)
    print("edgecosts:",T.edgeCosts)
    return T #What we have here is an ijv sparse rep

def graphRelationHelper(graph_type,graph_func,good_objects,scene):
    '''Partial helper function to allow us to paralleize graph matching'''
    objs = [obj for obj in sorted(scene.annotation3D) if obj.label in good_objects]
    graph = graph_type(objs,graph_func)
    if len(graph.vertices) == 0 or len(graph.edges) == 0:
        return None
    res = ([(i,graph.vertices[i].label) for i in range(len(graph.vertices))],[(e[0],e[1],1) for e in graph.edges],[ec for ec in graph.edgeCosts])
    del graph
    return res

def writeTestFile(graphs):
    '''gbolt doesn't support string labels, so we have to give it a non-string label.
    Hence, this function is necessary'''
    labels = []
    for graph in graphs:
        verts = graph[0]
        labels.extend([vert[1] for vert in verts])
    labels = set(labels)
    label_dict = {}
    counter = 0
    #print("----------Here are the labels--------------")
    #print(labels)
    for label in labels:
        label_dict[label] = str(counter)
        counter +=1
    with open('input.txt','w') as fi: #We keep it as input.txt, although we can always make that part of a configuration file
        for i in range(len(graphs)):
            fi.write("t # "+str(i)+"\n")#Says what graph we are
            print('graph is:')
            print(graphs[i])
            print(graphs[i][0])
            print(graphs[i][1])
            print(graphs[i][2])
            verts = graphs[i][0]
            edges = graphs[i][1]
            edgeCosts = graphs[i][2]
            for vert in verts:
                fi.write("v "+str(vert[0])+" "+str(label_dict[vert[1]])+"\n")
            for e in range(len(edges)):
                edge = edges[e]
                edgeCost = edgeCosts[e]
                fi.write("e "+str(edge[0])+" "+str(edge[1])+" "+str(edgeCost)+"\n")
    return label_dict

def return_data_frame(label_dict):
    '''Modified function from Kermani.py with different paths'''
    r_label_dict = {}
    for key in label_dict:
        r_label_dict[label_dict[key]] = key
    print(r_label_dict)
    #paths = [f for f in os.listdir(".") if os.path.isfile(f) and "out.txt" in f]
    report_df = {}
    keys = ["support","verts","dict obj ref","edge 0","edge 1", "edge cost" "index"]#,"num_vert"
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
                        report_df["edge cost"].append("null")                        
                    elif data[0] == "v":
                        report_df["verts"].append(data[1])
                        report_df["dict obj ref"].append(r_label_dict[data[2]])
                        report_df["edge 0"].append("null")
                        report_df["edge 1"].append("null")    
                        report_df["edge cost"].append("null")      
                        report_df["index"].append(i)
                        report_df["support"].append("null")
                    elif data[0] == "e":
                        report_df["edge 0"].append(data[1])
                        report_df["edge 1"].append(data[2])  
                        report_df["edge cost"].append(data[3]) 
                        report_df["index"].append(i)
                        report_df["support"].append("null")
                        report_df["verts"].append("null")
                        report_df["dict obj ref"].append("null")
                    else:
                        pass #We skip the x
        print("printing df")
        print(len(report_df["supports"]))
        print(len(report_df["verts"]))
        print(len(report_df["dict obj ref"]))
        print(len(report_df["edge 0"]))
        print(len(report_df["edge 1"]))
        print(len(report_df["edge cost"]))
        print(len(report_df["index"]))

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
    
def twoObjectRelationshipProbability(obj1,obj2, value_array = None):
    #basically finding the zscore of the relationship
    std=15.0
    mean=90.0
    distance = np.sqrt(np.sum((obj1.centroid-obj2.centroid)**2)) #d=sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
    all_distances.append(distance)
    print("distance:",distance)
    #print(distance)
    z_score = (distance-mean)/std
    #print(z_score)
    print("zscore:",z_score)
    probability = round(st.norm.cdf(z_score), 5)
    #print(probability)
    print("prob:",probability)
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
    #runOccurenceModel()
