#General Includes
import numpy as np
import pandas as pd
import math
import time
import os
import json

from .. import ObjectMetrics#For our metrics, obviously
from collections import Counter,defaultdict

#For speeding up the computation
from multiprocessing import Pool,Manager
from functools import partial,reduce
NUM_THREADS = 8




class Connections:
    '''This class controls how our scene connects to itself'''
    def __init__(self):
        self.__children    = [] #Contains the children as a list
        self.__parents     = [] #Contains the parents as a list
        self.__connections = [] #A mapping of parent index to child index

    def addConnection(self,parent,child):
        parent_idx = -1
        child_idx  = -1
        try:
            parent_idx = self.__parents.index(parent)
        except:
            parent_idx = len(self.__parents)
            self.__parents.append(parent)
            self.__connections.append([])
        try:
            child_idx = self.__children.index(child)
        except:
            child_idx = len(self.__children)
            self.__children.append(child)
        if child_idx not in self.__connections[parent_idx]:
            self.__connections[parent_idx].append(child_idx)

    #Our tests to see if the parent and child are in the list
    def hasParent(self,parent):
        return parent in self.__parents
    def hasChild(self,child):
        return child in self.__children
    #Finally, let's see if the connection exists
    def hasConnection(self,parent,child):
        parent_idx = -1
        child_idx  = -1
        try:
            parent_idx = self.__parents.index(parent)
            child_idx  = self.__children.index(child)
            return child_idx in self.__connections[parent_idx]
        except:
            print (parent_idx,child_idx)
        return False #If we don't have it, then we say so
    def getChildren(self):
        return self.__children
    def getParents(self):
        return self.__parents
    #We also count what we are working with
    def numParents(self):
        return len(self.__parents)
    def numChildren(self):
        return len(self.__children)
    def numConnections(self):
        if len(self.__connections) == 0:
            return 0
        return reduce(lambda x,y: x+y,map(lambda z:len(z),self.__connections))
            

def size_sort(x): #For parent child relationships, we use the larger one as the parent, and therefore, our parent should come first
    return -np.prod(x.size) #Negative so we sort by larger size


def TopSerfConnects(connections,scene,eps = 0.01):
    '''Given a first pass that says that a parent-child connection exists, we then figure out which way it could be.'''
    #Unlike the original paper, we are only looking at top and side connections, not top, bottom, or which side.
    v_con = {}
    h_con = {}
    names = [o.label for o in scene.annotation3D]
    n_counts = {}
    count = 0
    for n in set(names):
        n_counts[n] = names.count(n)
    for obj_c in scene.annotation3D:
        if connections.hasChild(obj_c.label):#This can be a child object
            for obj_p in scene.annotation3D:
                if(obj_c != obj_p):
                    if connections.hasParent(obj_p.label): #This can be a parent
                        if connections.hasConnection(obj_p.label,obj_c.label):
                            count +=1
                            #Now, we determine which connection it is (if any)
                            #If we've seen this multiple times, then we make a note of it
                            #However, we cannot have more connections than children
                            if ObjectMetrics.supportMiningStatic(obj_p,obj_c,[0,2],1,eps = eps):
                                key = (obj_c.label,obj_p.label)
                                if key in v_con and v_con[key] < n_counts[obj_c.label] and v_con[key] < n_counts[obj_p.label]:
                                    v_con[key] +=1
                                else:
                                    v_con[key] = 1
                            elif ObjectMetrics.supportMiningStatic(obj_p,obj_c,[1,2],0,eps = eps) or ObjectMetrics.supportMiningStatic(obj_p,obj_c,[0,1],2,eps = eps):
                                key = (obj_c.label,obj_p.label)
                                if key in h_con and h_con[key] < n_counts[obj_c.label] and h_con[key] < n_counts[obj_p.label]:
                                    h_con[key] += 1
                                else:
                                    h_con[key] = 1
    #print (v_con,h_con)
    #if count > 0:
    #    print("We looked at",count,"possible connections")
    return (v_con,h_con)

def getNumGoodConnections(prob_dict_list):
    '''Debugging function to determine how many scenes ended up
    having a proper connection'''
    return reduce(lambda x,y: x+y,map(lambda z: len(z),prob_dict_list))
    
    
def getProb(probs,counts, eps = 0.00005,debug = True):
    '''This is the single step probability counts for a given support'''
    support = []
    totals = {}
    for prob in probs:
        for key in prob:
            if key in totals:
                totals[key] +=1
            else:
                totals[key] = 1
    for key in totals:
        prob = totals[key] / counts[key[0]]
        if prob > eps:
            support.append([key[1],key[0],prob])
    return support

def getCocurProb(probs,counts,eps = 0.00005,debug = False,raw_counts = False):
    '''This function determines how many support connections we have and what their probabilities are'''
    maximums = {}
    support  = []
    for prob in probs:
        for key in prob:
            if key in maximums:
                new_arr = np.ones(prob[key])
                if new_arr.shape > maximums[key].shape:
                    maximums[key].resize(new_arr.shape)
                elif new_arr.shape < maximums[key].shape:
                    new_arr.resize(maximums[key].shape)
                maximums[key] += new_arr
            else:
                maximums[key] = np.ones(prob[key])
    #Now, we go through and figure out seperate counts for each
    for key in maximums:
        #scene suggest sums the counts for all k >1, so we need to do the same
        tots_prob = np.sum(maximums[key])#tots mcgoats
        for i in range(len(maximums[key])):
            #The first one is special, we treat that as a pure support surface
            if i == 0:
                if not raw_counts:
                    prob = maximums[key][i] / counts[key[0]]
                else:
                    prob = maximums[key][i]
                if debug:
                    print(key,prob)
                if prob > eps:
                    support.append([key[1],key[0],prob])
            else:
                #Otherwise, we are more concerned with the total counts
                if not raw_counts:
                    prob = np.sum(maximums[key][:i+1]) / tots_prob #While normally it is i < k, i'm going up to the one before because my zeroth represents all the non-k ones
                else:
                    prob = np.sum(maximums[key][:i+1])
                if prob > eps:
                    new_name = key[0] + "_" + str(i)
                    support.append([key[1],new_name,prob])
    return support
            
            
def supportSurfaceCounts(scenes,parents,threshold = 0.1):
    '''Given a known set of parents and a set of scenes, we construct a count prior
    for a specific set of scenes'''
    #Get the good items
    debug = False
    #For counts, our threshold becomes our eps
    eps = threshold*len(scenes)
    if parents.numParents() == 0: #No parents
        return [[],[],[]]#pass back our lists as empty
    #And the counts of all those items in each scene
    p = Pool(NUM_THREADS)
    objCounts = partial(ObjectMetrics.frequencyGoodObjects,set(parents.getChildren() + parents.getParents()))
    counts = p.map(objCounts,scenes)
    p.close()
    p.join()
    #Get the counts of all objects we are considering, this becomes our denominator
    sup_count = {}
    for scene in counts:
        for key in scene:
            if key in sup_count:
                sup_count[key] += 1
            else:
                sup_count[key]  = 1
    #Now, we get our surface connections
    connections = partial(TopSerfConnects,parents)
    p = Pool(NUM_THREADS)
    counts = p.map(connections,scenes)
    p.close()
    p.join()
    #And from that, create counts
    v_prob = [i[0] for i in counts]
    h_prob = [i[1] for i in counts]
    if debug:
        print("Found",getNumGoodConnections(v_prob),"vertical connections")
        print("Found",getNumGoodConnections(h_prob),"horizontal connections")
    supports = [getCocurProb(v_prob,sup_count,eps,raw_counts = True),getCocurProb(h_prob,sup_count,eps,raw_counts = True)]
    #Finally, get the frequency
    good_items = set([i[0] for i in supports[0]]+[i[0] for i in supports[1]] +[i[1] for i in supports[0]]+[i[1] for i in supports[1]])
    probs = [[i,sup_count[i]] for i in good_items]
    return (supports,probs)

def supportSurfacePriors(scenes,parents,debug = False):
    '''Given a known set of parents and a set of scenes, we construct a count prior
    for a specific set of scenes'''
    #Get the good items
    if parents.numParents() == 0: #No parents
        return [[],[],[]]#pass back our lists as empty
    #And the counts of all those items in each scene
    p = Pool(NUM_THREADS)
    objCounts = partial(ObjectMetrics.frequencyGoodObjects,set(parents.getChildren() + parents.getParents()))
    counts = p.map(objCounts,scenes)
    p.close()
    p.join()
    #Get the counts of all objects we are considering, this becomes our denominator
    sup_count = {}
    for scene in counts:
        for key in scene:
            if key in sup_count:
                sup_count[key] += 1
            else:
                sup_count[key]  = 1
    #Now, we get our surface connections
    connections = partial(TopSerfConnects,parents)
    p = Pool(NUM_THREADS)
    counts = p.map(connections,scenes)
    p.close()
    p.join()
    #And from that, create counts
    v_prob = [i[0] for i in counts]
    h_prob = [i[1] for i in counts]
    if debug:
        print("Found",getNumGoodConnections(v_prob),"vertical connections")
        print("Found",getNumGoodConnections(h_prob),"horizontal connections")
    supports = [getCocurProb(v_prob,sup_count),getCocurProb(h_prob,sup_count)]
    #Finally, get the frequency
    good_items = set([i[0] for i in supports[0]]+[i[0] for i in supports[1]] +[i[1] for i in supports[0]]+[i[1] for i in supports[1]])
    probs = [[i,sup_count[i]] for i in good_items]
    return supports+[probs]

def determineParents(scene,eps = 0.01):
    parents = {}
    #We sort by size because saava et al. mentions bounding box volume
    #and this implimentation has a precolunation towards objects sooner in the list
    #objs = sorted(scene.annotation3D,key = size_sort)
    #print([i.label for i in scene.annotation3D])
    for i in range(len(scene.annotation3D)):
        obj1 = scene.annotation3D[i]
        for j in range(i+1,len(scene.annotation3D)):
            obj2 = scene.annotation3D[j]
            sup = ObjectMetrics.supportMiningBoundingBox(obj1,obj2,eps,debug = False)#Favors size objects
            if sup is not None:
                #supportMiningBoundingBox(obj1,obj2,debug = True)
                if sup:
                    if obj1.label in parents:
                        parents[obj1.label].append(obj2.label)
                    else:
                        parents[obj1.label]=[obj2.label]
                else:
                    if obj2.label in parents:
                        parents[obj2.label].append(obj1.label)
                    else:
                        parents[obj2.label]=[obj1.label]
    for p in parents:
        parents[p] = list(set(parents[p]))
    return parents

def getEdges(all_edges,key_list):
    parents = []
    for edge in key_list:
        if (edge[1],edge[0]) in all_edges:
            if all_edges[edge] > all_edges[(edge[1],edge[0])]:
                parents.append(edge)

        else:
            parents.append(edge)
    return parents
#And our general scene processing function
def parentScene(scenes):
    p = Pool(NUM_THREADS)
    #parents = []
    #for scene in scenes:
    #    parents.append(determineParents(scene))
    parents = p.map(determineParents,scenes) #Gets all of our possible parents
    p.close()
    p.join()
    #Reduce to single edges
    edges = {}
    for parent in parents:
        for key in parent:
            for val in parent[key]:
                if (key,val) in edges:
                    edges[(key,val)] +=1
                else:
                    edges[(key,val)] = 1
    #Now, for each edge, determine if the converse edge exists. If it does, determine which is greater to determine the parent
    #Speed up later
    del parents
    edgeFunc = partial(getEdges,edges)
    #Break up the edges into a number of threads (should do this with process, but then I'd have to scroll to the top)
    key_names = list(edges.keys())
    if(len(key_names) < 10*NUM_THREADS): #We really need to make it worth it
         parents = list(map(edgeFunc,key_names))
    else:
        p = Pool(NUM_THREADS)
        breakup = math.ceil(len(key_names)/NUM_THREADS)
        key_names = [key_names[i:i+breakup] for i in range(0,len(key_names),breakup)]
        parents = p.map(edgeFunc,key_names)
        p.close()
        p.join()
        parents = [item for sublist in parents for item in sublist]
    connection = Connections()
    for edge in parents:
        connection.addConnection(edge[0],edge[1])
    return connection

def getObjectsForJson(scene):
    objs = []
    for i in range(len(scene.annotation3D)):
        obj = {}
        obj["id"] = i
        obj["name"] = scene.annotation3D[i].label
        obj["frequency"] = 1.0
        obj["Leaves"] = []

        objs.append(obj)
    return objs

def parentRelationships(scene):
    '''Calculates a parent relationship as a json list of relationships given
    the rote positions of the objects in the scene'''
    relationship_list = []
   
##    relationship_list = []
    count = 0
    connection = Connections()
    for i in range(len(scene.annotation3D)):
        obj1 = scene.annotation3D[i]
        for j in range(i+1,len(scene.annotation3D)):
            obj2 = scene.annotation3D[j]
            sup = determineParents(scene,100)#Favors size objects
            if sup is not None:
                if sup:
                    connection.addConnection(obj1.label,obj2.label)
                else:
                    connection.addConnection(obj2.label,obj1.label)
##            if sup is not None:
##                rule = {}
##                rule["id"] = count
##                count += 1
##                if sup:
##                    rule["source"] = i
##                    rule["target"] = j
##                else:
##                    rule["source"] = j
##                    rule["target"] = i
##                rule["factor"] = "support"
##                relationship_list.append(rule)
    information = TopSerfConnects(connection,scene,eps = 100)
    #V is info[0], H is info[1]
    count = 0
    for i in range(len(information)):
        info = information[i]
        for key in info:
            rule = {}
            rule["id"] = count
            count +=1
            rule["source"] = scene.annotation3D.index(scene.getObject(key[0]))
            rule["target"] = scene.annotation3D.index(scene.getObject(key[1]))
            if i == 0:
                rule["type"] = "Vertical-Support"
            else:
                rule["type"] = "Support"
            relationship_list.append(rule)
    #print(relationship_list)
    return relationship_list
            
def unrealJsonDataMining():
    #Mines from positional information from our static mesh saver
    import UnrealJson
    frames = defaultdict(list)
    a = path_to_data + "Independent Studies/Generated Data/handmade"
    paths = [f for f in os.listdir(a) if os.path.isfile(os.path.join(a,f))]
    frames = UnrealJson.getFrames(a,paths,frames)
    #print(frames)
    total_frames = reduce(lambda x,y: x+y,[len(frames[frame]) for frame in frames])
    #print("Finished loading the path with",total_frames)
    #with  open('SUNRGBD_scene_suggest.csv','w') as fi:
    #TODO: Make every connection discovered by subgraph pattern mining
    for frame in frames:
        data = frames[frame]
        count = 0
        for scene in data:
            json_data = {}
            json_data["ojbects"] = getObjectsForJson(scene)
            json_data["support"] = parentRelationships(scene)
            st = json.dumps(json_data,indent = 4)
            with open("Handmade/SceneSuggest/"+frame + "_"+str(count)+".json",'w') as writer:
                writer.write(st)
            count += 1


#Our general processing function
def sunCGDataMining():
    #This is for the sunCG data-set
    house_path = path_to_data+"data/suncg_data/house"
    parser = SUNCG.SUNCGParser(path_to_data+"data/suncg_data/meta_data/ModelCategoryMapping.csv",house_path+"/",size_sort)
    paths = [f for f in os.listdir(house_path) if not os.path.isfile(os.path.join(house_path,f))]
    frames = defaultdict(list)
    frames = SUNCG.parseSUNCG(paths,frames,parser)
    total_frames = reduce(lambda x,y: x+y,[len(frames[frame]) for frame in frames])
    print ("Finished sorting the file paths:",total_frames)
    with  open('SUNCG_scene_suggest.csv','w') as fi:
        #TODO: Make every connection discovered by subgraph pattern mining
        for (frame,data) in SUNCG.getSingleFrameSunCG(frames,parser):
            #Calcuate our occurance priors to know our base frequencies
            print (frame)
            parents = parentScene(data)
            print ("found",parents.numConnections(),"parents")
            probs = supportSurfacePriors(data,parents)
            print ("found",len(probs[0]),len(probs[1]),"relationships on",len(probs[2]),"objects")
            fi.write(frame+'\n')
            for i in probs[0]:
                fi.write(",".join(["vertical"]+[str(j) for j in i])+'\n')
            for i in probs[1]:
                fi.write(",".join(["horizontal"]+[str(j) for j in i])+'\n')
            for i in probs[2]:
                fi.write(','.join(["frequency"]+[str(j) for j in i])+'\n')
            fi.flush()
            del data #Clean up our mess

def sunRGBDDataMiningKermani( write_type = 'w'):
    '''max_amount controls the maximum number of rooms we look out, which helps us bound the problem
       starting_location tells us that there are rooms in the beginning that we can skip, most likely because we've already looked at them
       location_amount tells us to only look at a certain number of rooms, again so that we can run this in stages
       removed_rooms controls for our noisier rooms which may not give us good data. As an idea, if I was copying Kermani et al., I would remove all rooms except bedroom. Deep convo priors would be all rooms but living, bedroom, and office, etc.
       min_amount is another room remover. We determine a room threshold where we believe that we cannot get good data under the threshold
       write_type tells us if we are overwriting the file or appending.
    '''
    import SUNRGBD
    import NYUSupport

    #NYUSupport.mineNYU2Data(path_to_data,"support_mining.csv")
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
    #print ("Total Objects",len(getObjects(frames)))
    with  open('SUNRGBD_scene_suggest.csv','w') as fi:
        #TODO: Make every connection discovered by subgraph pattern mining
        for frame in frames:
            data = frames[frame]
        #for (frame,data) in SUNCG.getSingleFrameSunCG(frames,parser):
            #Calcuate our occurance priors to know our base frequencies
            print (frame)
            parents = parentScene(data)
            print ("found",parents.numConnections(),"parents")
            probs = supportSurfacePriors(data,parents)
            print ("found",len(probs[0]),len(probs[1]),"relationships on",len(probs[2]),"objects")
            fi.write(frame+'\n')
            for i in probs[0]:
                fi.write(",".join(["vertical"]+[str(j) for j in i])+'\n')
            for i in probs[1]:
                fi.write(",".join(["horizontal"]+[str(j) for j in i])+'\n')
            for i in probs[2]:
                fi.write(','.join(["frequency"]+[str(j) for j in i])+'\n')
            fi.flush()
            del data #Clean up our mess



if __name__ == "__main__":
    #Our data-processing tools
    #import SUNRGBD
    #import SUNCG
    #sunCGDataMining()
    #
    #sunRGBDDataMiningKermani()          
    unrealJsonDataMining()
