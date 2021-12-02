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

path_to_data = "../"



class SceneGraph:
    '''SceneGraph is used to store the vertices and edges that we build the furniture graphs from (for graph mining)'''
    def __init__(self,vertices = [], edges = []):
        self.vertices = vertices
        self.edges  = set(edges)

    def addVertex(self,vertex):
        self.vertices.add(vertex)
    def addEdge(self,edge):
        if edge not in self.edges:
            self.edges.add(edge)
        for vert in edge:
            if self.vertices[vert] not in self.vertices:
                self.vertices.add(vert)

class DataCleanup:
    #We identify some noise that can be fixed in our data-cleanup and is similar for all objects
    def __init__(self, removed_rooms = [],renamed_rooms = {},renamed_objects = {},room_amounts = (None,None)):
        self._removed_r     = removed_rooms
        self._renamed_r     = renamed_rooms
        self._renamed_o     = renamed_objects
        #Should do some type checking here
        self._min_amounts_r = room_amounts[0]
        self._max_amounts_r = room_amounts[1]
    def addRemovedRoom(self,room):
        #We are fine for over-writing all of these
        self._removed_r.append(room)
    def addRenamedRoom(self,old_name,new_name):
        self._renamed_r[old_name] = new_name
    def addRenamedObjects(self,old_name,new_name):
        self._renamed_o[old_name] = new_name

    def addMinRoomAmounts(self,amount):
        self._min_amounts_r = amount
    def addMaxRoomAmounts(self,amount):
        self._max_amounts_r = amount
        
    def cleanupObjects(self,scenes):
        '''One common data cleanup practice is to get everything under a consistant name. This is the technique to do so'''
        if isinstance(scenes,dict):
            for scene in scenes:
                for room in scenes[scene]:
                    for obj in room.annotation3D:
                        if(obj.label in self._renamed_o): #If the object is known to be a bad way of saying it, then we do
                            obj.label = self._renamed_o[obj.label]
        else:
            for room in scenes:
                for obj in room.annotation3D:
                    if(obj.label in self._renamed_o): #If the object is known to be a bad way of saying it, then we do
                        obj.label = self._renamed_o[obj.label]

    def cleanupRooms(self,scenes):
        '''Another common issue with these scenes is that we have rooms that simply do not work with our method
        A good example is a low number of highly connected rooms, as that will make the isomorphic graph problem explode'''
        #First, we get rid of any rooms we know we don't want. Then, we combine the possible
        #rooms before checking for minimum (as the combination will have an effect on the amounts)
        #Remove the objects that we do not want
        keys = [s_k for s_k in scenes.keys() if s_k in self._removed_r]
        for key in keys:
            del scenes[key]
        #Then, combine the rooms that need to be combined
        changed_names = [s_k for s_k in scenes.keys() if s_k in self._renamed_r.keys()]
        for name in changed_names:
            if self._renamed_r[name] in scenes:
                scenes[self._renamed_r[name]].extend(scenes[name])
            else:
                scenes[self._renamed_r[name]] = scenes[name]
            del scenes[name]
        if self._min_amounts_r is not None:
            keys = [k for k in keys if len(scenes[k]) <= self._min_amounts_r] #Lowest percentage is 5%, so in reality we want min_rooms to be at least greater than 5.
            for key in keys:
                del scenes[key]
        return sorted(scenes.keys())  

    def shrinkSet(self,scenes):
        '''Here, we control the total number of scenes, removing some if necessary'''
        if isinstance(scenes,dict):
            for scene in scenes:
                scenes[scene] = scenes[scene][:min(len(scenes),self._max_amounts_r)]
            return scenes #Actually don't need to return, but whatever
        else:
            #In this case, it's a single item, and we need to return it
            return  scenes[:min(len(scenes),self._max_amounts_r)]
            
            
    
def frequencyFind(scenes,support):
    '''Returns back all the objects that are above the given support threshold'''
    obj_prob = Counter()
    if len(scenes) < 10*NUM_THREADS:
        objects = (ObjectMetrics.labelObjects(scene) for scene in scenes)
    else:
        p = Pool(NUM_THREADS)
        objects = p.map(ObjectMetrics.labelObjects,scenes)
        p.close()
        p.join()
    #Flatten and count here    
    for obj_list in objects:
        for obj in obj_list:
            obj_prob[obj] +=1
    return [obj for obj in obj_prob if obj_prob[obj] >= support]

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
            verts = graphs[i][0]
            edges = graphs[i][1]
            for vert in verts:
                fi.write("v "+str(vert[0])+" "+str(label_dict[vert[1]])+"\n")
            for e in range(len(edges)):
                edge = edges[e]
                fi.write("e "+str(edge[0])+" "+str(edge[1])+" "+str(1)+"\n")
    return label_dict

def run_process(num_support):
    args = ["./../gspan_cpp/build/gbolt", #We use gbolt from https://github.com/Jokeren/DataMining-gSpan as a seperate process
            "-support",str(num_support), #For Kermani, our percentage is usually 10%
            "-output_file","out.txt", #Later, we'll read the patterns, so we need to have an idea of what they will be called
            "-pattern","true", 
            "-mvertices","10", #We also limit the maximum vertex size because of memory constraints. G-Span is a memory intensive algorithm, and by limiting the constraints, we bound the problem. 
            "-input_file","input.txt"] #Since we called our file above input.txt, we also call that
    subprocess.run(args)
    return

def return_data_frame(label_dict):
    r_label_dict = {}
    for key in label_dict:
        r_label_dict[label_dict[key]] = key
    paths = [f for f in os.listdir(".") if os.path.isfile(f) and "out.txt" in f]
    report_df = {}
    keys = ["support","verts","edges","index"]#,"num_vert"
    for key in keys:
        report_df[key]  = []
    try: #We wrap this in a try block because it is always possible that gbolt ran out of memory and when it does we need to say that our gspan failed
        for path in paths:
            vert = []
            edge = []
            with open(path,'r') as fi:
                lines = fi.readlines()
                for line in lines:
                    if len(line) == 1:
                        report_df["verts"].append(np.array(vert))
                        #report_df["num_vert"].append(len(vert))
                        report_df["edges"].append(np.array(edge))
                        vert  = []
                        edge  = []
                    else:
                        data = line.strip().split(" ")
                        if data[0] == "t":
                            #This is our support and index
                            report_df["index"].append(data[2])
                            report_df["support"].append(data[4])
                        elif data[0] == "v":
                            vert.append([data[1],r_label_dict[data[2]]])
                        elif data[0] == "e":
                            edge.append([data[1],data[2]])
                        else:
                            pass #We skip the x
            if len(vert) > 0 and len(edge) > 0: #Need to add the last one
                report_df["verts"].append(np.array(vert))
                #report_df["num_vert"].append(len(vert))
                report_df["edges"].append(np.array(edge))
        df = pd.DataFrame(report_df)
    except:
        return None
    df.set_index('index',inplace = True)
    return df


##These functions our more of our graph construction functions##
def testInsert(edge,T):
    '''Ensures that the edge does not create a cycle in the graph (which would make it no longer a tree)'''
    #We test by using a BFS of T + E
    edges = [a for a in [edge]+list(T.edges)]#Deep copy
    stack = [edges[0][0]]
    found_items = []
    found_edges = []
    while len(stack) > 0 and len(edges) >= 0:
        item = stack.pop()
        if item not in found_items:
            found_items.append(item)
            for edge in edges:
                if item == edge[0]:
                    stack.insert(0,edge[1])
                    found_edges.append(edge)
            edges = [e for e in edges if e not in found_edges] #Remove edges in stack
            if len(stack) == 0 and len(edges) > 0:
                stack.append(edges[0][0])
        else: #If we've already connected to the item, then we're in trouble
            return False
    return True


def minSpanningGraph(objects,c_func = ObjectMetrics.proximityCenter,value_array = None):
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
    T = SceneGraph(V) #Edges for minimum spanning tree
    while len(E) > 0:
        edge = E.pop(0)
        if testInsert(edge[0],T):
            T.addEdge(edge[0])
    return T #What we have here is an ijv sparse rep

def thresholdGraph(objects,function):
    #Instead of looking at a min-spanning graph, we can do the much more computationally expensive every actual connection graph
    E = SceneGraph(objects)
    for i in range(len(objects)):
        for j in range(i+1,len(objects)):
            if function(objects[i],objects[j]):
                E.addEdge((i,j)) #This has our cost function
    return E #What we have here is an ijv sparse rep

def graphFromEdges(edges):
    #Creates a scenegraph from the edges
    #First, we get all the vertices
    mapper = {}
    ordering = []
    counter = 0
    for e in edges:
        if e[0] not in mapper:
            mapper[e[0]] = counter
            counter += 1
            ordering.append(e[0])
        if e[1] not in mapper:
            mapper[e[1]] = counter
            counter += 1
            ordering.append(e[1])
    T = SceneGraph(ordering)
    #Then, we build the graph
    for e in edges:
        T.addEdge((mapper[e[0]],mapper[e[1]]))
    return T
def frequencyCosts(obj1,obj2,value_array):
    '''Determines the frequency cost of two objects based on their frequency value in the value_array'''
    if "$".join([obj1.label,obj2.label]) not in value_array:
        #then it doesn't exist and we return None
        return None
    return value_array["$".join([obj1.label,obj2.label])]

def graphRelationHelper(graph_type,graph_func,good_objects,scene):
    '''Partial helper function to allow us to paralleize graph matching'''
    objs = [obj for obj in sorted(scene.annotation3D) if obj.label in good_objects]
    graph = graph_type(objs,graph_func)
    if len(graph.vertices) == 0 or len(graph.edges) == 0:
        return None
    res = ([(i,graph.vertices[i].label) for i in range(len(graph.vertices))],[(e[0],e[1],1) for e in graph.edges])
    del graph
    return res

def subprocessGraphRelations(scenes,percent_threshold,graph_func,graph_type):
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
    #print (label_dict)
    run_process(percent_threshold)#Processes the file
    df = return_data_frame(label_dict) #Reads back in the file as a pandas dataframe
    if df is None or 'verts' not in df.columns.values:
        return None
    return df
    

def prtvHelper(data):#Cannot pickle local methods
    return minSpanningGraph(data)

def prtvGetEdges(graph): #Taking a specific graph, we figure out all the edges we have 
    relations = Counter()
    for edge in g.edges:
        label = "$".join([g.vertices[edge[0]].label,g.vertices[edge[1]].label])
        relations[label] +=1
    return relations

def vert2Series(row): #Quickly used to create a renamed vertex series. Not the best way to do this, but based on our old code, it was.
    names = {}
    vals = pd.Series(row[:,1]).apply(lambda x:renameVert(x,names)).values.tolist()
    return vals
    
def renameVert(val,names):
    '''Our applied helper function to get distinct vertices per graph'''
    rl = val
    if val in names:
        rl +="_"+str(names[val])
        names[val] +=1
    else:
        names[val] = 1
    return rl

def renameEdges(row):
    '''Once we rename the vertices, we need to rename the vertices in the edges as well'''
    try:
        edges = row.edges
        rv = row.renamed_verts
        li =  map(lambda x:[rv[int(i)] for i in x],edges)
        return [sorted(f) for f in li]
    except:
        pass
    return []

def subgraphs2Factors(dataframe):
    '''Converts freq subgraph minining into our factor graph representation'''
    if dataframe is None:
        return []
    dataframe['renamed_verts'] = dataframe.verts.apply(lambda x:vert2Series(x))
    dataframe['renamed_edges'] = dataframe[['edges','renamed_verts']].apply(lambda x:renameEdges(x),axis = 1)
    df = dataframe[['renamed_edges','support']].values.tolist()
    df = [[f+[d[1]] for f in d[0]] for d in df]
    df = [item for sublist in df for item in sublist]
    #Gets us the highest unique values
    try:
        df = pd.DataFrame(df).sort_values(by=[2],ascending = False).drop_duplicates(subset= [0,1] ).values.tolist() #This means we don't have duplicated items in our set, making what we do have easier to parse
        df = [[str(j) for j in i] for i in df]
        return df
    except: #Exceptions occur during an empty dataframe
        return []
    finally:
        del dataframe

def getName(label_1,label_2,dic,seen_objects):
    ''' Returns the name of the two objects based on what we have already seen.'''
    count = 1
    if label_1 == label_2:
        name = "$".join([label_1,label_2 +"_"+ str(count)])
    else:
        name = "$".join([label_1,label_2])
    if name not in dic:
        return name
    searching = True
    while name in dic and searching:
        #Here, we have to be careful about which one we are doing, because we may go over
        #All of the ones that use get name are symmetric, so we can switch
        l2 = label_2 + "_" +str(count)
        if l2 in seen_objects:
            name = "$".join([label_1,l2])
        else:
            searching = False
        count +=1
    if searching is False: #We did not find what we were looking for, switch objects and try again
        searching = True
        count = 1
        while searching:
            #Here, we have to be careful about which one we are doing, because we may go over
            #All of the ones that use get name are symmetric, so we can switch
            l2 = label_1 + "_" +str(count)
            if l2 in seen_objects:
                name = "$".join([label_2,l2])
                if name not in dic:
                    searching = False
            else:
                searching = False
            count +=1
        if name in dic: #We go with the first sure case
            if label_1 == label_2:
                name = "$".join([label_1,label_2 +"_1"])
            else:
                name = "$".join([label_1,label_2])
    return name

def frameBBHelper(scene):
    '''This gets all the relationships that appear to only be counts in Kermani (although more could be added)'''
    #These two allow us to not have more than one per frame
    seen_sym =  Counter() #symmetry (name and size)
    seen_orip = Counter() #orientation parallel
    seen_oris = Counter() #orientation perp
    seen_ss   = Counter() #side to side
    seen_objects = ObjectMetrics.labelObjects(scene) #Removes duplication
    #I am not sure if Kermani is using 3D or 2D (I now think 2D, although the data doesn't exist for some of it)
    #First we will test on 3D though because that is what everything was written to be based off of

    #It should be noted that the ordering of objects is important. This is why objects can be sorted in their generators
    for i in range(len(scene.annotation3D)):
        #print (scene.annotation3D[i].label,scene.annotation3D[i].orientation)
        for j in range(i+1,len(scene.annotation3D)):
            if(ObjectMetrics.orientationCosRelations(scene.annotation3D[i],scene.annotation3D[j])):
                name = getName(scene.annotation3D[i].label,scene.annotation3D[j].label,seen_orip,seen_objects)
                seen_orip[name] = True
            if(ObjectMetrics.orientationSinRelations(scene.annotation3D[i],scene.annotation3D[j])):
                name = getName(scene.annotation3D[i].label,scene.annotation3D[j].label,seen_oris,seen_objects)
                seen_oris[name] = True
            if(ObjectMetrics.symmetryRelations(scene.annotation3D[i],scene.annotation3D[j])):
                name = getName(scene.annotation3D[i].label,scene.annotation3D[j].label,seen_sym,seen_objects)
                seen_sym[name]  = True
            if ObjectMetrics.supportMiningBoundingBox(scene.annotation3D[i],scene.annotation3D[j]):
                name = getName(scene.annotation3D[i].label,scene.annotation3D[j].label,seen_ss,seen_objects)
                seen_ss[name] = True
    return (seen_sym,seen_orip,seen_oris,seen_ss,seen_objects)
    

def frameBoundingBox(frame,prox_objects = None):
    '''This function will determine thresholds, as per kermani et al.
        A frame is a list of frames. From each frame, we build a min-spanning graph for
        each undirected, weighted edge'''
    total_scenes = len(frame)
    support      = Counter() #support a$b, which is not in the dataset 
    symmetry     = Counter() #sym a$b count
    orientation  = Counter() #orientation a$b count
    orientationS = Counter() #Breaking up sine and cos orientation. This one is sin
    sts          = Counter() #Side to side a$b count
    freq         = Counter() #Frequency of seen objects
    p = Pool(NUM_THREADS)
    result = map(frameBBHelper,frame)
    p.close()
    p.join()
    for (seen_sym,seen_orip,seen_oris,seen_ss,seen_objects) in result:
        #We use the counters as they are intended to be here
        for key in seen_sym:
            symmetry[key] += 1
        for key in seen_orip:
            orientation[key] +=1
        for key in seen_oris:
            orientationS[key] +=1
        for key in seen_ss:
            sts[key] +=1
        for obj in seen_objects:
            freq[obj] +=1
    orientation  = [name.split("$")+[str(orientation[name])] for name in orientation if orientation[name] >= math.ceil(0.05*len(frame))]
    orientationS = [name.split("$")+[str(orientationS[name])] for name in orientationS if orientationS[name] >= math.ceil(0.01*len(frame))]
    symmetry     = [name.split("$")+[str(symmetry[name])] for name in symmetry if symmetry[name] >= math.ceil(0.01*len(frame))]
    sts          = [name.split("$")+[str(sts[name])] for name in sts if sts[name] >= math.ceil(0.05*len(frame))]
    #Here, we get all the names as a set, and use them for the total frequency
    names = []
    for i in orientation + orientationS + symmetry:
        names.append(i[0])
        names.append(i[1])
    if prox_objects is not None:
        names.extend(prox_objects)
    names = set(names)
    #print ("Missing:",[n for n in names if n not in freq],"from",[n for n in prox_objects if n not in freq])
    freq        = [[name,str(freq[name])] for name in names if name in freq] #We threshold this because this is the absolute minimum support from all the factors from the paper
    return (orientation,orientationS,symmetry,sts,freq)

def supportBoundingBox(scenes,good_objects= None,threshold = 0.1):
    total_scenes = len(scenes)
    support      = {} #support a$b, which is not in the dataset 
    for scene in scenes:
        found_items = {}
        for i in range(len(scene.annotation3D)):
            obj1 = scene.annotation3D[i]
            if good_objects is None or obj1.label in good_objects:
                for j in range(i+1,len(scene.annotation3D)):
                    obj2 = scene.annotation3D[j]
                    if good_objects is None or obj2.label in good_objects:
                        if supportMiningBoundingBox(obj1,obj2):
                            found_items[(obj1.label,obj2.label)] = True
                            #found_items[(obj2.label,obj1.label)] = True
        for key in found_items:
            if key in support:
                support[key] +=1
            else:
                support[key] = 1
    return [list(name) + [str(support[name])] for name in support if support[name] > threshold*total_scenes]
            




def sceneSupport(frame,data,fi):
    ''' Performs something similar to scenesupport for our data-set'''
    parents = parentScene(data)
    (total_support,freq) = supportSurfaceCounts(data,parents,0.1)
    for item in total_support[0]: #Vertical support
        str_item = [str(i) for i in item]
        fi.write(','.join(['Vertical-Support']+str_item)+'\n')
    for item in total_support[1]: #Horizontal support
        str_item = [str(i) for i in item]
        fi.write(','.join(['Horizontal-Support']+str_item)+'\n')
    return [i[0] for i in freq] #This is our good support items
    

def KermaniRelationships(frame,data,fi,prox_list,debug = False):
    '''Mines our different relationships based on Kermani et al.'''
    prox = subgraphs2Factors(subprocessGraphRelations(data,0.05,ObjectMetrics.proximityCenter,minSpanningGraph))#Should move all percents out so they can be tuned
    print("Finished determining proximity relations, writing:",len(prox) )
    for item in prox:
        fi.write(','.join(['proximity']+list(item))+'\n')
        prox_list.extend(item[0:2])
    del prox
    (orip,oris,sym,sts,freq) = frameBoundingBox(data,prox_list)
    for item in orip:
        fi.write(','.join(['orientation_parallel']+item)+'\n')
    for item in oris:
        fi.write(','.join(['orientation_perpendicular']+item)+'\n')
    for item in sym:
        fi.write(','.join(['symmetry']+item)+'\n')
    for item in sts:
        fi.write(','.join(['side_to_side']+item)+'\n')
    for item in freq: #Not mining freq right now
        fi.write(','.join(['frequency']+item)+'\n')
    print ("And all of our other information, writing:",len(orip),len(oris),len(sym),len(sts))
    print ("We have a total of",len(freq),"objects")

def determineParents(scene):
    parents = {}
    #We sort by size because saava et al. mentions bounding box volume
    #and this implimentation has a precolunation towards objects sooner in the list
    #objs = sorted(scene.annotation3D,key = size_sort)
    #print([i.label for i in scene.annotation3D])
    for i in range(len(scene.annotation3D)):
        obj1 = scene.annotation3D[i]
        for j in range(i+1,len(scene.annotation3D)):
            obj2 = scene.annotation3D[j]
            sup = ObjectMetrics.supportMiningBoundingBox(obj1,obj2)#Favors size objects
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
    #Converts this to the breakup form
    final_sol = {}
    for p in parents:
        for item in parents[p]:
            final_sol[p+"$"+item] = True
    return final_sol

def determineProximity(scene):
    #ObjectMetrics.proximityCenter,minSpanningGraph
    graph = minSpanningGraph(scene.annotation3D,ObjectMetrics.proximityCenter)
    (objs,rels) = ([(i,graph.vertices[i].label) for i in range(len(graph.vertices))],[(e[0],e[1],1) for e in graph.edges])
    output = {}
    for rel in rels:
        source = [i[1] for i in objs if i[0] == rel[0]][0]
        target = [i[1] for i in objs if i[0] == rel[1]][0]
        output[source+"$"+target]=True
    return output

def getObjects(frames):
    object_labels = []
    for frame in frames:
        for scene in frames[frame]:
            object_labels.extend([obj.label for obj in scene.annotation3D])
    return set(object_labels)

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

def convertObjects(li_objs,remove = False):
    '''
    converts a set of objects into a list of dictionaries
    '''
    end_li = []
    counter = 0
    for item in li_objs:
        dic = {}
        dic["id"] = str(counter)
        if remove:
            if re.search("\d+",item) is not None:
                dic["name"] = "_".join(item.split("_")[:-1])
            else:
                dic["name"] = item
        else:
            dic["name"] = item
        dic["frequency"] = 1.0
        end_li.append(dic)
        counter += 1
    return end_li

    

def convertFactors(info_dics,obj_dics):
    rules_li = []
    probs_li = []
    counter = 0
    factor_types = ['symmetry','orientation_perpendicular','orientation_parallel','side_to_side','support','proximity']
    factor_transform = {
        'symmetry':"Guassian-one",
        'orientation_perpendicular':"Guassian-one",
        'orientation_parallel':"Guassian-one",
        'side_to_side':"Distance-all",
        'support':"Distance-all",
        'proximity':"Distance-all"
        }
    for factor in range(len(info_dics)):
        infos = info_dics[factor]
        fact = factor_types[factor]
        for item in infos:
            #print(item,fact)
            rule = {}
            prob = {}
            rule["id"] = str(counter)
            rule["type"] = factor_transform[fact]
            rule["source"] = [int(o["id"]) for o in obj_dics if o["name"] == item.split("$")[0]][0]
            rule["target"] = [int(o["id"]) for o in obj_dics if o["name"] == item.split("$")[1]][0]
            prob["id"] = str(counter)
            prob["frequency"] = 1.0
            counter += 1
            rules_li.append(rule)
            probs_li.append(prob)
    return (rules_li,probs_li)

def sunCGDataMiningKermani(starting_location = None,data_cleanup = None,write_type = 'w'):
    '''max_amount controls the maximum number of rooms we look out, which helps us bound the problem
       starting_location tells us that there are rooms in the beginning that we can skip, most likely because we've already looked at them
       location_amount tells us to only look at a certain number of rooms, again so that we can run this in stages
       removed_rooms controls for our noisier rooms which may not give us good data. As an idea, if I was copying Kermani et al., I would remove all rooms except bedroom. Deep convo priors would be all rooms but living, bedroom, and office, etc.
       min_amount is another room remover. We determine a room threshold where we believe that we cannot get good data under the threshold
       write_type tells us if we are overwriting the file or appending.
    '''
    #This is for the sunCG data-set
    import SUNCG
    house_path = path_to_data+"data/SUNCG/suncg_data/house"
    parser = SUNCG.SUNCGParser(path_to_data+"data/SUNCG/suncg_data/meta_data/ModelCategoryMapping.csv",house_path+"/",size_sort)
    paths = [f for f in os.listdir(house_path) if not os.path.isfile(os.path.join(house_path,f))]
    frames = defaultdict(list)
    frames = SUNCG.getFramesSUNCG(paths,frames,parser)#SUNCG.parseSUNCG(paths,frames,parser)
    print ("Total Objects",len(getObjects(frames)))
    quit()
    if data_cleanup is not None:
        keys = data_cleanup.cleanupRooms(frames)
    from functools import reduce
    total_frames = reduce(lambda x,y: x+y,[len(frames[frame]) for frame in keys])
    print ("Finished sorting the file paths:",total_frames)
    with  open('mining_suncg_kermani.csv',write_type) as fi:
        #TODO: Make every connection discovered by subgraph pattern mining
        for (frame,data) in SUNCG.getSingleFrameSunCG(frames,parser,keys):
            if data_cleanup is not None:
                data = data_cleanup.shrinkSet(data)
                data_cleanup.cleanupObjects(data)
            fi.write(frame+','+str(len(data))+'\n')
            print (frame+','+str(len(data))+'\n')
            prox_list = sceneSupport(frame,data,fi)
            print ("Finished determining suppport relationships")
            KermaniRelationships(frame,data,fi,prox_list,True)
            fi.flush() #Make sure we write at the end of each mining operation 
            del data #Clean up our mess
            del prox_list
    print ("Finished mining the files")

def sunRGBDDataMiningKermani(starting_location = None,data_cleanup = None, write_type = 'w'):
    '''max_amount controls the maximum number of rooms we look out, which helps us bound the problem
       starting_location tells us that there are rooms in the beginning that we can skip, most likely because we've already looked at them
       location_amount tells us to only look at a certain number of rooms, again so that we can run this in stages
       removed_rooms controls for our noisier rooms which may not give us good data. As an idea, if I was copying Kermani et al., I would remove all rooms except bedroom. Deep convo priors would be all rooms but living, bedroom, and office, etc.
       min_amount is another room remover. We determine a room threshold where we believe that we cannot get good data under the threshold
       write_type tells us if we are overwriting the file or appending.
    '''
    import SUNRGBD
    import NYUSupport

    NYUSupport.mineNYU2Data(path_to_data,"support_mining.csv")
    frames = defaultdict(list)
    a = path_to_data+"data/SUNRGBD/sunrgbd/"
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
    with open('mining.csv',write_type) as fi:
        #TODO: Make every connection discovered by subgraph pattern mining
        for frame in keys:
            data = frames[frame]
            fi.write(frame+','+str(len(data))+'\n')
            print (frame+":"+str(len(data)))
            KermaniRelationships(frame,data,fi,[],True)
            del data #Clean up our mess
    print("Finished running file")

def KermaniRoomGen(location_set,path,name):
    '''max_amount controls the maximum number of rooms we look out, which helps us bound the problem
       starting_location tells us that there are rooms in the beginning that we can skip, most likely because we've already looked at them
       location_amount tells us to only look at a certain number of rooms, again so that we can run this in stages
       removed_rooms controls for our noisier rooms which may not give us good data. As an idea, if I was copying Kermani et al., I would remove all rooms except bedroom. Deep convo priors would be all rooms but living, bedroom, and office, etc.
       min_amount is another room remover. We determine a room threshold where we believe that we cannot get good data under the threshold
       write_type tells us if we are overwriting the file or appending.

       This function transforms a location into a json set where those relationships exist
    '''
    counter = 0
    import json
    big_li = []
    with open(os.path.join(path,name+'.json'),'w') as fi:
        for location in location_set:
            #Identify the relationships in the file
            information = frameBBHelper(location)
            support_info = determineParents(location)
            prox_info = determineProximity(location)
            factors = [i for i in information[:-1]]
            #Parallel, perpindicular,symmetry, and side to side
            objects = information[-1]
            #We need to convert this into vertices
            json_dic = {}
            json_dic["Objects"] = convertObjects(objects)
            json_dic["Constraints"],temp = convertFactors(factors+[support_info,prox_info],json_dic["Objects"])
            json_dic["Objects"] = convertObjects(objects)
            #print(information)
            room_holder = {}
            room_holder["setup"] = json_dic
            room_holder["location"] = name+"_"+str(counter)
            big_li.append(room_holder)
            counter +=1
        json_string = json.dumps(big_li,indent = 4)
        fi.write(json_string)
    print("Finished",name)

def sunRGBDGen():
    import SUNRGBD
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
    for frame in frames:
        KermaniRoomGen(frames[frame],"kermani-gen",frame)


def unrealJsonDataMining():
    #Mines from positional information from our static mesh saver
    import UnrealJson
    frames = defaultdict(list)
    a = path_to_data + "Independent Studies/Generated Data/handmade"
    paths = [f for f in os.listdir(a) if os.path.isfile(os.path.join(a,f))]
    frames = UnrealJson.getFrames(a,paths,frames)
    #print(frames)
    total_frames = reduce(lambda x,y: x+y,[len(frames[frame]) for frame in frames])
    for frame in frames:
        KermaniRoomGen(frames[frame],"Handmade/kermani",frame)

if __name__ == "__main__":
    import os
    #same_rooms   = {"idk":"corridor","recreation_room":"rest_space"}
    #same_objects = {"fridge":"refridgerator","bathroomvanity":"bathroom_vanity","toyhouse":"toy_house","bookshelf":"book_shelf","tissuebox":"tissue_box"}
    #removed_rooms = ["Dining_Room_Garage_Gym","Dining_Room_Kitchen_Office_Garage","Room","Living_Room_Dining_Room_Kitchen_Garage"]
    #support = (20,1000)
    #dc = DataCleanup(removed_rooms,same_rooms,same_objects,support)
    #sunCGDataMiningKermani(data_cleanup = dc)
    #sunRGBDDataMiningKermani(data_cleanup = dc)
    unrealJsonDataMining()

