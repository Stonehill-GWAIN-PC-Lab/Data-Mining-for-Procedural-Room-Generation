from os import listdir,path
import json
import numpy as np
from collections import defaultdict

def labelSort(x):
        return x.label

class FurnitureItem:
        '''Data structure to hold the different 3D properties of the object'''
        def __init__(self, label,box):
                self.label  = label.lower()
                try:
                        #The three items we need is the tranlation (self.centroid), orientation (self.orientation), and size (self.size)
                        #This code is pretty much copied from the matlab code at http://rgbd.cs.princeton.edu/
                        self.centroid    = box[:3]
                        orientation      = box[3:6]
                        self.size        = box[6:9]
                        if np.linalg.norm(orientation) > 0.0:
                                self.orientation = orientation / np.linalg.norm(orientation)
                        else:
                                self.orientation = orientation
                except:
                        self.centroid = np.zeros(3)
                        self.orientation = np.zeros(3)
                        self.size = np.zeros(3)
        def __eq__(self,other):
                if isinstance(self,other.__class__):
                        return self.label == other.label #For now, label equality
                return False
        def __gt__(self,other):
                if isinstance(self,other.__class__):
                   return self.label > other.label
                return False
        def __lt__(self,other):
                if isinstance(self,other.__class__):
                   return self.label < other.label
                return False
        def __hash__(self):
                return hash(self.label)
        def __str__(self):
                return self.label

class FrameData:
        def __init__(self, annotations, labels,scene_name = None):
                self.labels = labels
                self.annotation3D = sorted(annotations,key = labelSort)
                self.sceneName = scene_name

        def getObject(self,name,exclude_obj = None):
                for obj in self.annotation3D:
                        if name == obj.label:
                                if exclude_obj is None:
                                        return obj
                                else:
                                        if obj is not exclude_obj:
                                                return obj
                return None

def readFrame(file_name):
        try:
                with open(file_name) as data_file:
                        data = json.load(data_file)
        except:
                return None
        numberOfRooms = len(data["all_rooms"])
        rooms = []
        for idx in range(numberOfRooms):
                try:
                        objects = data["all_rooms"][idx]["data"]
                        name = data["all_rooms"][idx]["name"]
                        anootations = []; #Mirror the 3D
                        labels= [];
                        for i in range(len(objects)):
                                x = objects[i]["x"]
                                y = objects[i]["y"]
                                z = objects[i]["z"]
                                rx = objects[i]["rx"]
                                ry = objects[i]["ry"]
                                rz = objects[i]["rz"]
                                sx = objects[i]["length"]
                                sy = objects[i]["width"]
                                sz = objects[i]["height"]
                                box = np.array([x,y,z,rx,ry,rz,sx,sy,sz])
                                try:
                                        labels.append(objects[i]["name"].lower().split(":")[0])
                                except:
                                        labels.append(None)
                                anootations.append(FurnitureItem(labels[-1],box))
                except:
                        pass
                #        print(name)
                #Now we pull the data from the 3D files
                #print(anootations)
                if len(anootations) > 0:
                        frameData = FrameData(anootations,labels,name)
                else:
                        frameData = None
                rooms.append(frameData)
        return rooms;




################################################################################
##Reads the data file given the directory and list of scene names
def getFrames(directory,folder_names,frames = defaultdict(list)):
    for f in folder_names:
        print(f)
        data = readFrame(path.join(directory,f))
        for room in data:
                if room is not None:  
                        frames[room.sceneName].append(room)
    return frames

#This one works as a sieve to only find the rooms of a certain type
def getFramesFindType(directory,folder_names, frames = defaultdict(list), search_type = ""):
        for f in folder_names:
                data = readFrame(path.join(directory,f),True)
                if data is not None:
                        if data.sceneName == search_type:
                                frames[data.sceneName].append(data)
        return frames
