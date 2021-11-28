#import cv2
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
                #This code is pretty much copied from the matlab code at http://rgbd.cs.princeton.edu/
                x = box['X'] #box.X
                y = box['Z'] #box.Z
                vector1     = np.array([x[1]-x[0],y[1]-y[0],0])
                coeff1      = np.linalg.norm(vector1)
                vector1    /= coeff1
                vector2     = np.array([x[2]-x[1],y[2]-y[1],0])
                coeff2      = np.linalg.norm(vector2)
                vector2    /= coeff2
                up          = np.cross(vector1,vector2)
                vector1    *= up[2]/up[2] #No clue why this is in their, but it was in the matlab code
                vector2    *= up[2]/up[2]
                zmax        = -box['Ymax']
                zmin        = -box['Ymin']
                centroid2D  = np.array([0.5*(x[0]+x[2]),0.5*(y[0]+y[2])])
                orientation = np.array([0.5*(x[1]+x[0])-centroid2D[0],0.5*(y[1]+y[0])-centroid2D[1],0])
                self.basis       = np.stack((vector1,vector2,np.array([0,0,1]))) #axis = 0 is vstack
                self.coeffs      = np.absolute(np.array([coeff1,coeff2,zmax-zmin]))/2
                self.centroid    = np.array([centroid2D[0],centroid2D[1],0.5*(zmin+zmax)])
                if np.linalg.norm(orientation) > 0.0:
                        self.orientation = orientation / np.linalg.norm(orientation)
                else:
                        self.orientation = orientation 
                self.size        = np.array([abs(x[2]-x[0]),abs(y[2]-y[0]),abs(zmax-zmin)])
                                
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
        def __init__(self, imgRGB,imgD,annotation2D,labels2D,annotation3D=None,scene_name = None):
                self.imgRGB = imgRGB
                self.imgD = imgD
                self.annotation2D = annotation2D
                self.labels2D = labels2D
                self.annotation3D = sorted(annotation3D,key = labelSort)
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
                
def readFrame( framePath, bfx ):
        #read RGB information to numpy array    
        #rgbPath = framePath + "/image/" 
        #rgbPath += listdir(rgbPath)[0]
        #imgRGB = cv2.imread(rgbPath);
        imgRGB = None
        
        #read depth information to numpy array
        if not(bfx):
                depthPath = framePath + "/depth/" 
        else:
                depthPath = framePath + "/depth_bfx/" 
        depthPath += listdir(depthPath)[0]
        #imgD = cv2.imread(depthPath);
        imgD = None
        #read 2D annotations to a list o numpy arrays where each index is related with one object polygon and a list where the index links the object polygon to the object label.
        anotation2D = framePath + "/annotation2Dfinal/index.json"
        try:
                with open(anotation2D) as data_file:    
                        data = json.load(data_file)
        except:
                return None
        numberOfAnot = len(data["frames"][0]["polygon"]);
        
        anootation2D = [];
        anootation3D = []; #Mirror the 3D
        labels2D = [];
        for i in range(0,numberOfAnot):
                x = data["frames"][0]["polygon"][i]["x"]
                y = data["frames"][0]["polygon"][i]["y"]

                idxObj = data["frames"][0]["polygon"][i]["object"];
                pts2 = np.array([x,y], np.int32)
                pts2 = np.transpose(pts2);
                anootation2D.append(pts2);
                try:
                        labels2D.append(data['objects'][idxObj]["name"].lower().split(":")[0])
                except:
                        labels2D.append(None)
        sceneName = framePath + "/scene.txt"
        with open(sceneName) as data_file:
                name = data_file.readlines()[0]
        anotation2D = framePath + "/annotation3Dfinal/index.json"
        try:
                with open(anotation2D) as data_file:    
                        data = json.load(data_file)
                        dat = data["objects"]
                        for d in dat:
                                if d is not None and "polygon" in d and len(d["polygon"]) > 0:
                                        try:
                                                label = d["name"].split(":")[0] #Remove difference between obj and obj:occuluded
                                                box   = d["polygon"][0]
                                                anootation3D.append(FurnitureItem(label,box))
                                        except Exception as e:
                                                #print(e,d)#If we miss one, keep going
                                                #return None
                                                pass
        except:
                pass
        #Now we pull the data from the 3D files
        if len(anootation3D) > 0:
                frameData = FrameData(imgRGB,imgD,anootation2D,labels2D,anootation3D,name)
        else:
                frameData = None
        return frameData;


                
        
################################################################################
##Reads the data file given the directory and list of scene names
def getFrames(directory,folder_names,frames = defaultdict(list)):
    print("getFrames()")
    for f in folder_names:
        print('f:')
        print(f)
        print("readFrame path:")
        print(path.join(directory,f))
        data = readFrame(path.join(directory,f),True)
        if data is not None:
            frames[data.sceneName].append(data)
    return frames

#This one works as a sieve to only find the rooms of a certain type
def getFramesFindType(directory,folder_names, frames = defaultdict(list), search_type = ""):
        for f in folder_names:
                data = readFrame(path.join(directory,f),True)
                if data is not None:
                        if data.sceneName == search_type:
                                frames[data.sceneName].append(data)
        return frames
