import json
import numpy as np
from scipy.spatial import ConvexHull

#Parallel processing for speedup
from multiprocessing import Pool
from functools import partial,reduce
from collections import defaultdict

NUM_THREADS = 8

def labelSort(x):
        return x.label

def getPoints(obj):
    '''Returns a set of vertex points from an obj file'''
    points = [point.strip().split(" ") for point in obj] #Baseline all points
    points = [[float(p) for p in point[1:]] for point in points if point[0] == "v"]
    return np.array(points)
        

class FurnitureItem:
        '''Data structure to hold the different 3D properties of the object'''
        def __init__(self, label,bounding_box,transform = None,conversion = None):
                self.label    = label.lower()
                try:
                        if conversion is None:
                                transform = np.array(transform).reshape(4,4).T #4,4 matrix
                        else:
                                transform = np.array(transform).reshape(4,4).T *conversion[2]
                        self.centroid = transform[:3,3]
                        self.orientation = np.zeros(3)
                        self.orientation[0] = np.arctan2(transform[1,0],transform[0,0])
                        self.orientation[1] = np.arctan2(-transform[2,0],abs(transform[2,1]-transform[2,2]))
                        self.orientation[2] = np.arctan2(transform[2,1],transform[2,2])
                        if conversion is None:
                                self.size = (np.array(bounding_box['max']) - np.array(bounding_box['min']))
                        else:
                                self.size = (np.array(bounding_box['max']) - np.array(bounding_box['min']))*conversion[2]  
                except:
                        self.size = None
                        #print(label,transform,bounding_box)
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
        def __init__(self,annotation3D=None,scene_name = None, sort_function = labelSort):
                self.annotation3D = sorted(annotation3D,key = sort_function)
                self.sceneName = scene_name
                self.walls = None
        def getObject(self,name,exclude_obj = None):
                for obj in self.annotation3D:
                        if name == obj.label:
                                if exclude_obj is None:
                                        return obj
                                else:
                                        if obj is not exclude_obj:
                                                return obj
                return None

        def getWalls(self):
                if self.walls is not None:
                        return self.walls
        def setWalls(self,bounds):
                self.walls = bounds


#Helper functions for figuring out the extents
def fitBoundingBox(points):
    '''Fits a simple bounding box around the points'''
    min_p = np.min(points,axis = 0)#Minimum points
    max_p = np.max(points,axis = 0)#Maximum points
    return {'max':max_p,'min':min_p}

def buildTransform(bounding_box,floor,up):
    '''Takes a bounding box and builds a tranformation from it. SUNCG has it's tranformation (in nodes) as 
    [r11,r21,r31,0,r12,r22,r32,0,r13,r23,r33,0,x,y,z,1] So basically transpose. We don't have rotation for our
    building data so we instead assume an identity matrix
    '''
    transform = np.eye(4)
    for i in range(3):
        transform[3,i] = (bounding_box['min'][i]+bounding_box['max'][i])/2
    if floor:
        transform[3,up] = bounding_box['min'][up]
    else:
        transform[3,up] = bounding_box['max'][up]
    return transform.flatten()
def wallBoundingBox(points,up):
        '''Uses the Convex Hull to get Wall Bounding Boxes. Now they are aabb
        but in reality we probably want obbs and provide them with transforms'''
        twod = np.delete(points,[up],axis = 1)
        #twod = np.vstack({tuple(row) for row in twod}) #Stack overflow https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array second answer
        try:
                convex_hull = ConvexHull(twod)
                return twod[convex_hull.simplices]
        except Exception as e: #It is only that points cannot contain NAN, because there are litterally walls with NAN in them
                pass
                #print (points)
                #print (twod)
        return None
        

def createWallTransform(points,up,front,height):
    #For each pair of points, we create a size, translation, and orientation
    #Use the length to remove walls that fall below the threshold (small points that seem to creep in)
    result = []
    mask = [np.linalg.norm(point[0]-point[1]) >0.01 for point in points]  

    translate = [(point[0]+point[1])/2 for point in points[mask]]
    translate = [np.insert(point,up,0) for point in translate]
    
    bounds = [fitBoundingBox(point) for point in points[mask]]
    for point in bounds:
        point["min"] = np.insert(point["min"],up,height[0])
        point["max"] = np.insert(point["max"],up,height[1])
    rotations = []
    #We need to convert our 2 points into three points
    for point in points[mask]: #https://stackoverflow.com/questions/51565760/euler-angles-and-rotation-matrix-from-two-3d-points
        a_vec = np.zeros(3)
        a_vec[front] = 1
        b_vec = (point[0]-point[1]) / np.linalg.norm(point[0]-point[1])
        b_vec = np.insert(b_vec,up,0)
        cross = np.cross(a_vec,b_vec)
        ab_angle = np.arccos(np.dot(a_vec,b_vec))
        vx = np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
        R = np.identity(3)*np.cos(ab_angle) + (1-np.cos(ab_angle))*np.outer(cross,cross) + np.sin(ab_angle)*vx
        rotations.append(R)
    for i in range(len(rotations)):
        M = np.eye(4)
        M[:3,:3] = rotations[i]
        M[:3,3]   = translate[i]
        res = {}
        res["transform"] = M.T.flatten()
        res["bounding_box"] = bounds[i]
        result.append(res)
    return result

class SUNCGParser:
        '''Parsing mechanism as the model names are seperate from the json file'''
        def __init__(self,model_mapping,path,sort_function = labelSort,seperate_rooms = False):
                '''This function initalizes the model pointing function'''
                lines = open(model_mapping,'r').readlines()
                #index,model_id,fine_grained_class,coarse_grained_class,empty_struct_obj,nyuv2_40class,wnsynsetid,wnsynsetkey
                lines = lines[1:]#The first one isn't useful for our purposes
                #Here, we get the id and the fine grained name as an iterator
                self.object_mapping = {}
                for item in lines:
                        d = item.split(",")
                        self.object_mapping[d[1]] = d[2]
                self.sort = sort_function
                self.path = path
                self.sep_rooms = seperate_rooms
##        def buildExtents(self,bounding_box,conversion):
##                '''Builds a ceiling and floor for now'''
##                center = [(bounding_box['max'][i] + bounding_box['min'][i])/2*conversion[2] for i in range(3)]
##                #Floor transformation
##                f_center = [i for i in center]#Deep copy
##                f_center[1] = bounding_box['min'][1]+0.025#Our size of our floor and ceiling is 5 cm
##                f_trans = [1,0,0,0 ,0,1,0,0 ,0,0,1,0]+f_center + [1]
##                #Ceiling transformation
##                c_center = [i for i in center]#Deep copy
##                c_center[1] = bounding_box['max'][1]-0.025#Our size of our floor and ceiling is 5 cm
##                c_trans = [1,0,0,0 ,0,1,0,0 ,0,0,1,0]+c_center + [1]
##                #sizes (which are the same for both)
##                bounding_box["max"][1] = 0.05
##                bounding_box["min"][1] = 0.00
##                return [FurnitureItem("floor",bounding_box,f_trans),FurnitureItem("ceiling",bounding_box,c_trans)]+self.buildWalls(location,conversion)

        def buildBounds(self,name,front,up):
                path = '/'.join([i for i in self.path.split("/")][:-2] + ["room",''])+name
                items = []
                try: #Floors and ceilings don't always exist (at least ceilings don't)
                        ceiling = open(path+"c.obj").readlines() #c means ceiling
                        c_points = fitBoundingBox(getPoints(ceiling))#Fit a bounding box around ceiling and floor
                        items.append(FurnitureItem("Ceiling",c_points,buildTransform(c_points,False,up)))
                except:
                        pass
                try:
                        
                        floor   = open(path+"f.obj").readlines() #f for floor
                        f_points = fitBoundingBox(getPoints(floor))  #Fit a bounding box around ceiling and floor
                        items.append(FurnitureItem("Floor"  ,f_points,buildTransform(f_points,True,up)))
                except:
                        pass
                try:
                        walls   = open(path+"w.obj").readlines() #w for wall
                        #This works better for our bounding box
                        floor   = open(path+"f.obj").readlines() #f for floor
                        w_points = getPoints(walls)
                        #Now we get the walls
                        height = [np.min(w_points,axis = 0)[up],np.max(w_points,axis = 0)[up]]
                        wall_points = wallBoundingBox(getPoints(floor),up)
                        wt = createWallTransform(wall_points,up,front,height)
                        for wall in wt:
                                items.append(FurnitureItem("Wall",wall["bounding_box"],wall["transform"]))
                except:
                       pass
                return items
                
        def parseHouse(self,house_file):
                '''The main json type for us is the house, which contains one or more rooms'''
                house = json.loads(open(self.path+"/"+house_file).read())
                house_path = house_file.split("/")[0] #Gets the folder name
                #They parse out the front and up vector, so we will as well
                front = np.argwhere(np.array(house['front']) == 1)[0][0]
                up    = np.argwhere(np.array(house['up']) == 1)[0][0]
                scale = house['scaleToMeters']#Provides our multipler for positiion and bounding boxes
                conversion = (front,up,scale)
                #Each 'room' is inside the levels
                result = {}
                for level in house['levels']:
                        #Get all the levels inside the room
                        rooms = filter(lambda x:x['type'] == 'Room',level['nodes'])
                        for room in rooms:
                                #We assume that all are singular for now
                                if'roomTypes' in room and len(room['roomTypes']) > 0:
                                        rtype = "_".join(room['roomTypes']) #For now, we assume they are an open concept ;)
                                else:
                                        rtype = 'Unknown'
                                furniture = []#This is our room furniture list
                                if 'nodeIndices' in room:
                                        for node in room['nodeIndices']:
                                                node_data = level['nodes'][node]
                                                if 'modelId' in node_data and 'bbox' in node_data and 'transform' in node_data:
                                                        label = self.object_mapping[node_data['modelId']]#Label of the item
                                                        fi = FurnitureItem(label,node_data['bbox'],node_data['transform'],conversion)
                                                        if fi.size is not None:
                                                                furniture.append(fi)
                                if len(furniture) > 0:
                                        #Add in different items based on the levels bounding box
                                        #try:
                                        bounds = self.buildBounds(house_path +"/"+room['modelId'],front,up)
                                        if not self.sep_rooms:
                                                furniture.extend(bounds)
                                        #except:
                                        #        pass
                                        #        #print (house)
                                        new_room = FrameData(furniture,rtype,self.sort)
                                        if self.sep_rooms:
                                                new_room.setWalls(bounds)
                                        if rtype not in result:
                                                result[rtype] = [new_room]
                                        else:
                                                result[rtype].append(new_room)
                return result
        def parseHouseType(self,house_file):
                '''This simply returns the type of the house. We are going to use this for shelling the processing of the houses'''
                house = json.loads(open(self.path+house_file).read())
                #Each 'room' is inside the levels
                result = []
                for level in house['levels']:
                        #Get all the levels inside the room
                        rooms = filter(lambda x:x['type'] == 'Room',level['nodes'])
                        for room in rooms:
                                #We assume that all are singular for now
                                if'roomTypes' in room and len(room['roomTypes']) > 0:
                                        rtype = "_".join(room['roomTypes']) #For now, we assume they are an open concept ;)
                                else:
                                        rtype = 'Unknown'
                                result.append(rtype)
                return result

#############################################################################################################################
##These are our access functions
def gsfscgHelper(parser,frame,file):
        res = parser.parseHouse(file)
        if frame is None: #We pass in None for frame when we want back everything
                return res
        if frame in res: #We can be too careful
                return res[frame]
        return []

def getFramesSUNCG(folders,frames = defaultdict(list),parser = None):
        if parser is None:
                parser = SUNCGParser("D:/downloaded code/SUNCGtoolbox/metadata/ModelCategoryMapping.csv")
        names = [f+"/house.json" for f in folders]
        phelper = partial(gsfscgHelper,parser,None)
        p = Pool(NUM_THREADS)
        results = p.map(parser.parseHouse,names) #Writes it as a generator
        p.close()
        p.join()
        for res in results:
                for r in res:
                        frames[r].extend(res[r])
        return frames

def parseSUNCG(folders,frames = defaultdict(list),parser = None):
        if parser is None:
                parser = SUNCGParser("D:/downloaded code/SUNCGtoolbox/metadata/ModelCategoryMapping.csv")
        #This is going to be a costly function unless you are only looking at a few rooms
        names = [f+"/house.json" for f in folders]
        p = Pool(NUM_THREADS)
        results = p.map(parser.parseHouseType,names)# [parser.parseHouseType(name) for name in names]
        p.close()
        p.join()
        for i in range(len(results)): #Results is a list of lists
                res = results[i]
                for r in res:
                        frames[r].append(names[i])
        return frames

def getSingleFrameSunCG(frame_file_paths,parser = None,keys = None):
        '''Provides back the furniture for a given room, including the frame type'''
        if keys is None:
                keys = sorted(list(frame_file_paths.keys()))
        for frame in keys:
                phelper = partial(gsfscgHelper,parser,frame)
                p = Pool(NUM_THREADS)
                room_samples = p.map(phelper,frame_file_paths[frame])
                p.close()
                p.join()
                room_samples = [item for sublist in room_samples for item in sublist] #Basically flatten the list
                yield (frame,room_samples)#We yield it so that we only need to store one room type at a time. We process it, then we move on

if __name__ == "__main__":
        import os
        parser = SUNCGParser("D:/downloaded code/SUNCGtoolbox/metadata/ModelCategoryMapping.csv")
        house_path = "../../../data/SUNCG/house"
        paths = [f for f in os.listdir(house_path) if not os.path.isfile(os.path.join(house_path,f))]
        scenes = {}
        for p in paths[0:1]:
                whole_path = os.path.join(house_path,p)+"/house.json"
                res = parser.parseHouse(whole_path)
                for r in res:
                        if r not in scenes:
                                scenes[r] = []
                        scenes[r].extend(res[r])
        #Lets look at some stats
        for scene in scenes:
                print(scene,len(scenes[scene]))
