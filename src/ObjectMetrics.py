import numpy as np

#Object Metrics is an agomeration of equations and metrics pulled from a couple of different scene data-mining sources.
#They are from:
#[1]Z. S. Kermani, Z. Liao, P. Tan, and H. Zhang, “Learning 3D Scene Synthesis from Annotated RGB‐D Images,” Computer Graphics Forum, vol. 35, no. 5, pp. 197–206, 2016.
#[2]Y. Liang, F. Xu, S.-H. Zhang, Y.-K. Lai, and T. Mu, “Knowledge graph construction with structure and parameter learning for indoor scene design,” Computational Visual Media, vol. 4, no. 2, pp. 123–137, 2018.
#[3]M. Savva, A. X. Chang, and P. Hanrahan, “Semantically-enriched 3D models for common-sense knowledge,” presented at the Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2015, pp. 24–31.


def cleanup_word(word):
    '''This is our cleanup process which fixes some of our hierarchy issues'''
    return '_'.join([w.lower() for w in word.split(' ')])

def getClosestPoints(obj1,obj2):
    #Finds the closest points between two objects based on the bounding boxes
    #First, we start with the center and determine the direction obj2 is from obj1
    obj1_p    =  obj1.centroid #no deep copy needed
    obj2_p    =  obj2.centroid
    direction = (obj1_p -obj2_p)**2
    #h_point and the sign of direction tell me which surface
    h_point = direction.argmax()
    direction = (obj1_p-obj2_p)[h_point]#Now, we figure out the surface
    #Now, we know the direction, and the surface. So we construct our positive and negative surface bounds
    positive = obj1_p + obj1.size
    negative = obj1_p - obj1.size
    clamped_point_1 = np.maximum(np.minimum(obj2_p,positive),negative)
    if direction < 0:
        clamped_point_1[h_point] = obj1_p[h_point] - obj1.size[h_point]
    else:
        clamped_point_1[h_point] = obj1_p[h_point] + obj1.size[h_point]
    #Now, we determine for object 2
    #Axis points is the same, we switch direction
    positive = obj2_p + obj2.size
    negative = obj2_p - obj2.size
    clamped_point_2 = np.maximum(np.minimum(obj1_p,positive),negative)
    if direction > 0:
        clamped_point_2[h_point] = obj2_p[h_point] - obj2.size[h_point]
    else:
        clamped_point_2[h_point] = obj2_p[h_point] + obj2.size[h_point]
    return (clamped_point_1,clamped_point_2)


def calculateDirection(point1,point2):
    #defined as the normalized cross product between two points
    #note that it must be greater than zero, so we normalize it
    return np.cross(point1,point2) / np.linalg.norm(np.cross(point1,point2))


def calculateProjectedPlacement(point1,point2,prim_direct):
    #Meant to model more detailed placement of objects
    x = point1 - point2
    return prim_direct * (x - prim_direct*x)


def theta(vec1,vec2):
    '''Calculates the cosine of the angle between two distance vectors'''
    dot = np.dot(vec1,vec2)
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    return dot /(mag1*mag2)

def orientationCosRelations(obj1,obj2,eps = 0.1):
    '''We are going to switch to an object centric view like in the matlab code for SUNRGBD.
    With that, we have precomputed orientations'''
    #Orientations are numpy vectors. We are looking at two normalized vectors
    #We know the length of each vector is 1 because we have normalized it
    th= np.linalg.norm(np.absolute(obj1.orientation - obj2.orientation))
    #print (obj1.orientation,obj2.orientation,th)
    return th < eps #If abs of cosine is close to 0, then we have two aligned objects


def orientationSinRelations(obj1,obj2,eps = 0.1):
    '''We are going to switch to an object centric view like in the matlab code for SUNRGBD.
    With that, we have precomputed orientations'''
    th= np.sin(obj1.orientation - obj2.orientation)
    return np.any(1-th< eps) #We really care about a sine on one dim (especially around the yaw)

def calculateRelativeOrientation(obj1,obj2,prim_direct):
    #For some reason, they use the primary direction as a relative response
    return (obj1.orientation - obj2.orientation)*prim_direct
    
def proximityCenter(obj1,obj2,value_array = None):
    '''Function defines our cost based on the proximity between two objects'''
    return np.sum((obj1.centroid-obj2.centroid)**2)

def proximityClosestPoint(obj1,obj2,value_array = None):
    '''Function provides the distance between the closest point of two objects.
    So, if they are touching, their distance between each other will be zero.
    Everything we use this for is relative, so squared distance is good enough'''
    points = getClosestPoints(obj1,obj2)
    return np.sum((points[0] - points[1])**2)

def proximityPrimaryDistance(obj1,obj2,value_array = None):
    '''Liang et al. designates a distance between two points as the
    manhatten distence times the primary direction. '''
    points = getClosestPoints(obj1,obj2)
    return (points[0] - points[1]) * cacluateDirection(points[0],points[1])

def symmetryRelations(obj1,obj2,eps = 0.1):
    #In Kermani, they talk about object category. Going to assume that it is the labels and not something on a taxonomy
    if obj1.label != obj2.label: #Means things are only ever symmetrical with themselves
        return False
    #print (obj1.label,np.absolute(obj1.size - obj2.size),obj1.size,obj2.size)
    return (np.sum(np.absolute(obj1.size - obj2.size)) < eps) #EPS here means similar sizes, so our default eps has a greater fault tolerance

#This is our Support and attachment connections
def supportMiningBoundingBox(obj1,obj2,eps = 0.01,debug = False):#Saava did 1cm for there support hierarchy (Semantically-Enriched 3D models)
    '''Performs a support for two objects in the scene. Here, we aren't
    concerned with if it is horizontal or vertical, just that there is a
    support between them. We also figure out the surface based on which index size is bigger'''
    projections = np.absolute(obj1.centroid - obj2.centroid)
    distances   = np.absolute(obj1.size + obj2.size) / 2
    test        = projections - distances
    if debug:
        print("objects:",obj1.label,obj2.label)
        print("projections:",obj1.centroid,obj2.centroid,projections)
        print("distances:",obj1.size,obj2.size,distances,distances**2)
        print("test",test)
    #This is saying if the distance between them is less than the total size,
    #then we have an overlap
    if len(test[test < 0]) == 2:
        #Now, we determine if there is a touching on the third axis,
        #If there is, we have a support
        if np.any(test[test> 0] < eps): 
            #Now, we just figure out which is the larger surface
            return np.prod(obj1.size[test < 0]) >= np.prod(obj2.size[test < 0]) #Give the overall larger item the benefit of the doubt 
    return None #There is no support here

def supportMiningStatic(obj1,obj2,overlap = [0,1],touch = 2,eps = 0.01,debug = False):
    '''Returns none if the support doesn't exist. True if obj1 supports
    obj2, and False if obj2 supports obj1'''
    #First determine if they overlap in the x/y plane
    min_points = np.maximum(obj1.centroid-obj1.size/2,obj2.centroid-obj2.size/2)[overlap]
    max_points = np.minimum(obj1.centroid+obj1.size/2,obj2.centroid+obj2.size/2)[overlap]
    if np.prod(max_points-min_points) > 0:
        if debug:
            print ("testing",obj1,obj2)
            print ("positions:",obj1.centroid,obj2.centroid)
            print ("sizes:",obj1.size,obj2.size)
            print ("Overlap:",min_points,max_points,np.prod(max_points-min_points))
            print ("Touchness:",obj1.centroid[touch]+obj1.size[touch]/2 - obj2.centroid[touch]-obj2.size[touch]/2 ,obj2.centroid[touch]+obj2.size[touch]/2 -obj1.centroid[touch]-obj1.size[touch]/2)
        #Now, we determine if there is touching of obj1 under obj2
        if abs((obj1.centroid[touch]+obj1.size[touch]/2) -(obj2.centroid[touch]-obj2.size[touch]/2)) < eps:
            return True
        #If not, maybe it's the other way around (i.e, hanging instead)
        if abs((obj2.centroid[touch]+obj2.size[touch]/2) -(obj1.centroid[touch]-obj1.size[touch]/2)) < eps:
            return True
    return None

#Our object occurance names
#Here, we assume good_objects are known and we are figuring out the possible names
def frequencyGoodObjects(good_objects,scene):
    '''Determines object coOccurance for a given room'''
    seen_objects = []

    for obj in scene.annotation3D:
        if obj.label in good_objects: #Only care if we know it has a relationship
            if obj.label not in seen_objects:
                seen_objects.append(obj.label)
            else:
                #seen_objects[obj.label] += 1 #This deliniates our k for the scene
                count = 1
                name = obj.label
                while( name+"_"+str(count) in seen_objects):
                    count+=1
                #print (name,count)
                seen_objects.append(name+"_"+str(count))
    return seen_objects

def labelObjects(scene):
    '''If we do not have a list of good objects, then we have to
    count the frequency. Otherwise it is eqivalent
    to finding the frequency with good objects'''
    seen_objects = []
    for obj in scene.annotation3D:
        if obj.label not in seen_objects:
            seen_objects.append(obj.label)
        else:
            #seen_objects[obj.label] += 1 #This deliniates our k for the scene
            count = 1
            name = obj.label
            while name+"_"+str(count) in seen_objects:
                count+=1
            #print (name,count)
            seen_objects.append(name+"_"+str(count))
    return seen_objects
