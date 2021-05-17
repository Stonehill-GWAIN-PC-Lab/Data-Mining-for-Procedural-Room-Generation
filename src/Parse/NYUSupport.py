#It's a matlab file that needs to be cleaned up
from scipy.io import loadmat
import h5py,math

from collections import Counter,defaultdict
import numpy as np
from .. import ObjectMetrics

#Return back a parsed list of Support frequency connections from the nyu_depth_v2_labeled matlab matrix.
#That matrix was converted from the originial found at https://cs.nyu.edu/~silberman/datasets/
def readMat(matPath): #10 percent in the paper
        f = h5py.File(matPath+"nyu_depth_v2_labeled.mat",'r')
        rooms = f['labels']
        def conv1(x):
                maximum = len(first_change)
                if x[0] > 0 and x[0] < maximum:
                       x[0] = first_change[x[0]]
                if x[1] > 0 and x[1] < maximum:
                       x[1] = first_change[x[1]]
                return x
        def convert(x):
                if x[2] == 1:
                        x[2] = 'Vertical-Support'
                elif x[2] == 2:
                        x[2] = 'Horizontal-Support'
                else:
                        x[2] = 'Support'
                if x[0] in names_to_id:
                        x[0] = names_to_id[x[0]]
                else:
                        x[0] = 'room'
                if x[1] in names_to_id:
                        x[1] = names_to_id[x[1]]
                else:
                        x[1] = 'room'
                return x
        places = open(matPath + 'scenes.dat','r').readlines()
        places = [i.strip().split('$') for i in places]
        ps = {}
        ret_data = defaultdict(list)
        for x in places:
                ps[int(x[0])] = x[1]
        exitnames = open(matPath + 'names.dat','r').readlines()
        exitnames = [i.strip().split('$') for i in exitnames]
        names_to_id = {}
        for i in exitnames:
                names_to_id[int(i[1])] = i[0]
        support = loadmat(matPath + 'support_labels.mat')['supportLabels']

        #This loads the actual support code
        for i in range(len(support)):
                first_change = np.unique(rooms[i,:,:])
                x = support[i][0].tolist()
                place = ps[i]
                x = list(map(lambda j:conv1(j),x)) #First, convert to correct names per scene
                x = list(map(lambda j:convert(j),x))
                #if place == "bedroom":
                #        print (x)
                if place in ret_data:
                        ret_data[place].append(x)
                else:
                        ret_data[place] = [x]
        return ret_data

def cleanRules(scenes):
    #Returns a cleaner rule-set where we pre-remove some noise
    #This can be expanded into a better human-in-the-loop system
    r = {}
    for place in sorted(scenes.keys()):
        result = []
        for scene in scenes[place]:
            res = []
            for j in scene:
                #print (j)
                if 'None' not in j:
                    if j[2] == 'Vertical-Support' and 'wall' in j and not('floor' in j or 'room' in j):
                        pass
                        #print("did not add",j)
                    else:
                        res.append(j)
                else:
                    print("did not add",j)
            result.append(res)
        r[place] = result
    return r

def getSupportTypes(scenes):
    #Quick helper function to get the types we have for a given scene
    types = []
    for scene in scenes:
        for j in scene:
            if j[2] not in types:
                types.append(j[2])
    return types

def supportFrequency(scenes, threshold = 0.1, debug = False):
    ''' Determines the support factors per scene'''
    counts = {}
    frequency = Counter()
    thresh = len(scenes)*threshold

    for i in scenes:
        freq = Counter()
        for j in i:
            if 'None' not in j:
                freq[j[0]] +=1
                freq[j[1]] +=1
        for f in freq:
            frequency[f] +=1
    if debug:
        for i in scenes:
            #first = i.split("$")[1]
            #second = i.split("$")[2]
            #if frequency[first] > thresh and frequency[second] > thresh:
            for j in i:
                print(j)
        print (frequency)
    #Then, we run through counts getting only the ones that are higher than the threshold
    #set_freq   = set([str(i) for i in frequency if frequency[i] > thresh])
    #ret_counts = [i.split("$")+[str(counts[i])] for i in counts if counts[i] > s_thresh and i.split("$")[1] in set_freq and i.split("$")[2] in set_freq]
    ret_freq   = [["frequency",str(i),str(frequency[i])] for i in frequency if frequency[i] > thresh]
    return ret_freq

def supportMining(scenes,good_objects,threshold = 0.1):
    '''Performs frequent pattern mining on the directed acyclic graphs for support'''
    min_gap = math.ceil(len(scenes) * threshold) #If we dont' have our single object at the minimum threshold, we won't have any larger support
    counts = Counter()
    for scene in scenes:
        for i in [(edge[1],edge[0]) for edge in scene if edge[0] in good_objects and edge[1] in good_objects]:
            counts[i] +=1
    return [(pair[0],pair[1],str(counts[pair])) for pair in counts if counts[pair] >= min_gap]

def getSupportTypes(scenes):
    #Quick helper function to get the types we have for a given scene
    types = []
    for scene in scenes:
        for j in scene:
            if j[2] not in types:
                types.append(j[2])
    return types

def buildSupportValues(scene,support):
    #Returns a key-value dict that only has the support value in question
    result = []
    for i in scene:
        res = []
        for j in i:
            if j[2] == support:
                res.append([j[0],j[1]])
        if len(res) > 0:
           result.append(res)
    return result

def mineNYU2Data(path_to_data,output_path):
    ''' '''
    mat_path = path_to_data+"data/sunrgbd/nyu_2_mat/"
    debug = False
    support = readMat(mat_path)
    support = cleanRules(support)#Cleans up some of our known noisy data that we shouldn't have
    with open(output_path,'w') as fi:
        for place in sorted(support.keys()):
            print (place)
            fi.write(','.join([place,str(len(support[place]))])+'\n')
            support_types = getSupportTypes(support[place])
            #print (support[place])
            rf = supportFrequency(support[place],0.1)
            for i in rf: #This is each item that was high enough support
                fi.write(','.join([ObjectMetrics.cleanup_word(str(j)) for j in i])+'\n')
            #In rc, we split up into our different factors. That will allow us to search each graph independently
            for s in support_types:
                 prox = supportMining(buildSupportValues(support[place],s),[i[1] for i in rf],0.1)
                 for item in prox:
                     fi.write(','.join([ObjectMetrics.cleanup_word(s)]+[ObjectMetrics.cleanup_word(i) for i in item])+'\n')
                 del prox
