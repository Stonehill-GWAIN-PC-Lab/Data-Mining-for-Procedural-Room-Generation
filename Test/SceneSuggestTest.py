
import src.Parse.SUNRGBD as SUNRGBD
#This function used to exist inside the scenesuggest function
#so this saves a bit of time
from src.Methods.SceneSuggest import * 

path_to_data = "../../" #Directs to the root directory of your data-set

def sunRGBDDataMiningKermani( write_type = 'w'):
    '''max_amount controls the maximum number of rooms we look out, which helps us bound the problem
       starting_location tells us that there are rooms in the beginning that we can skip, most likely because we've already looked at them
       location_amount tells us to only look at a certain number of rooms, again so that we can run this in stages
       removed_rooms controls for our noisier rooms which may not give us good data. As an idea, if I was copying Kermani et al., I would remove all rooms except bedroom. Deep convo priors would be all rooms but living, bedroom, and office, etc.
       min_amount is another room remover. We determine a room threshold where we believe that we cannot get good data under the threshold
       write_type tells us if we are overwriting the file or appending.
    '''
    

    #File structure of the sun rgbd data-set
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
    sunRGBDDataMiningKermani()
