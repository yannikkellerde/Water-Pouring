import partio,sys,os
import numpy as np

def partio_uncompress(dirname):
    for f in os.listdir(dirname):
        if not f.endswith(".bgeo"):
            continue
        p = partio.read(os.path.join(dirname,f))
        partio.write(os.path.join(dirname,f),p)

def partio_write_rigid_body(vertices,filename):
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    particleSet=partio.create()
    P=particleSet.addAttribute("position",partio.VECTOR,3)
    id=particleSet.addAttribute("id",partio.INT,1)
    particleSet.addParticles(len(vertices))
    for i,vertex in enumerate(vertices):
        particleSet.set(P,i,[float(x) for x in vertex])
    partio.write(filename,particleSet)

def remove_particles(infile,outfile,keep_rate):
    particles = partio.read(infile)
    orig_attr = particles.attributeInfo("position")
    orig_num_p = particles.numParticles()
    all_indices = np.arange(orig_num_p)
    keep_indices = np.random.choice(all_indices,size=int(orig_num_p*keep_rate),replace=False)

    new_particles = partio.create()
    P=new_particles.addAttribute("position",partio.VECTOR,3)
    id=new_particles.addAttribute("id",partio.INT,1)
    new_particles.addParticles(len(keep_indices))
    for index,i in enumerate(keep_indices):
        pos = particles.get(orig_attr,i)
        new_particles.set(P,index,pos)
    partio.write(outfile,new_particles)

def count_particles(filename):
    return partio.read(filename).numParticles()

def evaluate_partio(dirname):
    files = os.listdir(dirname)
    files.sort(key=lambda x:-int(x.split(".")[0].split("_")[-1]))
    filename = os.path.join(dirname,files[0])
    particles = partio.read(filename)
    pos_attr = particles.attributeInfo(0)

    res = {
        "spilled":0,
        "bottle":0,
        "glas":0,
        "frames":len(files)
    }
    for i in range(particles.numParticles()):
        pos = particles.get(pos_attr,i)
        if pos[1]<0:
            res["spilled"]+=1
        elif abs(pos[0])<0.3 and abs(pos[2])<0.3 and pos[1]<0.6:
            res["glas"]+=1
        else:
            res["bottle"]+=1
    return res

if __name__=="__main__":
    #print(evaluate_partio("../partio"))
    partio_uncompress(sys.argv[1])