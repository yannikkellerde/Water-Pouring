import partio,sys,os
import numpy as np

def partio_uncompress(dirname):
    """Take a folder of compressed .bgeo files and uncompress those files.
    This can come in handy because SPlisHSPlasH stores it's particle data
    as compressed .bgeo files.

    Args:
        dirname: Path to a directory of .bgeo files.
    """
    for f in os.listdir(dirname):
        if not f.endswith(".bgeo"):
            continue
        p = partio.read(os.path.join(dirname,f))
        partio.write(os.path.join(dirname,f),p)

def partio_write_rigid_body(vertices,filename):
    """Write vertices of an object into a .bgeo file.

    Args:
        vertices: A num_vertices x 3 numpy array.
        filename: Path to the .bgeo file to create.
    """
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    particleSet=partio.create()
    P=particleSet.addAttribute("position",partio.VECTOR,3)
    id=particleSet.addAttribute("id",partio.INT,1)
    particleSet.addParticles(len(vertices))
    for i,vertex in enumerate(vertices):
        particleSet.set(P,i,[float(x) for x in vertex])
    partio.write(filename,particleSet)

def remove_particles(infile,outfile,keep_rate):
    """Remove a percentage of particles from a .bgeo file.

    Args:
        infile: Path to input .bgeo file.
        outfile: The file where the reduced number of particles will be stored.
        keep_rate: A float between 0 and 1 that describes how much percentage of
                   particles will be kept in the new file.
    """
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
    """Count the number of particles in a .bgeo file.
    """
    return partio.read(filename).numParticles()

if __name__=="__main__":
    #print(evaluate_partio("../partio"))
    partio_uncompress(sys.argv[1])