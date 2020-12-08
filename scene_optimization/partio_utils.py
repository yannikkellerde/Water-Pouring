import partio,sys,os

def partio_uncompress(dirname):
    for f in os.listdir(dirname):
        p = partio.read(os.path.join(dirname,f))
        partio.write(os.path.join(dirname,f),p)

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