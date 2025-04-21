#!/usr/bin/python

import copy
import random


def t(l,d,u,out):
#constructs a random tree T with l leaves plus the  cycle 12...l1 on the leaves 
# d+1 is an upper bound on the degree of any vertex in T
        
   


    # prepare an array to store leaves and tree vertices
   

    M=l*[0]
    for i in range (l):
        M[i]=i
      

    Edges =[]   # make an empty list of edges
    Leaf=[]
    
    p=0    # points to first non-covered vertex
    q=l  # points just after last vertex so always q-p uncovered vertices

    while q-p>3:   # still at least 4 uncovered vertices
        s=min(q-p,d)
        #r = int(random()*s)  # find degree-1 of new tree vertex
        r = random.randint(1,s)  # find degree-1 of new tree vertex
        #if r==0: r=1
                             # make a new tree vertex
        M=M+[q]
        #print r
        for i in range(r):
            j=random.randint(0,q-p-1)
            #if j==q-p: j=q-p-1
            
             # add the edge [q,M[p+j],1]
            Edges=Edges+[[q,M[p+j],0]]

            if M[p+j]<l: # a leaf is found
                Leaf=Leaf+[M[p+j]]  # add to leaf list
            # swap M[p] and M[p+j] and advance p by one
            x=M[p]
            M[p]=M[p+j]
            M[p+j]=x
            p=p+1
        q=q+1
    #print q-p
    
    # now q<=p-3
    if q-p>1:
        M=M+[q]    # make last vertex of T
        # and join it to the tree vertices just in front of it
        for i in range(q-p):
            Edges=Edges+[[q,M[q-i-1],0]]
            if M[q-i-1]<l:
                Leaf=Leaf+[M[q-i-1]]
        n=q+1
    else:
        n=q
    

    #add cycle on leaves in the order they were chosen
    for i in range(l-1):
        Edges=Edges+[[Leaf[i],Leaf[i+1]]]
    Edges=Edges+[[Leaf[l-1],Leaf[0]]]

    #add weights to edges. Weight = 1 if u=1 otherwise random in interval [1..l]

    e=len(Edges)   # total number of edges
    if u==1:
        for i in range(l):
            Edges[e-1-i]=Edges[e-1-i]+[1]
    if u==0:
        for i in range(l):
            Edges[e-1-i]=Edges[e-1-i]+[random.randint(1,l)]
   
    #print Edges
    fd = open(out,"w")
    fd.write("p cycle "+str(n)+" "+str(e)+'\n')
    #fd.write("dummy line"+'\n')
    for c in range(e):
        E=""
        f=Edges[c]  
        if c<e-l:
            E="t "+str(f[0]+1)+" "+str(f[1]+1)
        else:
            E="e "+str(f[0]+1)+" "+str(f[1]+1)+" "+str(f[2])
        fd.write(E+"\n")
    fd.close()
    #print 'ciao'
    return


#
#for x in [80,400,800]:
#    for i in range(10):
#        t(800,x,0,"C-800-"+str(x)+"-0-"+str(i+1)+".ins")
#    
#
for x in range (2,800):
    random.seed(x)
    t(800,x,0,"inst/out"+str(x)+".ins")
#
#
#for x in range (1,400):
#    random.seed(x+1)
#    t(400,x+1,0,"inst/out"+str(x+1)+".ins")
#
#
#for x in range (1,800):
#    random.seed(x+1)
#    t(800,x+1,0,"inst/out"+str(x+1)+".ins")
