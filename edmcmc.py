import numpy as np
import time
import pdb

class mcmcstr:
    def __init__(self, chains, flatchains, allneglogl, flatneglogl, whichwalker, whichlink,
                fullchains, fullneglogl, nburnin, nlink, nwalkers, npar, acceptancerate):
        self.acceptancerate = acceptancerate
        self.chains = chains
        self.flatchains = flatchains
        self.allneglogl = allneglogl
        self.flatneglogl = flatneglogl
        self.whichwalker = whichwalker
        self.whichlink = whichlink
        self.fullchains = fullchains
        self.nburnin = nburnin
        self.nlink = nlink
        self.nwalkers = nwalkers
        self.npar = npar
        self.lastpos = fullchains[:,nlink-1,:]
        self.fullneglogl = fullneglogl
        mnll = np.unravel_index(np.argmin(fullneglogl, axis=None), fullneglogl.shape)
        self.bestpar = fullchains[mnll[0],mnll[1],:]
        
    def get_chains(self, nburnin = None, nthin = 1, flat=False, returnDiag=False):
        if nburnin == None: 
            nburnin = self.nburnin
        indices = nburnin + np.arange(np.floor((self.nlink - nburnin)/nthin)) * nthin
        cutchains = self.fullchains[:,[int(index) for index in indices],:]
        
        if returnDiag:
            cutneglogl = self.fullneglogl[:,[int(index) for index in indices]]
            fullwhichwalker = np.transpose(np.outer(np.ones(self.nlink), np.arange(self.nwalkers)))
            cutwhichwalker = fullwhichwalker[:,[int(index) for index in indices]]
            fullwhichlink = np.outer(np.ones(self.nwalkers), np.arange(self.nlink))
            cutwhichlink = fullwhichlink[:,[int(index) for index in indices]]
        
        if not flat: 
            if not returnDiag: 
                return cutchains
            if returnDiag:
                return cutchains,cutwhichlink, cutwhichwalker, cutneglogl
        if flat: 
            cutflatchains = np.zeros((self.nwalkers * (len(indices)), self.npar))
            if returnDiag: 
                cutflatneglogl = np.zeros(self.nwalkers * (len(indices)))
                cutflatwhichwalker = np.zeros(self.nwalkers * (len(indices)))
                cutflatwhichlink = np.zeros(self.nwalkers * (len(indices)))
                
            for i in range(self.nwalkers):
                for j in range(self.npar):
                    cutflatchains[i * (len(indices)):(i + 1)* (len(indices)), j] = cutchains[i,:,j]
                if returnDiag:
                    cutflatneglogl[i * (len(indices)):(i + 1)* (len(indices))] = cutneglogl[i,:]
                    cutflatwhichwalker[i * (len(indices)):(i + 1)* (len(indices))] = cutwhichwalker[i,:]
                    cutflatwhichlink[i * (len(indices)):(i + 1)* (len(indices))] = cutwhichlink[i,:]
            if not returnDiag: 
                return cutflatchains 
            if returnDiag:
                return cutflatchains, cutflatwhichlink, cutflatwhichwalker, cutflatneglogl#, ...
    
    def onegelmanrubin(self, chain): #taken from http://joergdietrich.github.io/emcee-convergence.html
        ssq = np.var(chain, axis=1, ddof=1)
        W = np.mean(ssq, axis=0)
        thetab = np.mean(chain, axis=1)
        thetabb = np.mean(thetab, axis=0)
        m = chain.shape[0]
        n = chain.shape[1]
        B = n / (m - 1) * np.sum((thetabb - thetab)**2, axis=0)
        var_theta = (n - 1) / n * W + 1 / n * B
        rhat = np.sqrt(var_theta / W)
        return rhat
    
    def gelmanrubin(self, nburnin = None, nend = None):
        if nburnin == None: 
            nburnin = self.nburnin
        if nend == None:
            nend = self.nlink
        cutchains = self.fullchains[:,nburnin:nend,:]
        grstats = np.zeros(self.npar)
        for i in range(self.npar):
            grstats[i] = self.onegelmanrubin(cutchains[:,:,i])
        return grstats
    
def edmcmc(function, startparams_in, width_in, nwalkers=50, nlink=10000, nburnin=500, gamma_param=None, 
          method='loglikelihood',parinfo=None, quiet=False, pos_in=None, args = None, ncores=1, bigjump=False,
           m1mac=True,adapt=False, dispersion_param = 1e-2):
    #method can be loglikelihood, chisq, or mpfit
    #outputs = pnew, perror, chains, whichwalker, whichlink, allneglogl, 
    #pos_in is an array with size (nwalkers, npar) of starting positions. 
    if ncores ==1: ncorestouse = 1
    if ncores >1:
        if not m1mac: import multiprocessing as mp
        if m1mac: import multiprocess as mp
            
        numcorestotal = mp.cpu_count()
        ncorestouse = round(ncores)
        if ncorestouse > numcorestotal:
            print('Asked for more cores than exist on the machine, limiting to ' + str(numcorestotal))
            ncorestouse = numcorestotal
        if ncorestouse < 1:
            print('Asked for too few cores, reverting to 1')
            ncorestouse = 1


    
    possiblemethods = ['loglikelihood', 'negloglikelihood', 'chisq']
    if not method in possiblemethods: 
        method = 'loglikelihood'
        print('Method unrecognized, reverting to log likelihood')
        
    width = np.copy(width_in)
    startparams = np.copy(startparams_in)
    
    if len(width) != len(startparams): 
        print('Length of width array not equal to length of startparams. Returning')
        return 0
    
    if nburnin > nlink: 
        nburnin = np.floor(nlink/2.0)
        
    npar = len(startparams)
    
    if nwalkers < 3 * npar or nwalkers <10:
        print('Not enough walkers - increasing to 3x number of parameters or a minimum of 10.')
        nwalkers = max((3 * npar, 10))


    onedimension = False
    if npar == 1: 
        onedimension = True
        npar = 2
        startparams = np.append(startparams, 0)
        width = np.append(width, 0)
    
    limits = np.zeros((2,npar))
    limits[0,:] = -1*np.inf
    limits[1,:] = np.inf
    
    
    
    if parinfo != None: 
        if len(parinfo) != len(startparams): 
            print('Length of parinfo not equal to length of startparams. Returning')
            return 0
        
        for i in range(len(parinfo)): 
            pi = parinfo[i]
            if pi['fixed']:
                  width[i] = 0
            if pi['limited'][0]:
                limits[0,i] = pi['limits'][0]
            if pi['limited'][1]:
                limits[1,i] = pi['limits'][1]                
    
    nfree = np.sum(width !=0)
    
    llimits = np.tile(limits[0,:], (nwalkers, 1))
    ulimits = np.tile(limits[1,:], (nwalkers, 1))
    
    if gamma_param == None: gamma_param =  2.38/np.sqrt(2.0*nfree)
    

    
    
    position = np.zeros((nwalkers, (nlink), npar))    
    allneglogl = np.zeros((nwalkers, nlink))
    accepted = np.zeros((nlink)) + np.nan
    lastneglogl = np.zeros(nwalkers) + np.inf
    allneglogl[:,0] = lastneglogl
    thispos = np.zeros((nwalkers, npar))
    
    infs = np.zeros(nwalkers) + np.inf
    for i in range(nwalkers): 

        if np.all(pos_in == None): 
            counter = 0
            while lastneglogl[i] == np.inf:
                counter = counter + 1
                if counter % 1000000 == 0:
                    print("Can't find good starting parameters: Attempt number ", counter)
                thispos[i,:] = np.random.normal(0,1,npar) * width + startparams
                lowerlimit = thispos[i,:] < limits[0,:]
                upperlimit = thispos[i,:] > limits[1,:]
                if not any(lowerlimit.tolist() + upperlimit.tolist()):    
                    if args != None: 
                        output = function(thispos[i,:], *args)
                    if args == None: 
                        output = function(thispos[i,:])
                    if method == 'loglikelihood':
                        lastneglogl[i] = -1 * output
                    if method == 'negloglikelihood':
                        lastneglogl[i] =  output
                    if method == 'chisq':
                        lastneglogl[i] = 0.5 * output
                    
            
        if np.all(pos_in != None): 
            thispos[i,:] = pos_in[i,:]
            if args != None: 
                output = function(thispos[i,:], *args)
            if args == None: 
                output = function(thispos[i,:])
            if method == 'loglikelihood':
                lastneglogl[i] = -1 * output
            if method == 'negloglikelihood':
                lastneglogl[i] =  output
            if method == 'chisq':
                lastneglogl[i] = 0.5 * output
        position[i,0,:] = thispos[i,:]
        
        
    starttime = time.time()
    lastprinttime = starttime
    naccept = 0.0
    ntotal = 0.0
    ninstant = 10
    instantrate = np.nan
    
    randintbank1 = np.random.randint(0, nwalkers-1, (nlink, nwalkers))
    randintbank2 = np.random.randint(0, nwalkers-2, (nlink, nwalkers))
    normalbank = np.random.normal(0,1,(nlink, nwalkers, npar))
    uniformbank = np.random.uniform(0,1,(nlink, nwalkers))
    


    if ncorestouse > 1: P = mp.Pool(processes=ncorestouse)

    for i in range(1, nlink):
        js = randintbank1[i,:]#random.randint(nwalkers-1)
        ks = np.arange(nwalkers)
        js = js + (js >= ks)
        j2s = randintbank2[i,:]
        j2s = j2s + (j2s >= np.minimum(ks,js))
        j2s = j2s + (j2s >= np.maximum(ks,js))
        
        
        jthpos = position[js,i-1,:]
        j2thpos = position[j2s,i-1,:]
        
        if adapt and i % 10 == 9 and np.isfinite(instantrate): #Make adaptive changes to the gamma parameter to optimize the acceptance rate near 0.234. 
            ratedifference = instantrate / 0.234
            gamma_param = gamma_param * max([min([ratedifference, 1.5]), 0.5])

        
        thisgamma = gamma_param
        
            
        if bigjump and i % 10 == 9: thisgamma = 1
        newpars = position[:,i-1,:] + thisgamma * (1 + normalbank[i,:,:] * dispersion_param) * (j2thpos-jthpos)
        
        
        
        lowerlimits = newpars < llimits
        upperlimits = newpars > ulimits
        
        outofranges = np.logical_or(np.any(lowerlimits, axis=1), np.any(upperlimits, axis=1))
        
        outputs = np.zeros(nwalkers)
        theseneglogls = np.zeros(nwalkers)
        

        if ncorestouse > 1:
            tocalc = np.where(np.logical_not(outofranges))[0]
            inputs = []
            if args != None: 
                for k in tocalc:
                    inputs.append((newpars[k,:],) + args)
            if args == None: 
                for k in tocalc:
                    inputs.append((newpars[k,:],))
            
            
            outputs2 = P.starmap(function, inputs)       
            outputs[tocalc] = np.array(outputs2)

            


        
        if ncorestouse == 1: 
            tocalc = np.where(np.logical_not(outofranges))[0]

            if args != None: 
                for k in tocalc:
                        outputs[k] = function(newpars[k,:], *args)
            else:
                for k in tocalc:
                        outputs[k] = function(newpars[k,:])
        

        if method == 'loglikelihood':
            theseneglogls = np.choose(outofranges, (-1 * outputs,  infs))
        if method == 'negloglikelihood':
            theseneglogls = np.choose(outofranges, (outputs, infs))
        if method == 'chisq':
            theseneglogls = np.choose(outofranges, (0.5 * outputs, infs))


        
        qs = np.exp(lastneglogl - theseneglogls)
        rs = uniformbank[i,:]
        accept = np.transpose(np.tile(1*(rs <= qs),(npar,1)))
        thispos = np.choose(accept, (position[:,i-1,:], newpars))
        
        
        position[:,i,:] = thispos
        newneglogl = np.choose(accept[:,0], (lastneglogl, theseneglogls))
        lastneglogl = newneglogl
        allneglogl[:,i] = newneglogl
        naccept = naccept + np.sum(accept[:,0])
        accepted[i] = np.sum(accept[:,0])
        ntotal = ntotal + nwalkers
        
        thistime = time.time()
        tremaining = (thistime - starttime)/float(i) * (nlink - i - 1.0)
        days = np.floor(tremaining / 3600.0 / 24.0)
        hours = np.floor(tremaining / 3600.0 - 24 * days)
        minutes = np.floor(tremaining/60 - 24 * 60 * days - 60 * hours)
        seconds = (tremaining - 24 * 3600 * days - 3600 * hours - 60*minutes)
        if adapt and i > ninstant: 
            instantrate = round(np.sum(accepted[i-ninstant:i])/(nwalkers*ninstant),2)
            outstr = str(int(days)) + ' days, ' + str(int(hours)).zfill(2) + ':' + str(int(minutes)).zfill(2) + ':' + str(round(seconds, 1)).zfill(4) + ' remains. Link ' + str(i+1) + ' of ' +str(nlink) + '. Overall acceptance rate = ' + str(round(naccept/ntotal,2)) + ', instantaneous = ' + str(instantrate) 
            
        if (not adapt) or i < ninstant: 
            instantrate = np.nan
            outstr = str(int(days)) + ' days, ' + str(int(hours)).zfill(2) + ':' + str(int(minutes)).zfill(2) + ':' + str(round(seconds, 1)).zfill(4) + ' remains. Link ' + str(i+1) + ' of ' +str(nlink) + '. Acceptance rate = ' + str(round(naccept/ntotal,2))
        
        if not quiet and (thistime - lastprinttime > 0.01 or i == nlink -1):
            lastprinttime = thistime
            print(outstr,end="\r")
            if i == nlink -1:
                print(outstr)
        
    
    if ncorestouse > 1:
        P.close() 
        P.join()
    if not onedimension: 
        chainsout = position[:,nburnin:nlink,:]#np.zeros((nwalkers, (nlink - nburnin), npar))    
        flatchainsout = np.zeros((nwalkers * (nlink - nburnin), npar))
        allnegloglout = allneglogl[:, nburnin:nlink]#np.zeros((nwalkers, nlink - nburnin))
        flatnegloglout = np.zeros((nlink - nburnin)*nwalkers)
        whichwalkerout = np.zeros((nlink - nburnin)*nwalkers)
        whichlinkout = np.zeros((nlink - nburnin)*nwalkers)

        for i in range(nwalkers):
            for j in range(npar):
                flatchainsout[i * (nlink - nburnin):(i + 1)* (nlink - nburnin), j] = position[i,nburnin:nlink,j]
            flatnegloglout[i * (nlink - nburnin):(i + 1)* (nlink - nburnin)] = allneglogl[i,nburnin:nlink]
            whichwalkerout[i * (nlink - nburnin):(i + 1)* (nlink - nburnin)] += i
            whichlinkout[i * (nlink - nburnin):(i + 1)* (nlink - nburnin)] = np.arange(nburnin, nlink)
    if onedimension: 
        npar = 1
        chainsout = position[:,nburnin:nlink,0]#np.zeros((nwalkers, (nlink - nburnin), npar))    
        flatchainsout = np.zeros((nwalkers * (nlink - nburnin), 1))
        allnegloglout = allneglogl[:, nburnin:nlink]#np.zeros((nwalkers, nlink - nburnin))
        flatnegloglout = np.zeros((nlink - nburnin)*nwalkers)
        whichwalkerout = np.zeros((nlink - nburnin)*nwalkers)
        whichlinkout = np.zeros((nlink - nburnin)*nwalkers)


        for i in range(nwalkers):
            flatchainsout[i * (nlink - nburnin):(i + 1)* (nlink - nburnin), 0] = position[i,nburnin:nlink,0]
            flatnegloglout[i * (nlink - nburnin):(i + 1)* (nlink - nburnin)] = allneglogl[i,nburnin:nlink]
            whichwalkerout[i * (nlink - nburnin):(i + 1)* (nlink - nburnin)] += i
            whichlinkout[i * (nlink - nburnin):(i + 1)* (nlink - nburnin)] = np.arange(nburnin, nlink)
    

    
    return(mcmcstr(chainsout, flatchainsout, allnegloglout, flatnegloglout, whichwalkerout, 
                   whichlinkout, position, allneglogl, nburnin, nlink, nwalkers, npar, acceptancerate=naccept/ntotal))

