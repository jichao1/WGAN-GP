import os
from multiprocessing import Pool
import numpy as np
import flopy 
from shutil import copy2, rmtree
import flopy.utils.binaryfile as bf


class Model:
    def __init__(self, params = None):
        self.idx = 0
        self.homedir = os.path.abspath('./')
        self.deletedir = True
        self.outputdir = None
        self.parallel = False
        self.record_cobs = False


        from psutil import cpu_count  # physcial cpu counts
        self.ncores = cpu_count(logical=False)

        if params is not None: 
            if 'deletedir' in params:
                self.deletedir = params['deletedir']
            if 'homedir' in params:
                self.homedir = params['homedir']
            if 'ncores' in params:
                self.ncores = params['ncores']
            if 'outputdir' in params:
                self.outputdir = params['outputdir']
            if 'parallel' in params:
                self.parallel = params['parallel']
            
            self.Lx = params['Lx']
            self.Ly = params['Ly']
            self.Q  = params['Q']
            self.Rch  = params['Rch']
            self.nlay = params['nlay']
            self.nrow = params['nrow']
            self.ncol = params['ncol']
            self.ztop = params['ztop']
            self.zbot = params['zbot']
            self.obs_locmat = params['obs_locmat']
            self.Q_locs = params['Q_locs']
            

    def create_dir(self,idx=None):
        
        mydirbase = "./simul/simul"
        if idx is None:
            idx = self.idx
        
        mydir = mydirbase + "{0:04d}".format(idx)
        mydir = os.path.abspath(os.path.join(self.homedir, mydir))
        
        if not os.path.exists(mydir):
            os.makedirs(mydir)

        return mydir

    def run_model(self, s, idx=0):
        # Assign name and create modflow model object
        sim_dir = self.create_dir(idx)
        
        while not os.path.exists(sim_dir): 
            sim_dir = self.create_dir(idx)
            
        Lx = self.Lx; Ly = self.Ly
        Q_locs = self.Q_locs; Q = self.Q; Rch = self.Rch
        nlay = self.nlay; nrow = self.nrow; ncol = self.ncol
        ztop = self.ztop; zbot = self.zbot
        HK = (np.power(10,s)).reshape(nlay,nrow,ncol)   # hydraulic conductivity 
        

        delr = Lx / ncol
        delc = Ly / nrow
        delv = (ztop - zbot) / nlay
        botm = np.linspace(ztop, zbot, nlay + 1)
        
        modelname = "Test"
        model_ws = sim_dir
        mymf = flopy.modflow.Modflow(modelname, exe_name="mf2005", model_ws = model_ws)

        dis = flopy.modflow.ModflowDis(mymf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm[1:])
        
        ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
        ibound[:, :, 0] = -1
        ibound[:, :, -1] = -1
        strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
        strt[:, :, 0] = 0.
        strt[:, :, -1] = -10.
        bas = flopy.modflow.ModflowBas(mymf, ibound=ibound, strt=strt)
        
        lpf = flopy.modflow.ModflowLpf(mymf, hk=HK, vka=HK, ipakcb=53)
        
        flopy.modflow.mfrch.ModflowRch(mymf, nrchop=3, rech=Rch)
        
        pcg = flopy.modflow.ModflowPcg(mymf)
        
        simul_obs = []
        for Q_loc in Q_locs:
            obs_locmat = np.copy(self.obs_locmat)
            # Create the well package
            # Remember to use zero-based layer, row, column indices!
            wel_sp = [[Q_loc[0], Q_loc[1], Q_loc[2], Q]]  # lay, row, col index, pumping rate
            stress_period_data = {0: wel_sp}  # define well stress period {period, well info dictionary}
            wel = flopy.modflow.ModflowWel(mymf, stress_period_data=stress_period_data)
            spd = {(0, 0): ['save head']}
            oc = flopy.modflow.ModflowOc(mymf, stress_period_data=spd, compact=True)
            
            while not os.path.exists(sim_dir):
                sim_dir = self.create_dir(idx)
            
            # Write the model input files
            mymf.write_input()

            # Run the model
            success, buff = mymf.run_model(silent=True)
            if not success:
                raise Exception("MODFLOW did not terminate normally.")
                               
            # Post process the results
            hds = bf.HeadFile(model_ws+'/'+modelname + '.hds')
            times = hds.get_times()
                
            head = hds.get_data(totim=times[-1])
            obs_locmat[Q_loc] = False # don't count head at pumping well
            temp_obs = head[obs_locmat]
            temp_obs = temp_obs.reshape(-1,1)
            simul_obs.append(temp_obs)
            
            if self.deletedir:
                rmtree(sim_dir, ignore_errors=True)
            
        simul_obs = np.array(simul_obs).reshape(-1,1)
        return simul_obs

    def run(self,s,par,ncores=None):
        if ncores is None:
            ncores = self.ncores

        method_args = range(s.shape[0])
        args_map = [(s[arg:arg + 1,:,:], arg) for arg in method_args]

        if par:
            print('parallel run with ncores = %d' % ncores)
            pool = Pool(processes=ncores)
            simul_obs = pool.map(self, args_map)
        else:
            simul_obs =[]
            for item in args_map:
                simul_obs.append(self(item))

        return np.array(simul_obs).T

        #pool.close()
        #pool.join()

    def __call__(self,args):
        return self.run_model(args[0],args[1])


