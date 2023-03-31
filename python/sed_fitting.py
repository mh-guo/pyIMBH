import numpy as np
from scipy.interpolate import interp1d

class SED_Fitting():
    def __init__(self,data,template,filter_funs,
                 band_keys,band_to_filter,band_to_err,filter_keys):
        self.data=data
        self.template=template
        self.filter_funs=filter_funs
        self.band_keys=band_keys
        self.band_to_filter=band_to_filter
        self.band_to_err=band_to_err
        self.filter_keys=filter_keys
        self.filter_norms={}
        self.con_fr_sed={}
        self.sed_0={}
        self.sed_0['x']=self.template[0]
        self.sed_0['Fnu']=self.template[1]
        return
    def config(self):
        self.distill_()
    def logl2_(self,y,y0):
        b=np.mean(np.log10(y)-np.log10(y0))
        return np.mean((np.log10(y)-np.log10(y0)-b)**2)
    def logl2_a_(self,y,y0):
        b=np.mean(np.log10(y)-np.log10(y0))
        return np.mean((np.log10(y)-np.log10(y0)-b)**2),10**b
    def l2_err_(self,y,y0,err):
        y,y0,err=np.asarray(y),np.asarray(y0),np.asarray(err)
        a=np.sum(y0*y/err**2)/np.sum(y0**2/err**2)
        return np.mean(((y-y0*a)/err)**2)
    def l2_err_a_(self,y,y0,err):
        y,y0,err=np.asarray(y),np.asarray(y0),np.asarray(err)
        a=np.sum(y0*y/err**2)/np.sum(y0**2/err**2)
        return np.mean(((y-y0*a)/err)**2),a
    def sed_z_(self,z,z_0):
        sed_0=self.sed_0
        sed={}
        sed['x']=sed_0['x']*((1+z)/(1+z_0))
        sed['Fnu']=sed_0['Fnu']*((1+z)/(1+z_0))
        #speed_of_light=1
        sed['Fx']=sed['Fnu']/sed['x']**2
        return sed
    def sed_to_func(self,sed):
        return interp1d(sed['x']*1e4,sed['Fnu']/(sed['x']*1e4)**2,fill_value="extrapolate")
    def ref_x2Fx(self,x):
        unity=interp1d([-1,1],[1,1],fill_value="extrapolate")
        lin=interp1d([-1,1],[-1,1],fill_value="extrapolate")
        return unity(x)/lin(x)**2
    def distill_(self):
        data=self.data
        Y=self.Y={}
        for k in ['ID','z_best']:
            Y[k]=np.array([data[k][i] for i in range(data['ID'].shape[0])])
        Y['band_names']=[[] for i in range(data['ID'].shape[0])]
        Y['filter_names']=[[] for i in range(data['ID'].shape[0])]
        Y['err_names']=[[] for i in range(data['ID'].shape[0])]
        Y['data']=[{} for i in range(data['ID'].shape[0])]
        for i in range(data['ID'].shape[0]):
            dat=data[i]
            for bn in self.band_keys:
                if(not(np.isnan(dat[bn]) or dat[bn]<=0)):
                    Y['band_names'][i].append(bn)
                    Y['filter_names'][i].append(self.band_to_filter[bn])
                    Y['err_names'][i].append(self.band_to_err[bn])
                    Y['data'][i][bn]=dat[bn]
    def fitting_(self,dat,sed,filter_names,band_names):
        sed_x2Fx=self.sed_to_func(sed)
        y,y0=[],[]
        for fn,bn in zip(filter_names,band_names):
            fr=self.filter_funs[fn]
            norm=self.filter_norms[fn]
            #tmp=fr.convolve_with_function(ref_x2Fx)
            #print(fn,norm,tmp,)
            y0.append(fr.convolve_with_function(sed_x2Fx)/norm)
            y.append(dat[bn])
        return self.logl2_a_(y,y0)
    def compute_(self):
        Y=self.Y
        template=self.template
        sed_0={}
        sed_0['x']=template[0]
        sed_0['Fnu']=template[1]
        filter_norms=self.filter_norms
        for k in self.filter_keys:
            fr=self.filter_funs[k]
            filter_norms[k]=fr.convolve_with_function(self.ref_x2Fx)
        self.metrics=np.ones(Y['ID'].shape[0])
        self.norms=np.ones(Y['ID'].shape[0])
        for i in range(Y['ID'].shape[0]):
            dat=Y['data'][i]
            sed=self.sed_z_(z=Y['z_best'][i],z_0=0.0)
            self.metrics[i],self.norms[i]=self.fitting_(dat,sed,
                            filter_names=Y['filter_names'][i],
                            band_names=Y['band_names'][i])
        return self.metrics,self.norms
    def fitting_z_(self,dat,z,filter_names,band_names):
        y,y0=[],[]
        con_fr_sed=self.con_fr_sed
        for fn,bn in zip(filter_names,band_names):
            y0.append(con_fr_sed[fn](z))
            y.append(dat[bn])
        return self.logl2_a_(y,y0)
    def fitting_z_err_(self,dat,z,filter_names,band_names,err_names):
        y,y0,err=[],[],[]
        con_fr_sed=self.con_fr_sed
        for fn,bn,en in zip(filter_names,band_names,err_names):
            y0.append(con_fr_sed[fn](z))
            y.append(dat[bn])
            err.append(dat[en])
        return self.l2_err_a_(y,y0,err=err)
    def con_fr_sed_z_(self):
        sed_0=self.sed_0
        filter_norms=self.filter_norms
        con_fr_sed=self.con_fr_sed
        for k in self.filter_keys:
            fr=self.filter_funs[k]
            norm=filter_norms[k]=fr.convolve_with_function(self.ref_x2Fx)
            zs=np.linspace(0,10,1000)
            flxs=np.zeros(1000)
            for i in range(1000):
                sed=self.sed_z_(z=zs[i],z_0=0.0)
                sed_x2Fx=self.sed_to_func(sed)
                flxs[i]=fr.convolve_with_function(sed_x2Fx)/norm
            con_fr_sed[k]=interp1d(zs,flxs)
        #return con_fr_sed
    def compute_z_(self):
        Y=self.Y
        template=self.template
        self.metrics=np.ones(Y['ID'].shape[0])
        self.norms=np.ones(Y['ID'].shape[0])
        for i in range(Y['ID'].shape[0]):
            dat=Y['data'][i]
            sed=self.sed_z_(z=Y['z_best'][i],z_0=0.0)
            self.metrics[i],self.norms[i]=self.fitting_z_(dat,z=Y['z_best'][i],
                                filter_names=Y['filter_names'][i],
                                band_names=Y['band_names'][i])
    def compute_z_err_(self):
        Y=self.Y
        template=self.template
        self.metrics=np.ones(Y['ID'].shape[0])
        self.norms=np.ones(Y['ID'].shape[0])
        for i in range(Y['ID'].shape[0]):
            dat=self.data[i]#Y['data'][i]
            sed=self.sed_z_(z=Y['z_best'][i],z_0=0.0)
            self.metrics[i],self.norms[i]=self.fitting_z_err_(dat,z=Y['z_best'][i],
                                filter_names=Y['filter_names'][i],
                                band_names=Y['band_names'][i],
                                err_names=Y['err_names'][i])

class SED_estimator():
    def __init__(self,data,templates,filter_funs,
                 band_keys,band_to_filter,band_to_err,filter_keys):
        self.data=data
        self.templates=templates
        self.template_0=self.templates[:2,:]
        self.filter_funs=filter_funs

        self.band_keys=band_keys
        self.band_to_filter=band_to_filter
        self.band_to_err=band_to_err
        self.filter_keys=filter_keys
        
        self.metrics_best=np.ones(data['ID'].shape[0])+1e10
        self.l_edds_best=np.ones(data['ID'].shape[0])
        self.norms_best=np.ones(data['ID'].shape[0])
        self.sed_fittings=[]
        self.l_edds=np.zeros(0)
        
    def sampling(self,l_edds,error=True):
        data=self.data
        self.l_edds=np.sort(np.concatenate((self.l_edds,l_edds)))
        for i in range(len(l_edds)):
            print(i)
            l_edd=l_edds[i]
            template=np.zeros(self.template_0.shape)
            template[0]=self.templates[0]
            template[1]=self.templates[1]+l_edd*self.templates[2]
            self.sed_fittings.append(SED_Fitting(data=self.data,template=template,filter_funs=self.filter_funs,
                                                 band_keys=self.band_keys,band_to_filter=self.band_to_filter,
                                                 band_to_err=self.band_to_err,filter_keys=self.filter_keys))
            sed_fitting=self.sed_fittings[-1]
            sed_fitting.config()
            sed_fitting.con_fr_sed_z_()
            if(error):
                sed_fitting.compute_z_err_()
            else:
                sed_fitting.compute_z_()
            metrics,norms=sed_fitting.metrics,sed_fitting.norms
            for j in range(data['ID'].shape[0]):
                if(metrics[j]<self.metrics_best[j]):
                    self.metrics_best[j]=metrics[j]
                    self.norms_best[j]=norms[j]
                    self.l_edds_best[j]=l_edd
