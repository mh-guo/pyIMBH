import numpy as np
import multiprocessing as mp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import astropy.units as units
from dust_attenuation.averages import C00
from dust_attenuation.radiative_transfer import WG00

class SED_Fitting():
    def __init__(self,data,template,filter_funs,
                 band_keys,band_to_filter,band_to_err,filter_keys,
                 loss='l2_err'):
        self.data=data
        self.template=template
        self.filter_funs=filter_funs
        self.band_keys=band_keys
        self.band_to_filter=band_to_filter
        self.band_to_err=band_to_err
        self.filter_keys=filter_keys
        self.loss=loss
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
    def logl2_err_(self,y,y0,err):
        y,y0,err=np.asarray(y),np.asarray(y0),np.asarray(err)
        logerr=err/y/np.log(10)
        b=np.sum((np.log10(y)-np.log10(y0))/logerr**2)/np.sum(1/logerr**2)
        return np.mean((np.log10(y)-np.log10(y0)-b)**2/logerr**2)
    def logl2_err_a_(self,y,y0,err):
        y,y0,err=np.asarray(y),np.asarray(y0),np.asarray(err)
        logerr=err/y/np.log(10)
        b=np.sum((np.log10(y)-np.log10(y0))/logerr**2)/np.sum(1/logerr**2)
        return np.mean((np.log10(y)-np.log10(y0)-b)**2/logerr**2),10**b
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
        loss = self.loss
        y,y0,err=[],[],[]
        con_fr_sed=self.con_fr_sed
        for fn,bn,en in zip(filter_names,band_names,err_names):
            if(z<10):
                y0.append(con_fr_sed[fn](z))
            else:
                fr=self.filter_funs[filter_names]
                sed=self.sed_z_(z=z,z_0=0.0)
                sed_x2Fx=self.sed_to_func(sed)
                norm=self.filter_norms[filter_names]
                y0.append(fr.convolve_with_function(sed_x2Fx)/norm)
            y.append(dat[bn])
            err.append(dat[en])
        if (loss=='l2_err'):
            return self.l2_err_a_(y,y0,err=err)
        else:
            return self.logl2_err_a_(y,y0,err=err)
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
                 band_keys,band_to_filter,band_to_err,filter_keys,
                 loss='l2_err'):
        self.data=data
        self.templates=templates
        self.template_0=self.templates[:2,:]
        self.filter_funs=filter_funs

        self.band_keys=band_keys
        self.band_to_filter=band_to_filter
        self.band_to_err=band_to_err
        self.filter_keys=filter_keys
        self.loss=loss
        
        self.metrics_best=np.ones(data['ID'].shape[0])+1e10
        self.l_edds_best=np.ones(data['ID'].shape[0])
        self.norms_best=np.ones(data['ID'].shape[0])
        self.sed_fittings=[]
        self.l_edds=np.zeros(0)

    def conf_samples(self,l_edds):
        self.l_edds=np.sort(np.concatenate((self.l_edds,l_edds)))
        for i in range(len(l_edds)):
            print(i)
            l_edd=l_edds[i]
            template=np.zeros(self.template_0.shape)
            template[0]=self.templates[0]
            template[1]=self.templates[1]+l_edd*self.templates[2]
            self.sed_fittings.append(SED_Fitting(data=self.data,template=template,filter_funs=self.filter_funs,
                                                 band_keys=self.band_keys,band_to_filter=self.band_to_filter,
                                                 band_to_err=self.band_to_err,filter_keys=self.filter_keys,
                                                 loss=self.loss))
    def set_samples(self,path):
        data=self.data
        l_edds=self.l_edds
        for i in range(len(l_edds)):
            print(i)
            l_edd=l_edds[i]
            sed_fitting=self.sed_fittings[i]
            arr=np.loadtxt(path+f'{i:05}.txt')
            sed_fitting.metrics,sed_fitting.norms=arr[0],arr[1]
            metrics,norms=sed_fitting.metrics,sed_fitting.norms
            for j in range(data['ID'].shape[0]):
                if(metrics[j]<self.metrics_best[j]):
                    self.metrics_best[j]=metrics[j]
                    self.norms_best[j]=norms[j]
                    self.l_edds_best[j]=l_edd
    
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
            '''        if (cpu_count>1):
            with mp.Pool(cpu_count) as p:
                #p.map(run,list(range(0,201)))
                p.map(run,self.sed_fittings)
        else:
            for i in range(len(l_edds)):
                run(i)
        for i in range(len(l_edds)):
            print(i)'''
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

class Dust():
    def __init__(self,tau_V=1.5) -> None:
        self.att_model = WG00(tau_V = tau_V, geometry = 'shell',
                 dust_type = 'smc', dust_distribution = 'clumpy')
        self.att_model = C00(Av = Av)
        pass
    def __call__(self,x):
        # generate the curves and plot them
        #ix = np.arange(1.0/3.0,1.0/0.1,0.1)/units.micron
        #x = 1./ix
        return self.att_model(x)

class SED_curve_fit():
    # expect to get distilled data, but still need filters
    # use template and parameters to construct the fitting function
    def __init__(self,xdata,ydata,sigma,templates,filter_funs,norms,
                 loss='l2_err'):
        self.xdata=np.asarray(xdata)
        self.ydata=np.asarray(ydata)
        self.sigma=np.asarray(sigma)
        self.templates=templates
        self.filter_funs=filter_funs
        self.norms=norms
        self.loss=loss
        self.res={}
        return
    # fitting function, depending on what exactly you want to fit
    def _f(self,curve_x,curve_y,return_a=False):
        y=[]
        y0=self.ydata
        err=self.sigma
        sed_x2Fx=interp1d(curve_x,curve_y,fill_value="extrapolate")
        y=[fr.convolve_with_function(sed_x2Fx)/norm for fr,norm in zip(self.filter_funs,self.norms)]
        y=np.asarray(y)
        a=np.sum(y0*y/err**2)/np.sum(y**2/err**2)
        if (return_a):
            return a
        return a*y
    def shift_x(self,x_0,z,z_0=0):
        return x_0*((1+z)/(1+z_0))
    def shift_Fnu(self,y_0,z,z_0=0):
        return y_0*((1+z)/(1+z_0))
    def shift_Fx(self,y_0,z,z_0=0):
        return y_0/((1+z)/(1+z_0))
    def dust(self,tau_V,sed_x_0):
        #dust=Dust(max(tau_V,0.25))
        Av=2.5/np.log(10)*tau_V
        dust=C00(Av=Av)
        x_0=sed_x_0/1e4*units.micron
        #dust_x0=x_0[x_0<3*units.micron]
        dust_x0=x_0[np.logical_and(x_0>0.12*units.micron,x_0<2.2*units.micron)]
        dust_att=dust(dust_x0)
        dust_att=dust_att*tau_V/max(tau_V,0.25)
        #dust_x2att=interp1d(dust_x0,dust_att,fill_value="extrapolate")
        #dust_x2att=interp1d(1/dust_x0,dust_att,fill_value="extrapolate")
        #att=dust_x2att(1/x_0)
        dust_x2att=interp1d(np.append(1/dust_x0**3,0.0),np.append(dust_att,0.0),fill_value="extrapolate")
        att=np.maximum(dust_x2att(1/x_0**3),0.0)
        return att
    def f_ledd(self,x,l_edd):
        sed_x=self.templates['x']
        sed_y=self.templates['galaxy']+l_edd*self.templates['agn']
        return self._f(sed_x,sed_y)
    def sed_ledd_tauv(self,l_edd,tau_V,with_a=True):
        sed_x=self.templates['x']
        sed_x_0=self.templates['x_0']
        att=self.dust(tau_V,sed_x_0)
        sed_y=self.templates['galaxy']+l_edd*self.templates['agn']/10**(att/2.5)
        sed_y_0=self.templates['Fnu_gal_0']+l_edd*self.templates['Fnu_agn_0']/10**(att/2.5)
        a=self._f(sed_x,sed_y,return_a=True) if with_a else 1.0
        return sed_x_0,sed_y_0*a
    def f_ledd_tauv(self,x,l_edd,tau_V):
        sed_x=self.templates['x']
        sed_x_0=self.templates['x_0']
        att=self.dust(tau_V,sed_x_0)
        sed_y=self.templates['galaxy']+l_edd*self.templates['agn']/10**(att/2.5)
        return self._f(sed_x,sed_y)
    def sed_ledd_tauv_z(self,l_edd,tau_V,z,with_a=True):
        att=self.dust(tau_V,self.templates['x_0'])
        sed_x=self.shift_x(self.templates['x_0'],z)
        sed_y=self.shift_Fx(self.templates['galaxy_0']+l_edd*self.templates['agn_0']/10**(att/2.5),z)
        sed_y_0=self.templates['Fnu_gal_0']+l_edd*self.templates['Fnu_agn_0']/10**(att/2.5)
        a=self._f(sed_x,sed_y,return_a=True) if with_a else 1.0
        return self.templates['x_0'],sed_y_0*a
    def f_ledd_tauv_z(self,x,l_edd,tau_V,z):
        att=self.dust(tau_V,self.templates['x_0'])
        sed_x=self.shift_x(self.templates['x_0'],z)
        sed_y=self.shift_Fx(self.templates['galaxy_0']+l_edd*self.templates['agn_0']/10**(att/2.5),z)
        return self._f(sed_x,sed_y)
    def fit(self,f,**kwargs):
        self.res['params']=curve_fit(f,self.xdata,self.ydata,sigma=self.sigma,**kwargs)
        popt=self.res['popt']=self.res['params'][0]
        self.res['chi2']=np.mean((self.ydata-f(self.xdata,*popt))**2/self.sigma**2)
        return self.res

class SEDs_Fitting():
    def __init__(self,data,templates,filter_funs,
                 band_keys,band_to_filter,band_to_err,filter_keys,
                 loss='l2_err',fit_method=''):
        self.data=data
        self.templates=templates
        self.filter_funs=filter_funs
        self.band_keys=band_keys
        self.band_to_filter=band_to_filter
        self.band_to_err=band_to_err
        self.filter_keys=filter_keys
        self.loss=loss
        self.n_data=data['ID'].shape[0]
        self.filter_norms={}
        for k in self.filter_keys:
            fr=self.filter_funs[k]
            self.filter_norms[k]=fr.convolve_with_function(self.ref_x2Fx_)
        self.sed_0={}
        self.sed_0['x']=self.templates[0]
        self.sed_0['Fnu_gal']=self.templates[1]
        self.sed_0['Fnu_agn']=self.templates[2]
        self.fit_method=fit_method
        return
    
    def config(self):
        self.distill_()
    
    def distill_(self):
        data=self.data
        Y=self.Y={}
        for k in ['ID','z_best','z_best_type']:
            Y[k]=np.array([data[k][i] for i in range(self.n_data)])
        Y['band_names']=[[] for i in range(self.n_data)]
        Y['filter_names']=[[] for i in range(self.n_data)]
        Y['err_names']=[[] for i in range(self.n_data)]
        Y['data']=[{} for i in range(self.n_data)]
        for i in range(self.n_data):
            dat=data[i]
            for bn in self.band_keys:
                if(not(np.isnan(dat[bn]) or dat[bn]<=0)):
                    Y['band_names'][i].append(bn)
                    Y['filter_names'][i].append(self.band_to_filter[bn])
                    Y['err_names'][i].append(self.band_to_err[bn])
                    Y['data'][i][bn]=dat[bn]
        Y['num_bands']=np.array([len(Y['band_names'][i]) for i in range(self.n_data)])
        locs=data['z_best_type']=='p'
        for k in ['z683_low','z683_high','z954_low','z954_high']:
            Y[k]=np.array(Y['z_best'])
            Y[k][locs]=np.array(data['mFDa4_'+k])[locs]

    def ref_x2Fx_(self,x):
        unity=interp1d([-1,1],[1,1],fill_value="extrapolate")
        lin=interp1d([-1,1],[-1,1],fill_value="extrapolate")
        return unity(x)/lin(x)**2
    
    def sed_z_(self,z,z_0):
        sed_0=self.sed_0
        sed={}
        sed['x']=sed_0['x']*((1+z)/(1+z_0))
        #sed['Fnu']=sed_0['Fnu']*((1+z)/(1+z_0))
        sed['Fnu_gal']=sed_0['Fnu_gal']*((1+z)/(1+z_0))
        sed['Fnu_agn']=sed_0['Fnu_agn']*((1+z)/(1+z_0))
        #speed_of_light=1
        #sed['Fx']=sed['Fnu']/sed['x']**2
        return sed
    
    def set_one(self,i):
        Y=self.Y
        filter_names=Y['filter_names'][i]
        band_names=Y['band_names'][i]
        err_names=Y['err_names'][i]
        dat=self.data[i]
        xdata=np.arange(len(filter_names))
        frs=[self.filter_funs[fn] for fn in filter_names]
        norms=[self.filter_norms[fn] for fn in filter_names]
        y=[dat[bn] for bn in band_names]
        err=[dat[en] for en in err_names]
        z=Y['z_best'][i]
        sed=self.sed_z_(z=z,z_0=0.0)
        templates={}
        templates['z']=z
        templates['x_0']=self.sed_0['x']*1e4
        templates['Fnu_gal_0']=self.sed_0['Fnu_gal']
        templates['Fnu_agn_0']=self.sed_0['Fnu_agn']
        templates['galaxy_0']=self.sed_0['Fnu_gal']/(self.sed_0['x']*1e4)**2
        templates['agn_0']=self.sed_0['Fnu_agn']/(self.sed_0['x']*1e4)**2
        templates['x']=sed['x']*1e4
        templates['galaxy']=sed['Fnu_gal']/(sed['x']*1e4)**2
        templates['agn']=sed['Fnu_agn']/(sed['x']*1e4)**2
        scf=SED_curve_fit(xdata,ydata=y,sigma=err,templates=templates,
                            filter_funs=frs,norms=norms)
        return scf
    
    def fit_one(self,i):
        scf=self.scfs[i]
        if(self.Y['num_bands'][i]>6):
            if(self.fit_method=='ledd_tauv_z'):
                if(self.Y['z_best_type'][i]=='p'):
                    scf.fit(scf.f_ledd_tauv_z,
                        bounds=([1e-6,0.0,self.Y['z954_low'][i]],[1,50.,self.Y['z954_high'][i]]))
                else:
                    scf.fit(scf.f_ledd_tauv,bounds=([1e-6,0.0],[1,50.]))
            elif(self.fit_method=='ledd_tauv'):
                #print('fit_method')
                scf.fit(scf.f_ledd_tauv,bounds=([0,0.0],[1,50.]))
            else:
                scf.fit(scf.f_ledd,bounds=[0,1])
        return scf
        
    def set(self):
        self.scfs=[]
        for i in range(self.n_data):
            if(i%100==0): print(i)
            self.scfs.append(self.set_one(i))

    def fit(self):
        for i in range(self.n_data):
            if(i%100==0): print(i)
            self.fit_one(i)

    def run_one(self,i):
        self.set_one(i)
        self.fit_one(i)

    def run(self):
        self.set()
        self.fit()
