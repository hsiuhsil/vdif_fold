import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from pylab import *
import os, time, glob, itertools
from astropy.time import Time
import astropy.units as u, astropy.constants as c
from baseband.helpers import sequentialfile as sf
from baseband import vdif
from pulsar.predictor import Polyco
from scipy.ndimage.filters import median_filter, uniform_filter1d
import pyfftw.interfaces.numpy_fft as fftw
from numpy import random

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
mstr = f'[{rank:2d}/{size:2d}]:'

_fftargs = {'threads': int(os.environ.get('OMP_NUM_THREADS', 2)), 
            'planner_effort': 'FFTW_ESTIMATE', 
            'overwrite_input': True}

D = 4148.808 * u.s * u.MHz**2 * u.cm**3 / u.pc

class AROPulsarAnalysis():
    """Class to analysis pulsar data at ARO.""" 
    def __init__(self,):

        self.site_name = 'aro'        
        self.psr_name = 'B0329+54'
        self.DM = 26.7641 * u.pc / u.cm**3

        if self.site_name == 'aro':
            self.folder = ('/scratch/p/pen/hsiuhsil/recalled/20210204T234822Z_aro_vdif/00001*/')
            self.polyco = Polyco('/home/p/pen/hsiuhsil/psr_B0329+54/polycoB0329+54_aro_mjd59240_59250.dat')
        elif self.site_name == 'stk':
            self.folder = ('/scratch/p/pen/hsiuhsil/backup/20210205T000326Z_stockert_vdif_256nchan/0000[3-4]*/')
            self.polyco = Polyco('/home/p/pen/hsiuhsil/psr_B0329+54/polycoB0329+54_stockert_mjd59240_59250.dat')

        self.filenames =  '*.vdif' #'0003[0-1]*.vdif'
        self.files = sorted(glob.glob(self.folder + self.filenames))        

        self.fref = 800. * u.MHz

        if self.site_name == 'aro':
            self.full_bw = 400. * u.MHz
            self.nchan = 1024
        elif self.site_name == 'stk':
            self.full_bw = 100. * u.MHz
            self.nchan = 256


        self.npols = 2
        self.chan_bw = self.full_bw / self.nchan
        self.dt = (1 / self.chan_bw).to(u.s)

        fh = self.get_file_handle()
        #2021-02-05T02:24:00.000000000
        self.start_time = Time('2021-02-05T02:24:00.000000000',
                              format='isot', precision=9) + 535*u.s#fh.start_time #+ 0 * u.min
        self.stop_time = self.start_time + 12 * u.s
#        self.stop_time = Time('2018-09-27T13:13:48.086400000',
#                              format='isot', precision=9)
        # self.stop_time = fh.stop_time

        f0 = self.fref - self.full_bw
        wrap_time = (D * self.DM * (1/f0**2 - 1/self.fref**2)).to(u.s)
        print('wrap_time: ',wrap_time)
        wrap_samples = (wrap_time/self.dt).decompose().value
        self.wrap = int(np.ceil(wrap_samples))

        self.ftop = np.linspace(self.fref, self.fref - self.full_bw,
                                self.nchan, endpoint=False)

    def get_file_handle(self):
        """Returns file handle for a given list of channels."""

        fraw = sf.open(self.files, 'rb')
        fh = vdif.open(fraw, mode='rs', sample_rate=self.chan_bw)
        return fh

    def find_outlier_threshold(self, x, n):
        """ Finds threshold for outliers. Function assumes that a clean signal will have mean 0."""

        assert 0 < x.ndim < 3
        x = x[..., np.newaxis] if x.ndim == 1 else x
        thres = []
        for x in x:
            s0 = np.std(x)
            s1 = np.std(x[abs(x) < n*s0])
            while not np.isclose(s0, s1):
                s0 = s1
                s1 = np.std(x[abs(x) < n*s0])
            thres += [n*s1]
        return thres

    def remove_rfi(self, z, freq_smoothing=16, time_smoothing=2048, 
                    nstd=5, cutoff_factor=0):
        """ Remove RFI from a signal """

        y = z.real**2 + z.imag**2
        # All channels are good channels until they become bad channels!
        good_channels = np.ones(y.shape[-1], dtype=bool)
        # Finding mean power in channels
        mean_freq_power = y.mean(0)
        smooth_mean_freq_power = np.zeros_like(mean_freq_power)
        for i in range(self.npols):
            smooth_mean_freq_power[i] = median_filter(mean_freq_power[i],
                                                      freq_smoothing, 
                                                      mode='mirror')
        # Normalizing mean power of channels and recomputing power in channels
        z /= smooth_mean_freq_power[np.newaxis]
        y = z.real**2 + z.imag**2
        # Finding and tagging extra-bright channels as bad channels!
        mean_freq_power = y.mean(0)
        smooth_mean_freq_power = np.zeros_like(mean_freq_power)
        for i in range(self.npols):
            smooth_mean_freq_power[i] = median_filter(mean_freq_power[i],
                                                      freq_smoothing, 
                                                      mode='mirror')
        res = mean_freq_power - smooth_mean_freq_power
        bright_channels = abs(res).T > self.find_outlier_threshold(res, nstd)
        good_channels[bright_channels.any(-1)] = False
        # Finding and tagging highly variable channels as bad channels!
        var_freq_power = y.var(0)
        smooth_var_freq_power = np.zeros_like(var_freq_power)
        for i in range(self.npols):
            smooth_var_freq_power[i] = median_filter(var_freq_power[i],
                                                     freq_smoothing,
                                                     mode='mirror')
        res = var_freq_power - smooth_var_freq_power
        variable_channels = abs(res).T > self.find_outlier_threshold(res, nstd)
        good_channels[variable_channels.any(-1)] = False
        # Excising bad channels, and recomputing power
        z *= good_channels[np.newaxis, np.newaxis, ...]
        y = z.real**2 + z.imag**2
        # Finding time variability and normalizing it
        mean_time_power = y[..., good_channels].mean(-1)
        smooth_mean_time_power = uniform_filter1d(mean_time_power,
                                                  time_smoothing,
                                                  axis=0)
        # Normalizing power in time
        z /= smooth_mean_time_power[..., np.newaxis]
        return z

    def coherent_dedispersion(self, z, channel, axis=0):
        """Coherently dedisperse signal."""

        fcen = self.ftop[channel]
        print('co. dd. fcen: ',fcen)
        tag = "{0:.2f}-{1:.2f}M_{2}".format(self.fref.value, fcen.value,
                                            z.shape[axis])
        ddcoh_file = "saved/ddcoh_{0}.npy".format(tag)
        try:
            dd_coh = np.load(ddcoh_file)
        except:
            f = fcen + np.fft.fftfreq(z.shape[axis], self.dt)
            dang = D * self.DM * u.cycle * f * (1./self.fref - 1./f)**2
            with u.set_enabled_equivalencies(u.dimensionless_angles()):
                dd_coh = np.exp(dang * 1j).conj().astype(np.complex64).value
            np.save(ddcoh_file, dd_coh)
        if z.ndim > 1:
            ind = [np.newaxis] * z.ndim
            ind[axis] = slice(None)
        if z.ndim > 1: dd_coh = dd_coh[ind]
#        z = fftw.fft(z, axis=axis, **_fftargs)
#        z = fftw.ifft(z * dd_coh, axis=axis, **_fftargs)
        z = np.fft.fft(z, axis=axis)
        z = np.fft.ifft(z * dd_coh, axis=axis)
        return z 

    def process_file_test(self, timestamp, num_samples):
        """Seeks, reads and dedisperses signal from a given timestamp"""
        fh = self.get_file_handle()
        print(f'print fh.shape: {fh.shape}')
        fh.seek(timestamp)
        print(f'print fh.seek: {fh.seek(timestamp)}')
#        print(timestamp)
        print(f'print num_samples: {num_samples}')
        z = fh.read(num_samples).astype(np.complex64)
        return z

    def convert_drop_packets(self, z0):
        # z is the voltage data in the shape of (ntime, npol, nfreq)
    
        z = z0.copy()
        for pol in range(2):

            x = z[:,pol,:].copy()
            amp = abs(x)
            phase = np.angle(x)
            # masking out the dropping packets
            drop_time = np.where((np.count_nonzero(amp,axis=-1)==0)&(np.count_nonzero(phase,axis=-1)==0))[0]
#            print('drop_time',drop_time)
            x[drop_time,:]=np.nan
            amp = abs(x)
            phase = np.angle(x)
    
            for f in range(x.shape[-1]):
                s = np.where(np.isnan(x[:,f])==True)[0]
  
                # the distribution of amp and phase
                a = amp[:,f]
                p = phase[:,f]
            
                a = a[~np.isnan(a)]
                p = p[~np.isnan(p)]

                random.seed(42)
                da = random.choice(a,len(s))
                dp = random.choice(p,len(s))    
            
#            print('a.mean(),da.mean()',a.mean(),da.mean(), '.mean(),dp.mean()',p.mean(),dp.mean())
            
                if False:
                    if f%250==0:
                        plt.figure(figsize=(16,6))
                        #plt.suptitle('fraction of dropping packets: '+"%.3f"%drop_frac)
                        plt.subplot(221)
                        plt.title(str(ftop[f])+', Amplitude')
                        plt.hist(a,bins=20,label='data')
                        plt.hist(da,bins=20,label='random')
                        plt.yscale('log')
                        plt.legend()

                        plt.subplot(222)
                        plt.title(str(ftop[f])+', phase')    
                        plt.hist(p,bins=20,label='data')
                        plt.hist(dp,bins=20,label='random')
                        plt.yscale('log')
                        plt.ylim(0,1e4)
                        plt.legend()
                        plt.show()
            
                dz = da*np.exp(-1j*dp)
                z[s,pol,f]=dz
        return z

    def process_file(self, timestamp, num_samples):
        """Seeks, reads and dedisperses signal from a given timestamp"""

        if num_samples <= self.wrap:
            raise Exception(f'num_samples must be larger than {self.wrap}!')
        else:
            t0 = time.time()
            fh = self.get_file_handle()
            print('fh.start_time', fh.start_time)
            print('fh.stop_time', fh.stop_time)
            print('fh.shape', fh.shape)
            fh.seek(timestamp)
            print('fh.seek(timestamp)', fh.seek(timestamp))
            print ('timestamp', timestamp)
            print('num_samples',num_samples)
            z = fh.read(num_samples).astype(np.complex64)
            print ('z_original.shape', z.shape)

#            if True: #for ARO, 0th freq is the lowest. and then selecting freq channel
#                z = z[:,:,::-1]
#                z = z[:,:,0:384]
#                print('selected z shape: ',z.shape)

            if False: #for CHIME
                if z.shape[-1] != 2:
                    z = z.reshape(z.shape[0],z.shape[1],4,2).transpose(0,2,1,3).reshape(z.shape[0],z.shape[1]*int(z.shape[-1]/2),2)
                    z = z.transpose(0,2,1)
#                z = self.remove_rfi(z) 
            print('z_reshape.shape',z.shape)
            print('z[0]',z[0])
            t1 = time.time()
            print(f'{mstr} Took {t1 - t0:.2f}s to read.')
            t2 = time.time()

            '''converting the dropping packets'''
            if True:
                z = self.convert_drop_packets(z)

#            for channel in range(self.nchan):
            for channel in range(int(self.nchan)):
                print('channel:',channel)
                z[..., channel] = self.coherent_dedispersion(z[..., channel],                                         channel)
            z = z[:-self.wrap]
            t3 = time.time()
            print(f'{mstr} Took {t3 - t2:.2f}s to dedisperse.')
#        print ('z return shape', z.shape)
        return z

    def get_phases(self, timestamp, num_samples, dt, ngate):
        """Returns pulse phase."""

        phasepol = self.polyco.phasepol(timestamp, rphase='fraction', 
                                        t0=timestamp, time_unit=u.second,
                                        convert=True)
        ph = phasepol(np.arange(num_samples) * dt.to(u.s).value)
        ph -= np.floor(ph[0])
        ph = np.remainder(ph * ngate, ngate).astype(np.int32)
        return ph

def make_waterfall(pa, timestamp, num_samples, tbin=1024):
    fh = pa.get_file_handle()
    fh.seek(timestamp)
    t0 = time.time()
    z = fh.read(num_samples).astype(np.complex64)
    t1 = time.time()
    print(f'{mstr} Took {t1 - t0:.2f}s to read data.')
    t2 = time.time()
    for channel in range(pa.nchan):
        z[..., channel] = pa.coherent_dedispersion(z[..., channel], channel)
    t3 = time.time()
    print(f'{mstr} Took {t3 - t2:.2f}s to dedisperse.')
    wrap = pa.wrap + (-pa.wrap % tbin)
    z = z[:-wrap]
    z = (z.real**2 + z.imag**2).astype(np.float32)
    z = z.reshape(-1, tbin, 2, 1024).mean(1)
    return z

def fold_band(pa, timestamp, num_samples, ngate, NFFT):
    z = pa.process_file(timestamp, num_samples)
    t0 = time.time()
#    z = fftw.fft(z.reshape(-1, NFFT, pa.npols, pa.nchan), axis=1, **_fftargs)
#    z = fftw.fftshift(z, axes=(1,))
    print('z.shape', z.shape)
#    z_pol = z
#    print('z_pol.shape', z_pol.shape)
    z = np.fft.fft(z.reshape(-1, NFFT, pa.npols, pa.nchan), axis=1)
    z = np.fft.fftshift(z, axes=(1,))
    print('z.shape after fft',z.shape)
    z = (z.real**2 + z.imag**2).sum(2).astype(np.float32)
    z = z.transpose(0, 2, 1).reshape(z.shape[0], -1)
    print('z.shape after transpose: ',z.shape)
    ph = pa.get_phases(timestamp, z.shape[0], NFFT*pa.dt, ngate)
    count = np.bincount(ph, minlength=ngate)
    print('count.shape',count.shape)
    pp = np.zeros((ngate, z.shape[-1]))
    print('pp.shape',pp.shape)
    for channel in range(z.shape[-1]):
        pp[..., channel] = np.bincount(ph, z[..., channel], minlength=ngate)

#    pp_pol = np.zeros((ngate, pa.npols, z.shape[-1]),dtype=np.complex64)
#    for pol_chan in range(z_pol.shape[1]):
#        for channel in range(z_pol.shape[-1]):
#            pp_pol[..., pol_chan, channel] = np.bincount(ph, z_pol[..., pol_chan, channel], minlength=ngate)
    t1 = time.time()
    print(f'{mstr} Took {t1 - t0:.2f}s to fold 1 block.')
#    return pp, pp_pol, count[..., np.newaxis]
    return pp, count[..., np.newaxis]

x = AROPulsarAnalysis()
N = 2**20

# print(f'Making waterfall for {x.psr_name}.')
# z = make_waterfall(x, x.start_time, N)
# np.save(f"{x.psr_name}_waterfall_plus10min.npy", z)

ngate = 512
NFFT = 1

x.wrap += (-x.wrap) % NFFT
#block_length = ((N - x.wrap) * x.dt).to(u.s)
block_length = 5 * u.s
print('block_length:',block_length)
max_time = ((x.stop_time - x.start_time) - x.wrap * x.dt).to(u.s)
print('max_time',max_time)
max_blocks = int(floor((max_time / block_length).decompose().value))
print('max_blocks: ',max_blocks)
num_blocks = max_blocks#1
assert num_blocks <= max_blocks
timestamps = [x.start_time + i * block_length for i in range(num_blocks)]

ppfull = np.zeros((num_blocks, ngate, x.nchan * NFFT), dtype=np.float64)
#ppfull_pol = np.zeros((ngate, x.npols, x.nchan * NFFT), dtype=np.float64) 
counts = np.zeros((num_blocks, ngate, x.nchan * NFFT), dtype=np.int64)

if rank == 0:
    print(f"------------------------\n"
          f"Folding {x.psr_name} data.\n"
          f"Observation Details --\n"
          f"{x.start_time} -> {x.stop_time}\n"
          f"Total Duration (s): {max_time}\n"
          f"Block Length (s): {block_length.to(u.s)}\n"
          f"No. of blocks: {num_blocks} (Max: {max_blocks})\n"
          f"Time to fold: {(num_blocks * block_length).to(u.s)}\n"
          f"------------------------", flush=True)

comm.Barrier()

time.sleep(rank)
for timestamp, k in zip(timestamps[rank::size], (np.arange(num_blocks/size)*size).astype(int)+rank):
    print(f'{mstr} {timestamp}')
#    pp, pp_pol, count = fold_band(x, timestamp, N, ngate, NFFT)
    pp,  count = fold_band(x, timestamp, N, ngate, NFFT)
#    ppfull += pp
#    ppfull_pol += pp_pol
#    counts += count
    ppfull[k] = pp
    counts[k] = count


all_pp = None
#all_pp_pol = None
all_count = None
if rank == 0:
    all_pp = np.zeros((num_blocks, ngate, x.nchan * NFFT), dtype=np.float64)
#    all_pp_pol = np.zeros((ngate, x.npols, x.nchan * NFFT), dtype=np.float64)
    all_count = np.zeros((num_blocks, ngate, x.nchan * NFFT), dtype=np.int64)

comm.Barrier()
comm.Reduce(ppfull, all_pp, root=0)
#comm.Reduce(ppfull_pol, all_pp_pol, root=0)
comm.Reduce(counts, all_count, root=0)

if rank == 0:
    pp_final = all_pp / all_count
#    pp_pol_final = all_pp_pol / all_count
    print(f'{mstr} Folded {(num_blocks * block_length).to(u.s)} of data.')
    print(f'{mstr} Generated {ngate}-gate, {x.nchan * NFFT}-channel pulse profile!')
    np.savez(f"nondrop3_{x.site_name}_{x.psr_name}_{ngate}g_{x.nchan * NFFT}c_start{x.start_time.value}.npz", fold=pp_final, start_time=x.start_time.value, stop_time=x.stop_time.value)

#    np.save(f"{x.psr_name}_{ngate}g_{x.nchan * NFFT}c_pol.npy", pp_pol_final)


