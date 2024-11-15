import spikeinterface.full as si
from spikeinterface.extractors import read_nwb
from pynwb import NWBHDF5IO
from ndx_sound import AcousticWaveformSeries

import numpy as np
import pandas as pd
import copy

import pickle
import scipy.stats as stats


import warnings
warnings.filterwarnings("ignore", message="Ignoring cached namespace 'hdmf-common'")
warnings.filterwarnings("ignore", message="Ignoring cached namespace 'hdmf-experimental'")

from . import analysis

basepath = '/Volumes/Zimo/Auditory/Data/'
recordingpath = '/Volumes/Research/GapInNoise/Data/Recordings/'
mouse = pd.read_csv('/Volumes/Research/GapInNoise/Code/Mouse_Tones.csv')


class Group:
    def __init__(self, geno_type, hearing_type):
        self.geno_type = geno_type 
        self.hearing_type = hearing_type
        self.label = geno_type + '_' + hearing_type
        self.recording = self.Get_Group_Recording()
        self.gaps = [0., 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]
        self.gaps_label = self.Get_Gaps_Label()
        
        self.unit_onset, self.unit_offset = [], []
        self.unit_type = np.array([])
        self.unit_id = np.array([[0, 0, 0]]) 
        self.response_per_recording = self.Get_Response_per_Recording()
        self.pop_spikes = self.Get_Pop_Spikes()
        self.pop_response = self.Get_Pop_Response()
        self.pop_response_stand = self.Get_Pop_Response_Standardized()
        
        self.unit_num = self.pop_response.shape[0]

        self.pca = analysis.PCA(self.pop_response_stand)
        self.periods_pca = self.Get_PCA_for_periods()
    
    def Get_Group_Recording(self):
        if self.hearing_type == 'NonHL':
            return mouse[(mouse['Geno']==self.geno_type)&(mouse['L_Thres']>42)]['Recording'].values
        else:
            return mouse[(mouse['Geno']==self.geno_type)&(mouse['L_Thres']<42)]['Recording'].values
    
    def Get_Gaps_Label(self):
        def Create_Sound_Cond(gap_duration):
            pre_noise = [0 for i in range(100)]
            pre_gap = [1 for i in range(250)]
            gap = [0 for i in range(round(gap_duration*1000))]
            post_gap = [1 for i in range(100)]

            sound_cond = pre_noise + pre_gap + gap + post_gap
            
            for i in range(len(sound_cond), 1000): sound_cond.append(0)
            
            return sound_cond
        
        gaps_label = []
        for i in range(len(self.gaps)):
            gaps_label.append(Create_Sound_Cond(self.gaps[i]))
        return np.array(gaps_label)
        
    def Get_Response_per_Recording(self):
        response_per_recording = {}
        for Exp_name in self.recording:
            try:
                with open(recordingpath + Exp_name + '.pickle', 'rb') as file:
                    recording = pickle.load(file)
            except FileNotFoundError:
                recording = Recording(Exp_name)
            response_per_recording[Exp_name] = recording.response
            self.unit_onset += recording.unit_onset
            self.unit_offset += recording.unit_offset
            self.unit_type = np.concatenate((self.unit_type, recording.unit_type))
            self.unit_id = np.concatenate((self.unit_id, recording.response['unit_id']))
        self.unit_id = self.unit_id[1:]
        return response_per_recording
    
    def Get_Pop_Spikes(self):
        meta_spikes = []
        for gap_idx in range(len(self.gaps)):
            spikes = []
            for Exp_name in self.recording:
                spikes += self.response_per_recording[Exp_name]['spike'][gap_idx]
            meta_spikes.append(spikes)
        return meta_spikes
    
    def Get_Pop_Response(self):
        meta_psth = np.zeros((2, 10, 1000))
        for Exp_name in self.recording:
            meta_psth = np.concatenate((meta_psth,
                                        self.response_per_recording[Exp_name]['sig_psth'][:,:,:,:].mean(axis=2)),
                                       axis=0)
        return meta_psth[2:]
            
    def Get_Pop_Response_Standardized(self):
        def Normalize(X): #standardize each row
            # X: ndarray, shape (n_features, n_samples)
            X = X[0]
            baseline_mean = np.mean(X[0:100])
            whole_std = max(abs(X-baseline_mean))
            if whole_std == 0: return X - baseline_mean
            return (X - baseline_mean) / whole_std

        meta_psth_z = np.zeros(self.pop_response.shape)
        for i in range(self.pop_response.shape[0]):
            for j in range(self.pop_response.shape[1]):
                meta_psth_z[i,j] = Normalize(self.pop_response[i,j].reshape(1,-1)) # 1*1000
                
        return meta_psth_z

    def Get_PCA_for_periods(self):
        periods_pca = []
        for gap_idx in range(10):
            gap_dur = round(self.gaps[gap_idx]*1000+350)

            N1_onset = self.pop_response_stand[:, gap_idx,100:200] # first 100 ms of noise1
            N2_onset = self.pop_response_stand[:,gap_idx, gap_dur:gap_dur+100] # first 100 ms of noise2
            N1_offset = self.pop_response_stand[:,gap_idx, gap_dur + 100: gap_dur + 100 + 100] # first 100 ms of post-N2 silence
            N2_onset_exc_N1_on = N2_onset - N1_onset
            N2_onset_exc_N1_on_off = N2_onset - N1_offset - N1_onset
            
            periods = [N1_onset, N2_onset, N2_onset_exc_N1_on, N2_onset_exc_N1_on_off, N1_offset]
            periods_pca_per_gap = [analysis.PCA(period, multiple_gaps = False) for period in periods]
            periods_pca.append(periods_pca_per_gap)
        return np.array(periods_pca)
            
class Recording:
    def __init__(self, rec_name):
        self.rec_name = rec_name
        self.geno_type = mouse[mouse['Recording'] == rec_name]['Geno'].to_numpy()[0]
        self.hearing_type = None
        self.hearing_threshold = None
        
        self.unit = None
        self.unit_id = None
        self.sorting = None
        self.gap_onset = None
        self.gaps = [0., 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]
        self.Get_Info()
        
        self.unit_onset, self.unit_offset = [], []
        self.unit_type = None
        self.response = self.Get_Neural_Response()
        self.response_per_gap = self.Get_Neural_Response_per_Gap()
        
        self.pop_response = self.Get_Pop_Response()
        self.pop_response_stand = self.Get_Pop_Response_Standardized()
        
        self.Save_File()
        
    
    def Get_Info(self):
        if mouse[mouse['Recording'] == self.rec_name]['L_Thres'].to_numpy()[0] < 42: self.hearing_type = 'HL'
        else: self.hearing_type = 'NonHL'
        self.hearing_threshold = (mouse[mouse['Recording'] == self.rec_name]['L_Thres'].to_numpy()[0] 
                                  + mouse[mouse['Recording'] == self.rec_name]['R_Thres'].to_numpy()[0])/2
        
        # Get Channel
        self.unit = np.load(basepath + self.rec_name + '/FRA_unit.npy')
        
        # Get signal
        self.sorting = read_nwb(basepath + self.rec_name + '/' + self.rec_name + '.nwb', 
                                load_recording=False, load_sorting=True, electrical_series_name='ElectricalSeriesAP')
        
        # Get gap_onset time/frame
        io = NWBHDF5IO(basepath + self.rec_name + '/' + self.rec_name + '.nwb', "r")
        nwbfile = io.read()
        GiN = nwbfile.get_time_intervals("GiN").to_dataframe()
        io.close()
        
        gaps = np.unique(GiN['gap']) # [0, ..., 0.256] gap durations (ms)
        gap_onset = np.zeros((len(gaps), 1+len(GiN)//len(gaps))) # m*(n+1), m= gap types, n = number of trials in each gap duration
        gap_onset[:,0] = np.transpose(gaps) #gap_onset[i][0] = gap duration
        for i in range(len(gaps)):
            gap_onset[i,1:]=np.array(GiN[GiN['gap']==gaps[i]]['start_time']+0.1)
        self.gap_onset = gap_onset
    
        # Load phy_id and si_id match list
        qm = pd.read_csv(basepath + self.rec_name + '/we/quality_metrics/metrics.csv')
        si_id = np.array(qm.index)
        phy_id = qm['Unnamed: 0'].values

        # Load neuron_identity (phy_id)
        filename = basepath + 'waveform/' + self.rec_name + '_waveform_analysis_label.npz'
        num = np.load(filename)['arr_0']
        label = np.load(filename)['arr_3']

        unit_id = np.zeros([len(self.unit),3])
        for i in range(len(self.unit)):
            unit_id[i,0] = self.unit[i]
            unit_id[i,1] = phy_id[np.where(si_id==self.unit[i])[0][0]]
            unit_id[i,2] = label[np.where(num==unit_id[i,1])][0]
        self.unit_id = unit_id
    
    def Get_Neural_Response(self):
        bin = 1/1000
        samplerate = self.sorting.get_sampling_frequency()

        # Background time
        bkg_t = np.zeros((round((np.shape(self.gap_onset)[1]-1)/15),2))
        for i in range(bkg_t.shape[0]):
            bkg_t[i] = [self.gap_onset[:,round(1+i*3):round(4+i*3)].min()-3.5,
                        self.gap_onset[:,round(1+i*3):round(4+i*3)].min()-0.5] # -3.5 to -0.5 s before the first gap

        # Background response
        bkg_psth = np.zeros((len(self.unit),bkg_t.shape[0],len(np.arange(bkg_t[0,0],bkg_t[0,1],bin))))
        for idx_unit in range(len(self.unit)):
            for i in range(bkg_t.shape[0]):
                st = self.sorting.get_unit_spike_train(unit_id=self.unit[idx_unit],start_frame=np.round(bkg_t[i,0]*samplerate),end_frame=np.round(bkg_t[i,1]*samplerate))/samplerate-bkg_t[i,0]
                bkg_psth[idx_unit,i] = np.histogram(st,np.arange(0,3+bin,bin))[0]/bin
        bkg_mean = np.mean(bkg_psth.mean(axis=1),axis=1) # Response average to background noise for each unit
        bkg_std = np.std(bkg_psth.mean(axis=1),axis=1)   # Response std to background noise for each unit

        # Signal information
        gap_onset_ = self.gap_onset[:,1:]
        sig_psth = np.zeros((len(self.unit),gap_onset_.shape[0], gap_onset_.shape[1],round(1/bin))) # 1s response time
        for idx_unit in range(len(self.unit)):
            for idx_gap in range(sig_psth.shape[1]):
                for idx_trial in range(sig_psth.shape[2]):
                    st = self.sorting.get_unit_spike_train(
                        unit_id=self.unit[idx_unit],
                        start_frame=np.round((gap_onset_[idx_gap, idx_trial] - 0.1) * samplerate),
                        end_frame=np.round((gap_onset_[idx_gap, idx_trial]+0.9) * samplerate))/samplerate-(gap_onset_[idx_gap, idx_trial]-0.1)
                    sig_psth[idx_unit, idx_gap, idx_trial] = np.histogram(st, np.arange(0, 1 + bin, bin))[0]/bin

        spikes = [] # 10*N_*45*t
        for idx_gap in range(sig_psth.shape[1]):
            spikes_per_gap = []
            for idx_unit in range(len(self.unit)):
                if bkg_std[idx_unit] < 1e-10: continue
                spikes_per_unit = []
                for idx_trial in range(sig_psth.shape[2]):
                    st = self.sorting.get_unit_spike_train(
                        unit_id=self.unit[idx_unit],
                        start_frame=np.round((gap_onset_[idx_gap, idx_trial] - 0.1) * samplerate),
                        end_frame=np.round((gap_onset_[idx_gap, idx_trial]+0.9) * samplerate))/samplerate-(gap_onset_[idx_gap, idx_trial]-0.1)
                    spikes_per_unit.append(st)
                spikes_per_gap.append(spikes_per_unit)
            spikes.append(spikes_per_gap)

        sig_psth, unit_id = sig_psth[bkg_std>0], self.unit_id[bkg_std>0] #Only want responsive neurons?
        return {'spike':spikes, 'sig_psth':sig_psth, 'unit_id':unit_id}
    
    def Get_Neural_Response_per_Gap(self):      
        response_per_gap = []
        for i in range(len(self.gaps)):
            matrix = np.array([[] for i in range(self.response['sig_psth'].shape[0])])
            for trial in range(self.response['sig_psth'].shape[2]):
                matrix = np.hstack((matrix, self.response['sig_psth'][:, i, trial, :]))
            response_per_gap.append(matrix)
        
        return np.array(response_per_gap)
    
    def Get_Pop_Response(self):
        def Calculate_Z_Score(pre, post):
            pre_mean, pre_std = np.mean(pre), np.std(pre)
            post_mean = np.mean(post)
            return (post_mean - pre_mean)/(pre_std + 1e-10)
                
            
        def Calculate_Onset_Offset(matrix):
            for i in range(len(matrix)):
                self.unit_onset.append(Calculate_Z_Score(matrix[i][50:100], matrix[i][100:200]))
                self.unit_offset.append(Calculate_Z_Score(matrix[i][400:450], matrix[i][460:560]))
            
            onset, offset = np.array(self.unit_onset), np.array(self.unit_offset)
            self.unit_type = np.array(['none' for i in range(len(self.unit_onset))])
            self.unit_type[np.where((onset > 3) & (offset < 3))] = 'on'
            self.unit_type[np.where((onset < 3) & (offset > 3))] = 'off'
            self.unit_type[np.where((onset > 3) & (offset > 3))] = 'both'
            self.unit_type[np.where((onset < 3) & (offset < 3))] = 'none'
                
        meta_psth = np.zeros((2, 10, 1000))
        meta_psth = np.concatenate((meta_psth,
                                    self.response['sig_psth'][:,:,:,:].mean(axis=2)),
                                    axis=0)
        Calculate_Onset_Offset(meta_psth[2:][:,0,:])
        return meta_psth[2:]
            
    def Get_Pop_Response_Standardized(self):
        def Normalize(X): #Normalize each row
            X = X[0]
            baseline_mean = np.mean(X[50:100])
            whole_std = max(abs(X-baseline_mean))
            if whole_std == 0: return X - baseline_mean
            return (X - baseline_mean) / whole_std

        meta_psth_z = np.zeros(self.pop_response.shape)
        for i in range(self.pop_response.shape[0]):
            for j in range(self.pop_response.shape[1]):
                meta_psth_z[i,j] = Normalize(self.pop_response[i,j].reshape(1,-1)) # 1*200
                
        return meta_psth_z
    
    def Save_File(self):
        recording_ = copy.deepcopy(self)
        recording_.sorting = None
        
        with open(recordingpath + self.rec_name + '.pickle', 'wb') as file:
            pickle.dump(recording_, file)

