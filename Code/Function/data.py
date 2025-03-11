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
        self.recording_names = self.Get_Group_Recording()
        self.gaps = [0., 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]
        self.gaps_label = self.Get_Gaps_Label()
        
        self.unit_type = np.array([])
        self.unit_id = np.array([[0, 0, 0]]) 
        self.bkg_psth = None
        self.response_per_recording = self.Get_Response_per_Recording()
        self.pop_spikes = self.Get_Pop_Spikes()
        self.pop_response = self.Get_Pop_Response()
        self.pop_response_stand = self.Get_Pop_Response_Standardized()
        
        self.unit_num = self.pop_response.shape[0]

        self.pca = analysis.PCA(self.pop_response_stand)
        self.periods, self.periods_pca = self.Get_PCA_for_periods()
    
    def Get_Group_Recording(self):
        if self.hearing_type == 'NonHL':
            return mouse[(mouse['Geno']==self.geno_type)&(mouse['L_Thres']<42)]['Recording'].values
        else:
            return mouse[(mouse['Geno']==self.geno_type)&(mouse['L_Thres']>42)]['Recording'].values
    
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
        for Exp_name in self.recording_names:
            try:
                with open(recordingpath + Exp_name + '.pickle', 'rb') as file:
                    recording = pickle.load(file)
            except FileNotFoundError:
                recording = Recording(Exp_name)
            response_per_recording[Exp_name] = recording
            self.unit_type = np.concatenate((self.unit_type, recording.unit_type))
            self.unit_id = np.concatenate((self.unit_id, recording.unit_id))
        self.unit_id = self.unit_id[1:]
        return response_per_recording
    
    def Get_Pop_Spikes(self):
        meta_spikes = []
        for gap_idx in range(len(self.gaps)):
            spikes = []
            for Exp_name in self.recording_names:
                spikes += self.response_per_recording[Exp_name].response['spike'][gap_idx]
            meta_spikes.append(spikes)
        return meta_spikes
    
    def Get_Pop_Response(self):
        meta_psth = np.zeros((2, 10, 1000))
        for Exp_name in self.recording_names:
            meta_psth = np.concatenate((meta_psth,
                                        self.response_per_recording[Exp_name].pop_response),
                                       axis=0)

        return np.array(meta_psth[2:])

            
    def Get_Pop_Response_Standardized(self):
        def Normalize(pop_stand): #standardize each row
            # X: ndarray, shape (n_features, n_samples)
            #whole_std = max(abs(X-baseline_mean))
            X = pop_stand.copy()
            baseline_mean = np.mean(X[0:100])
            baseline_std = np.std(X[0:100])
            for j in range(len(X)):
                if abs(X[j]) > 1e-5:  
                    if baseline_std == 0: X[j] -= baseline_mean
                    else: X[j] = (X[j]-baseline_mean)/baseline_std
            
            return X

        meta_psth_z = np.zeros(self.pop_response.shape)
        for i in range(self.pop_response.shape[0]):
            for j in range(self.pop_response.shape[1]):
                #meta_psth_z[i,j] = Normalize(self.pop_response[i,j,:]) # 1*1000
                meta_psth_z[i,j] = self.pop_response[i,j,:]
                
        return meta_psth_z

    def Get_PCA_for_periods(self):
        periods_all = []
        periods_pca_all = []
        for gap_idx in range(10):
            gap_dur = round(self.gaps[gap_idx]*1000+350)

            N1_onset = self.pop_response_stand[:, gap_idx,110:210] # first 100 ms of noise1
            N2_onset = self.pop_response_stand[:, gap_idx, gap_dur + 10:gap_dur+110] # first 100 ms of noise2
            N2_offset = self.pop_response_stand[:,gap_idx, gap_dur + 110: gap_dur + 100 + 110] # first 100 ms of post-N2 silence
            N2_onset_exc_N1_on = N2_onset - N1_onset
            N2_onset_exc_N1_on_off = N2_onset - N1_onset- N2_offset
            
            periods = [N1_onset, N2_onset, N2_onset_exc_N1_on, N2_onset_exc_N1_on_off, N2_offset]
            periods_all.append(periods)
            periods_pca_all.append([analysis.PCA(period, multiple_gaps=False) for period in periods])
        return np.array(periods_all), np.array(periods_pca_all)
            
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
        
        self.unit_type = None
        self.bkg_psth = None
        self.response = self.Get_Neural_Response()
        self.response_per_gap = self.Get_Neural_Response_per_Gap()
        
        self.pop_response = self.Get_Pop_Response()
        self.pop_response_stand = self.Get_Pop_Response_Standardized()
        
        self.Save_File()
         
    def Get_Info(self):
        if mouse[mouse['Recording'] == self.rec_name]['L_Thres'].to_numpy()[0] > 42: self.hearing_type = 'HL'
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
            bkg_t[i] = [self.gap_onset[:,round(1+i*15):round(16+i*15)].min()-3.5,
                        self.gap_onset[:,round(1+i*15):round(16+i*15)].min()-0.5] # -3.5 to -0.5 s before the first gap

        # Background response
        bkg_psth = np.zeros((len(self.unit),bkg_t.shape[0],len(np.arange(bkg_t[0,0],bkg_t[0,1],bin))))
        for idx_unit in range(len(self.unit)):
            for i in range(bkg_t.shape[0]):
                st = self.sorting.get_unit_spike_train(
                    unit_id=self.unit[idx_unit],
                    start_frame=np.round(bkg_t[i,0]*samplerate),
                    end_frame=np.round(bkg_t[i,1]*samplerate))/samplerate-bkg_t[i,0]
                bkg_psth[idx_unit,i] = np.histogram(st,np.arange(0,3+bin,bin))[0]/bin
        bkg_mean = np.mean(bkg_psth.mean(axis=1),axis=1) # Response average to background noise for each unit
        bkg_std = np.std(bkg_psth.mean(axis=1),axis=1)   # Response std to background noise for each unit
        self.bkg_psth = bkg_psth

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

        spikes = [] # 10*Neuron*45*t
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

        sig_psth, self.unit_id = sig_psth[bkg_std>0], self.unit_id[bkg_std>0] #Only want responsive neurons?
        return {'spike':spikes, 'sig_psth':sig_psth}
    
    def Get_Neural_Response_per_Gap(self):      
        response_per_gap = []
        for i in range(len(self.gaps)):
            matrix = np.array([[] for i in range(self.response['sig_psth'].shape[0])])
            for trial in range(self.response['sig_psth'].shape[2]):
                matrix = np.hstack((matrix, self.response['sig_psth'][:, i, trial, :]))
            response_per_gap.append(matrix)
        
        return np.array(response_per_gap)
    
    def Get_Pop_Response(self):
        def Detect_Transient(trasient_psth, upper_thres, lower_thres):
            # For the common case activity increasing with stim
            flag = 0
            for i in range(trasient_psth.shape[0]-1):
                if trasient_psth[i]>=upper_thres:
                    for j in range(i+1, trasient_psth.shape[0]-1):
                        if trasient_psth[i] <= trasient_psth[j]:
                            return 1

            # For the rare case activity reducing with stim
            if lower_thres>0:
                for i in range(trasient_psth.shape[0]-1):
                    if trasient_psth[i] < lower_thres:
                        for j in range(i+1, trasient_psth.shape[0]-1):
                            if trasient_psth[i] >= trasient_psth[j]:
                                return 1
            return flag
        
        def Calculate_Unit_Type(matrix):
            unit_type = []
            for unit_idx in range(len(matrix)):
                response = np.zeros((2, 10))
                for gap_idx in range(10):
                    gap_dur = round(self.gaps[gap_idx]*1000)
                    
                    on_background = matrix[unit_idx, gap_idx, 50:100].reshape(10, -1).sum(axis=1)/5
                    on_period = matrix[unit_idx, gap_idx, 50:150].reshape(20, -1).sum(axis=1)/5
                    mean, std = np.mean(on_background), np.std(on_background)
                    flag = Detect_Transient(on_period[10:], mean + 3*std, mean - 3*std)
                    response[0, gap_idx] = flag

                    off_background = matrix[unit_idx, gap_idx, 400+gap_dur:450+gap_dur].reshape(10, -1).sum(axis=1)/5
                    off_period = matrix[unit_idx, gap_idx, 400+gap_dur:560+gap_dur].reshape(32, -1).sum(axis=1)/5
                    mean, std = np.mean(off_background), np.std(off_background)
                    flag = Detect_Transient(off_period[12:], mean + 2*std, mean - 2*std)
                    response[1, gap_idx] = flag
                if np.mean(response[0]) > 0.6 and np.mean(response[1]) < 0.7: unit_type.append('on')
                if np.mean(response[0]) < 0.7 and np.mean(response[1]) > 0.6: unit_type.append('off')
                if np.mean(response[0]) > 0.6 and np.mean(response[1]) > 0.6: unit_type.append('both')
                if np.mean(response[0]) < 0.7 and np.mean(response[1]) < 0.7: unit_type.append('none')
            return np.array(unit_type)
        
            
        meta_psth = np.zeros((2, 10, 1000))
        meta_psth = np.concatenate((meta_psth,
                                    self.response['sig_psth'][:,:,:,:].mean(axis=2)),
                                    axis=0)
        
        '''# Filter rows based on the standard deviation condition
        filtered_meta_psth = []
        filtered_unit_id = []
        for i in range(len(meta_psth[2:])):
            row = meta_psth[2:][i]
            if np.all(np.std(row[:, :100], axis=1) > 3):
                filtered_meta_psth.append(row)
                filtered_unit_id.append(self.unit_id[i])
        filtered_meta_psth = np.array(filtered_meta_psth)
        filtered_unit_id = np.array(filtered_unit_id)
        
        self.unit_id = filtered_unit_id
        self.unit_type = Calculate_Unit_Type(filtered_meta_psth)
        return filtered_meta_psth'''
        
        self.unit_type = Calculate_Unit_Type(meta_psth[2:])
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

