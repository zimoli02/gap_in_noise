import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pickle
import copy

def Flip(PC):
    max_idx = np.argsort(abs(PC)[100:125])[::-1]
    if PC[100:125][max_idx[0]] < 0: return True 
    else: return False


class DynamicalSystem:
    def __init__(self, group, train_gap_idx):
        self.group = group 
        self.gap_idx = train_gap_idx 
        
        self.gap_dur = None
        self.PCs = None
        self.OnS = None
        self.OffS = None

        self.times = np.arange(1000)
        self.dt = 1 
        
        self.Nt = None
        self.W = None 
        self.OnRe = None 
        self.OffRe = None 
        
        self.S_on, self.S_off = 60, 10
        if self.group.hearing_type == 'HL': self.S_on = 50
        
        self.tau_I_on, self.tau_I_on_coef = 10, 1
        self.tau_A_on, self.tau_A_on_coef = 10, 1
        self.delay_on, self.delay_on_coef = 5, 1
        
        self.tau_I_off, self.tau_I_off_coef = 5, 1
        self.tau_A_off, self.tau_A_off_coef = 50, 1
        self.delay_off, self.delay_off_coef = 10, 1
        
        self.Set_Gap_Dependent_Params()
        self.Init_Params()
        self.N = np.zeros((3, len(self.times)))
        
        self.train_gap_idx, self.train_start, self.train_end = 8, 0, 350 + self.gap_dur
        self.validate_gap_idx, self.validate_start, self.validate_end = 8, 0, 350 + self.gap_dur
        self.lr = 0.001
        if self.group.geno_type == 'Df1': self.lr = 0.005
        
        
    def Set_Gap_Dependent_Params(self):
        self.gap_dur = round(self.group.gaps[self.gap_idx]*1000)
        self.Get_PCs()
        
        self.OnS = self.Get_Input(self.gap_idx, self.tau_I_on * self.tau_I_on_coef, self.tau_A_on* self.tau_A_on_coef, round(self.delay_on * self.delay_on_coef), invert = False)
        self.OffS = self.Get_Input(self.gap_idx, self.tau_I_off * self.tau_I_off_coef, self.tau_A_off * self.tau_A_off_coef, round(self.delay_off * self.delay_off_coef), invert = True)
        
    def Flip(self,PC):
        max_idx = np.argsort(abs(PC)[100:125])[::-1]
        if PC[100:125][max_idx[0]] < 0: return True 
        else: return False
        
    def Get_PCs(self):
        PC, PCs = [0,1,2],[]
        for j in range(len(PC)):
            scores = self.group.pca.score_per_gap[PC[j]]
            
            score_per_gap = scores[self.gap_idx]
            #score_per_gap = (score_per_gap-np.mean(score_per_gap[:100]))/np.max(abs(score_per_gap))
            if Flip(score_per_gap): score_per_gap = score_per_gap * (-1)
            PCs.append(score_per_gap)
        self.PCs = np.array(PCs)
        
    def Get_Input(self, gap_idx, tau_I, tau_A, delay, invert = False):
        def Get_w(N, tau):
            w = [np.exp(-i/tau) for i in range(N)]
            w = w/np.sum(w)
            return np.array(w)

        def Get_rI(S, tau_I):
            rI = np.zeros(len(S))
            N = max(1, round(5*tau_I))
            wI = Get_w(N, tau_I)
            for t in range(0, len(S)):
                if t < N:
                    rI[t] = np.sum(wI[:t][::-1] * S[:t])
                else:
                    rI[t] = np.sum(wI[::-1] * S[t-N:t])
            return rI

        def Get_rA(S, rI, tau_A):
            rA = np.zeros(len(S))
            M = round(5*tau_A)
            wA = Get_w(M, tau_A)
            for t in range(0, len(S)):
                if t < M:
                    rA[t] = 1 / (1 + np.sum(wA[:t][::-1] * rI[:t]))
                else:
                    rA[t] = 1 / (1 + np.sum(wA[::-1] * rI[t-M:t]))
            return rA

        def Get_rIA(S, tau_I, tau_A):
            rI = Get_rI(S, tau_I)
            rA = Get_rA(S, rI, tau_A)
            rIA = np.zeros(len(S))
            for t in range(0, len(S)):
                rIA[t] = rI[t] * rA[t]
            return rI, rA, rIA
        
        gap_dur = round(self.group.gaps[gap_idx]*1000)
        S = np.zeros(1000) + self.S_off
        for t in range(100 + delay, 100 + 250+ delay): S[t] = self.S_on 
        for t in range(350+gap_dur + delay, 350+gap_dur+100+ delay): S[t] = self.S_on
        rI, rA, rIA = Get_rIA(S, tau_I, tau_A)
        if not invert:
            rIA -= np.mean(rIA[50:100])
        if invert: 
            rIA *= -1
            rIA -= np.mean(rIA[300:350])
        for i in range(len(rIA)): 
            if rIA[i] < 0: rIA[i] = 0
        for i in range(100+delay+1): rIA[i] = 0
        return rIA
    
    def Get_Input_torch(self, gap_idx, tau_I, tau_A, delay, invert=False):
        def Get_w_torch(N, tau):
            # Create exponential decay weights using torch
            N = max(1, int(N))
            indices = torch.arange(N, dtype=torch.float32)
            w = torch.exp(-indices/tau)
            w = w/torch.sum(w)  # Normalize
            return w

        def Get_rI_torch(S, tau_I):
            N = int(5*tau_I)  # Convert to int for indexing
            wI = Get_w_torch(N, tau_I)
            rI = torch.zeros_like(S)
            
            # Compute convolution for each time step
            for t in range(len(S)):
                if t < N:
                    if t > 0:  # Avoid empty slice
                        rI[t] = torch.sum(wI[:t].flip(0) * S[:t])
                else:
                    rI[t] = torch.sum(wI.flip(0) * S[t-N:t])
            return rI

        def Get_rA_torch(S, rI, tau_A):
            M = int(5*tau_A)  # Convert to int for indexing
            wA = Get_w_torch(M, tau_A)
            rA = torch.zeros_like(S)
            
            # Compute adaptive component
            for t in range(len(S)):
                if t < M:
                    if t > 0:  # Avoid empty slice
                        weighted_sum = torch.sum(wA[:t].flip(0) * rI[:t])
                        rA[t] = 1 / (1 + weighted_sum)
                else:
                    weighted_sum = torch.sum(wA.flip(0) * rI[t-M:t])
                    rA[t] = 1 / (1 + weighted_sum)
            return rA

        def Get_rIA_torch(S, tau_I, tau_A):
            rI = Get_rI_torch(S, tau_I)
            rA = Get_rA_torch(S, rI, tau_A)
            rIA = rI * rA
            return rI, rA, rIA

        # Convert delay to int for indexing
        delay = int(delay.item())
        
        # Create input signal S
        S = torch.zeros(1000, dtype=torch.float32)
        S.fill_(self.S_off)  # S_off = 1
        
        # Set S_on periods
        S[100 + delay:100 + 250 + delay] = self.S_on 
        gap_dur = int(round(self.group.gaps[gap_idx]*1000))
        S[350 + gap_dur + delay:350 + gap_dur + 100 + delay] = self.S_on
        
        # Get response components
        rI, rA, rIA = Get_rIA_torch(S, tau_I, tau_A)
        
        # Process rIA based on invert flag
        if not invert:
            baseline = torch.mean(rIA[50:100])
            rIA = rIA - baseline
        else:
            rIA = -rIA
            baseline = torch.mean(rIA[300:350])
            rIA = rIA - baseline
        
        # Apply non-negative constraint
        rIA = torch.clamp(rIA, min=0)
        
        # Zero out initial period
        rIA[:100+delay+1] = 0
        
        return rIA
    
    def Init_Params(self):
        ## Timescale
        self.Nt = np.array(
            [
                [0.15],
                [0.40],
                [0.20]
            ]
        ) 
        
        ## Connections
        self.W = np.array(
            [
                [-0.12, -0.02, -0.20],
                [-0.05, -0.09, 0.03],
                [-0.20, 0.03, -0.9]
            ]
        ) 

        ## Response to Stimuli
        self.OnRe = np.array(
            [
                [1.3],
                [0.7],
                [1.5]
            ]
        ) 
        self.OffRe = np.array(
            [
                [0.2],
                [-0.02],
                [1.45]
            ]
        ) 
        
    def Optimize_Params(self):
        def Calculate_Validate_Loss():
            temp_self = copy.deepcopy(self)  # Creates a deep copy
            temp_self.gap_idx = self.validate_gap_idx
            temp_self.Set_Gap_Dependent_Params()
            temp_self.Run()
            return np.mean((temp_self.N[:, self.validate_start:self.validate_end] 
                            - temp_self.PCs[:, self.validate_start:self.validate_end])**2)

        def Detach():
            self.W = W.detach().numpy()
            self.OnRe = OnRe.detach().numpy()
            self.OffRe = OffRe.detach().numpy()
            self.Nt = Nt.detach().numpy()
            self.tau_I_on_coef = tau_I_on_coef.detach().numpy()
            self.tau_A_on_coef = tau_A_on_coef.detach().numpy()
            self.tau_I_off_coef = tau_I_off_coef.detach().numpy()
            self.tau_A_off_coef = tau_A_off_coef.detach().numpy()
            self.delay_on_coef = delay_on_coef.detach().numpy()
            self.delay_off_coef = delay_off_coef.detach().numpy()
        
        self.gap_idx = self.train_gap_idx
        self.Set_Gap_Dependent_Params()
        PCs = torch.tensor(self.PCs[:3, self.train_start:self.train_end], dtype=torch.float32)

        # Initialize the parameters as torch tensors with requires_grad=True
        W = torch.tensor(self.W, dtype=torch.float32, requires_grad=True)
        OnRe = torch.tensor(self.OnRe, dtype=torch.float32, requires_grad=True)
        OffRe = torch.tensor(self.OffRe, dtype=torch.float32, requires_grad=True)
        Nt = torch.tensor(self.Nt, dtype=torch.float32, requires_grad=False)
        
        ## Parameters for OnS and OffS computation
        tau_I_on_coef = torch.tensor(self.tau_I_on_coef, dtype=torch.float32, requires_grad=True)
        tau_A_on_coef = torch.tensor(self.tau_A_on_coef, dtype=torch.float32, requires_grad=True)
        tau_I_off_coef = torch.tensor(self.tau_I_off_coef, dtype=torch.float32, requires_grad=True)
        tau_A_off_coef = torch.tensor(self.tau_A_off_coef, dtype=torch.float32, requires_grad=True)
        delay_on_coef = torch.tensor(self.delay_on_coef, dtype=torch.float32, requires_grad=True)
        delay_off_coef = torch.tensor(self.delay_off_coef, dtype=torch.float32, requires_grad=True)


        # Define the optimizer
        optimizer = torch.optim.Adam([
            W, OnRe, OffRe,
            tau_I_on_coef, tau_A_on_coef, tau_I_off_coef, tau_A_off_coef, delay_on_coef, delay_off_coef
        ], lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)

        
        # Number of iterations
        num_iterations = 1000
        
        # Set threshold for minimum change in loss
        patience = 5  # Number of iterations to wait
        min_delta = 1e-4  # Minimum change in loss to be considered significant
        best_loss = float('inf')
        validate_patience_counter = 0
        train_patience_counter = 0

        # Optimization loop
        self.train_loss, self.validate_loss = [], []
        total_steps = num_iterations * (self.train_end - self.train_start)
        with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
            for iter in range(num_iterations):
                optimizer.zero_grad()  # Zero the gradients for each iteration
                
                OnS = self.Get_Input_torch(
                    self.gap_idx, 
                    self.tau_I_on * tau_I_on_coef, 
                    self.tau_A_on * tau_A_on_coef, 
                    self.delay_on * delay_on_coef, 
                    invert=False
                )
                
                OffS = self.Get_Input_torch(
                    self.gap_idx, 
                    self.tau_I_off * tau_I_off_coef, 
                    self.tau_A_off * tau_A_off_coef, 
                    self.delay_off * delay_off_coef,
                    invert=True
                )

                # Initialize N with zeros (as a tensor)
                N = torch.zeros((3, self.train_end - self.train_start))

                # Calculate N over time steps
                N[:,[0]] = PCs[:,[0]]
                for t in range(1, self.train_end - self.train_start):
                    #noise = torch.normal(0, np.sqrt(0.001), size=(3, 1))  # Add Gaussian noise
                    N[:, [t]] = N[:, [t-1]] + (W @ N[:, [t-1]] + OnRe * OnS[self.train_start + t-1] + OffRe * OffS[self.train_start + t-1]) * Nt
                    pbar.update(1)

                # Compute MSE loss
                loss = torch.mean((N - PCs) ** 2)
                self.train_loss.append(loss.item())
                Detach()
                self.validate_loss.append(Calculate_Validate_Loss())
                if iter % 50 == 0: print(f'Iter {iter}, Training Loss: {round(loss.item(), 5)}, Validation Loss: {round(self.validate_loss[-1], 5)}')

                # Backpropagate
                loss.backward()
                optimizer.step()  # Update parameters
                scheduler.step(loss) # Update learning rate based on loss

                if iter > 2:
                    if best_loss < self.validate_loss[-1]:
                        validate_patience_counter += 1
                        if validate_patience_counter >= patience:
                            print(f"Early Stopped: Validate Loss Increased for {patience} iterations")
                            break
                    else:
                        validate_patience_counter = 0
                        best_loss = self.validate_loss[-1]

                    if  abs(self.train_loss[-1] - self.train_loss[-2]) < min_delta:
                        train_patience_counter += 1
                        if train_patience_counter >= patience:
                            print(f"Converged: Train Loss hasn't improved by {min_delta} for {patience} iterations")
                            break
                    else:
                        train_patience_counter = 0

        Detach()
        self.train_loss = np.array(self.train_loss)
        self.validate_loss = np.array(self.validate_loss)

        print('----------------Iter ' + str(iter) + '----------------')
        print('Integration for On-response = ' + str(self.tau_I_on * self.tau_I_on_coef) + 'ms')
        print('Adaptation for On-response = ' + str(self.tau_A_on * self.tau_A_on_coef) + 'ms')
        print('Delay for On-response = ' + str(self.delay_on * self.delay_on_coef) + 'ms')
        print("")
        print('Integration for Off-response = ' + str(self.tau_I_off * self.tau_I_off_coef) + 'ms')
        print('Adaptation for Off-response = ' + str(self.tau_A_off * self.tau_A_off_coef) + 'ms')
        print('Delay for Off-response = ' + str(self.delay_off * self.delay_off_coef) + 'ms')

    def Cross_Validate_Optimization(self):
        def Select_Period(length_session):
            gap_idx = np.random.randint(10)
            start = np.random.randint(50, 550+round(self.group.gaps[gap_idx]*1000)-length_session)
            end = start + length_session
            return gap_idx, start, end
            
        def Calculate_MSE_for_All_Trials():
            mse_per_trials = []
            for gap_idx in np.arange(10):
                self.gap_idx = gap_idx
                self.Set_Gap_Dependent_Params()
                self.Run()
                mse_per_trials.append(np.mean((self.N - self.PCs)**2))
            return np.mean(mse_per_trials)
        
        def Reset_Params():
            for i in range(len(param_groups)): setattr(self, param_groups[i], init_params[i])

        num_session = 100
        length_session = 400
        self.mse_per_session = []
        param_groups = ['W', 'OnRe', 'OffRe', 'tau_I_on_coef', 'tau_A_on_coef', 'tau_I_off_coef', 'tau_A_off_coef', 'delay_on_coef', 'delay_off_coef']
        init_params = [getattr(self, param_group) for param_group in param_groups]
        self.train_params = []
        for session in range(num_session):
            print('Session ' + str(session+1) + ": Start Training")
            Reset_Params()
            self.train_gap_idx, self.train_start, self.train_end = Select_Period(length_session)
            self.validate_gap_idx, self.test_start, self.test_end = Select_Period(length_session)
            
            self.Optimize_Params() 
            
            self.train_params.append([getattr(self, param_group) for param_group in param_groups])
            self.train_params[-1].append(self.train_loss)
            self.train_params[-1].append(self.validate_loss)
            
            test_loss = Calculate_MSE_for_All_Trials()
            self.mse_per_session.append(test_loss)
            print('Average MSE from Model Trained by Session ' + str(session+1)  + ': ' + str(test_loss) + '\n')
            
            params = {'W':self.W, 
                'OnRe': self.OnRe, 
                'OffRe':self.OffRe, 
                'Nt':self.Nt, 
                'tau_I_on':self.tau_I_on * self.tau_I_on_coef, 
                'tau_A_on':self.tau_A_on * self.tau_A_on_coef,
                'delay_on':self.delay_on * self.delay_on_coef,
                'tau_I_off':self.tau_I_off * self.tau_I_off_coef, 
                'tau_A_off':self.tau_A_off * self.tau_A_off_coef,
                'delay_off':self.delay_off * self.delay_off_coef,
                'train_loss': self.train_loss,
                'validate_loss': self.validate_loss,
                'test_loss': test_loss
                }
        
            file_name = '/Volumes/Research/GapInNoise/Data/Model_New/'+ self.group.geno_type + '_' + self.group.hearing_type + '_' + str(session+1) + '.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(params, f)

            self.Set_Params_of_Least_Loss()
            file_name = '/Volumes/Research/GapInNoise/Data/TrainedModel/'+ self.group.geno_type + '_' + self.group.hearing_type + '.pickle'
            with open(file_name, 'wb') as file:
                pickle.dump(self, file)

        self.Set_Params_of_Least_Loss()
        
    def Set_Params_of_Least_Loss(self):
        param_groups = ['W', 'OnRe', 'OffRe', 'tau_I_on_coef', 'tau_A_on_coef', 'tau_I_off_coef', 'tau_A_off_coef', 'delay_on_coef', 'delay_off_coef']
        best_model_idx = np.argsort(self.mse_per_session)[0]
        for i in range(len(param_groups)): setattr(self, param_groups[i], self.train_params[best_model_idx][i])
        self.train_loss =self.train_params[best_model_idx][-2]
        self.validate_loss = self.train_params[best_model_idx][-1]
    
    def Set_Params_Median(self):
        Train_Params = {'W':{}, 'OnRe':{}, 'OffRe':{}, 'tau_I_on_coef':{}, 'tau_A_on_coef':{},'delay_on_coef':{},'tau_I_off_coef':{}, 'tau_A_off_coef':{},'delay_off_coef':{}}
        train_params_labels = ['W', 'OnRe', 'OffRe', 'tau_I_on_coef', 'tau_A_on_coef', 'delay_on_coef','tau_I_off_coef', 'tau_A_off_coef','delay_off_coef']
        for train_params_label in train_params_labels: Train_Params[train_params_label] = []
        for i in range(len(train_params_labels)):
            for j in range(len(self.train_params)):
                Train_Params[train_params_labels[i]].append(self.train_params[j][i])
            Train_Params[train_params_labels[i]] = np.array(Train_Params[train_params_labels[i]])
        
        for rol in range(3):
            for col in range(3):
                self.W[rol, col] = np.median(Train_Params['W'][:,rol,col])

        for rol in range(3):
            self.OnRe[rol] = np.median(Train_Params['OnRe'][:,rol])
            self.OffRe[rol] = np.median(Train_Params['OffRe'][:,rol])
        
        self.tau_I_on_coef = np.median(Train_Params['tau_I_on_coef'])
        self.tau_A_on_coef = np.median(Train_Params['tau_A_on_coef'])
        self.delay_on_coef = np.median(Train_Params['delay_on_coef'])

        self.tau_I_off_coef = np.median(Train_Params['tau_I_off_coef'])
        self.tau_A_off_coef = np.median(Train_Params['tau_A_off_coef'])
        self.delay_off_coef = np.median(Train_Params['delay_off_coef'])
            
    def Run(self):
        self.Get_PCs()
        self.OnS = self.Get_Input(self.gap_idx, self.tau_I_on * self.tau_I_on_coef, self.tau_A_on* self.tau_A_on_coef, round(self.delay_on * self.delay_on_coef), invert = False)
        self.OffS = self.Get_Input(self.gap_idx, self.tau_I_off * self.tau_I_off_coef, self.tau_A_off * self.tau_A_off_coef, round(self.delay_off * self.delay_off_coef), invert = True)
        
        self.N = np.zeros((3, len(self.times)))
        for t in range(1,len(self.times)):
            noise = np.random.normal(0, np.sqrt(0.001), (3, 1))
            self.N[:,[t]] = self.N[:,[t-1]] +(self.W@self.N[:,[t-1]] + self.OnRe*self.OnS[t-1] + self.OffRe*self.OffS[t-1])*self.Nt + noise

        # Normalize()