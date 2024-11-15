import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

def Flip(PC):
    max_idx = np.argsort(abs(PC)[100:125])[::-1]
    if PC[100:125][max_idx[0]] < 0: return True 
    else: return False
        
class DynamicalSystem_Exponential:
    def __init__(self, group, gap_idx):
        self.group = group 
        self.gap_idx = gap_idx 
        
        self.gap_dur = None
        self.PCs = None
        self.SoundS = None
        self.Set_Gap_Dependent_Params()
        
        self.times = np.arange(1000)
        self.dt = 1 
        
        self.Nt = None
        self.W = None 
        self.OnRe = None 
        self.OffRe = None 
        
        self.tau_on, self.tau_off = 1, 1
        self.OnS, self.OffS = None, None
        
        self.Init_Params()
        self.x, self.y, self.z = None, None, None
        self.N = np.zeros((3, len(self.times)))
        
        self.opti_start, self.opti_end = 0, 1000
        self.lr = 0.007
        self.opti_corre = []
        
    def Set_Gap_Dependent_Params(self):
        self.gap_dur = round(self.group.gaps[self.gap_idx]*1000)
        self.SoundS = np.array(self.group.gaps_label[self.gap_idx]) * 1
        self.Get_PCs()
  
    def Get_PCs(self):
        PC, PCs = [0,1,2],[]
        for j in range(len(PC)):
            scores = self.group.pca.score_per_gap[PC[j]]
            
            score_per_gap = scores[self.gap_idx]
            score_per_gap = (score_per_gap-np.mean(score_per_gap[:100]))/np.max(abs(score_per_gap))
            if Flip(score_per_gap): score_per_gap = score_per_gap * (-1)
            PCs.append(score_per_gap)
        self.PCs = np.array(PCs)

    def Get_Input(self):
        def exponential_decay_np(start, t, tau):
            return np.exp(-(t-start) / tau)
        
        starts = [0]
        start = 0
        for i in range(1, len(self.times)):
            if self.SoundS[i] != self.SoundS[i-1]: start = i 
            starts.append(start)
            
        self.OnS, self.OffS = np.zeros(len(self.times)), np.zeros(len(self.times))
        
        for t in range(100, len(self.times)):
            if self.SoundS[t] == 1:  # Sound is on, use tau_on
                self.OnS[t] = exponential_decay_np(starts[t], t, self.tau_on)
            else:  # Sound is off, use tau_off
                self.OffS[t] = exponential_decay_np(starts[t], t, self.tau_off)

    def Get_Input_torch(self, OnS, OffS, tau_on, tau_off):
        def exponential_decay(start, t, tau):
            return torch.exp(-(t-start) / tau)

        starts = [0]
        start = 0
        for i in range(1, len(self.times)):
            if self.SoundS[i] != self.SoundS[i-1]: start = i 
            starts.append(start)
        
        # Precompute the exponential decay for sound-on and sound-off periods
        for t in range(100, len(self.times)):
            if self.SoundS[t] == 1:  # Sound is on, use tau_on
                OnS[t] = exponential_decay(starts[t],torch.tensor(t, dtype=torch.float32), tau_on*100)
            else:  # Sound is off, use tau_off
                OffS[t] = exponential_decay(starts[t], torch.tensor(t, dtype=torch.float32), tau_off*50)
        
        return OnS, OffS
        
    def Init_Params(self, random = False):
     
        if random:
            self.Nt = np.random.uniform(0, 1, size=(3, 1))
            self.W = np.random.uniform(-1, 1, size=(3, 3))
            self.OnRe = np.random.uniform(-1, 1, size=(3, 1))
            self.OffRe = np.random.uniform(-1, 1, size=(3, 1))
            
        else:
            ## Timescale
            self.Nt = np.array(
                [
                    [0.32],
                    [0.40],
                    [0.71]
                ]
            ) 
            ## Connections
            self.W = np.array(
                [
                    [-0.03, -0.06, 0.44],
                    [-0.65, -0.06, 0.45],
                    [-0.35, -0.23, -0.1]
                ]
            ) 

            ## Response to Stimuli
            self.OnRe = np.array(
                [
                    [0.68],
                    [0.79],
                    [0.46]
                ]
            ) 
            self.OffRe = np.array(
                [
                    [-0.28],
                    [-0.03],
                    [0.92]
                ]
            ) 
        
    def Optimize_Params(self):
        def pearson_corr(x, y):
            x_mean = torch.mean(x)
            y_mean = torch.mean(y)
            x_centered = x - x_mean
            y_centered = y - y_mean
            covariance = torch.sum(x_centered * y_centered)
            x_std = torch.sqrt(torch.sum(x_centered ** 2))
            y_std = torch.sqrt(torch.sum(y_centered ** 2))
            correlation = covariance / (x_std * y_std)
            return correlation
        
        def constrain_W_columns(W):
            with torch.no_grad():
                # First column positive
                W[:, 0] = -torch.abs(W[:, 0])
                # Second column negative
                W[:, 1] = -torch.abs(W[:, 1])
                # Third column positive
                W[:, 2] = torch.abs(W[:, 2])
            return W
        
        # Initialize the parameters as torch tensors with requires_grad=True
        W = torch.tensor(self.W, dtype=torch.float32, requires_grad=True)
        OnRe = torch.tensor(self.OnRe, dtype=torch.float32, requires_grad=True)
        OffRe = torch.tensor(self.OffRe, dtype=torch.float32, requires_grad=True)
        Nt = torch.tensor(self.Nt, dtype=torch.float32, requires_grad=True)
        OnS = torch.tensor(self.OnS, dtype=torch.float32)
        OffS = torch.tensor(self.OffS, dtype=torch.float32)
        tau_on = torch.tensor(self.tau_on, dtype=torch.float32, requires_grad=True)  # Time constant for OnRe_D decay
        tau_off = torch.tensor(self.tau_off, dtype=torch.float32, requires_grad=True)  # Time constant for OffRe_D decay

        # Define the optimizer
        optimizer = torch.optim.Adam([W, OnRe, OffRe, Nt, tau_on, tau_off], lr= self.lr)
        #W = constrain_W_columns(W)
         
        # Number of iterations
        num_iterations = 1000

        # Store the maximum correlation and max_N during training
        max_corre = -1

        # Optimization loop
        total_steps = num_iterations * (len(self.times))
        with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
            for iter in range(num_iterations):
                optimizer.zero_grad()  # Zero the gradients for each iteration
                
                OnS, OffS = self.Get_Input_torch(OnS, OffS, tau_on, tau_off)

                # Initialize N with zeros (as a tensor)
                N = torch.zeros((3, len(self.times)))
                
                # Calculate N over time steps
                for t in range(1, len(self.times)):
                    noise = torch.normal(0, np.sqrt(0.001), size=(3, 1))  # Add Gaussian noise
                    N[:, [t]] = N[:, [t-1]] + (W @ N[:, [t-1]] + OnRe * OnS[t-1] + OffRe * OffS[t-1]) * Nt + noise
                    pbar.update(1)
                    
                # Calculate correlation with PCs (row-wise) using differentiable correlation in PyTorch
                corre = []
                for i in range(3): corre.append(pearson_corr(N[i][self.opti_start:self.opti_end], 
                                                             torch.tensor(self.PCs[i][self.opti_start:self.opti_end], 
                                                                          dtype=torch.float32)))
                average_corre = torch.mean(torch.stack(corre))

                # Convert to a loss (we want to maximize correlation, so minimize -average_corre)
                loss = -average_corre

                # Backpropagate
                loss.backward()
                optimizer.step()  # Update parameters
                #W = constrain_W_columns(W)

                # Keep track of the maximum correlation and best N
                if average_corre.item() > max_corre:
                    max_corre = average_corre.item()
                    self.opti_corre.append([iter, max_corre])
                    self.W = W.detach().numpy()
                    self.OnRe = OnRe.detach().numpy()
                    self.OffRe = OffRe.detach().numpy()
                    self.Nt = Nt.detach().numpy()
                    self.tau_on = tau_on.detach().numpy()*100
                    self.tau_off = tau_off.detach().numpy()*50

                # Check for convergence (if correlation change is very small)
                if len(self.opti_corre) > 3 and abs(self.opti_corre[-1][1] - self.opti_corre[-2][1]) < 5e-5:
                    print("Convergence reached.")
                    break

        self.opti_corre = np.array(self.opti_corre)

    def Run(self): 
        self.Get_Input()
               
        for t in range(1,len(self.times)):
            noise = np.random.normal(0, np.sqrt(0.001), (3, 1))
            self.N[:,[t]] = self.N[:,[t-1]] +(self.W@self.N[:,[t-1]] + self.OnRe*self.OnS[t-1] + self.OffRe*self.OffS[t-1] )*self.Nt + noise

        # Normalize
        for i in range(3):
            self.N[i] = (self.N[i]-np.mean(self.N[i][:100]))
            self.N[i] = self.N[i] / np.max(abs(self.N[i]))
            if self.Flip(self.N[i]): self.N[i] *= -1
        
        self.x, self.y, self.z = self.N[0], self.N[1], self.N[2]


class DynamicalSystem_Complex:
    def __init__(self, group, gap_idx):
        self.group = group 
        self.gap_idx = gap_idx 
        
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
        self.tau_I_on, self.tau_I_on_coef = 12, 1
        self.tau_A_on, self.tau_A_on_coef = 6, 1
        self.tau_I_off, self.tau_I_off_coef = 2, 1
        self.tau_A_off, self.tau_A_off_coef = 20, 1
        self.delay_on, self.delay_on_coef = 5, 1
        self.delay_off, self.delay_off_coef = 10, 1
        
        self.Set_Gap_Dependent_Params()
        self.Init_Params()
        self.x, self.y, self.z = None, None, None
        self.N = np.zeros((3, len(self.times)))
        
        self.opti_start, self.opti_end = 0, 1000
        self.lr = 0.007
        self.opti_loss, self.full_loss = [], []
            
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
            if self.Flip(score_per_gap): score_per_gap = score_per_gap * (-1)
            PCs.append(score_per_gap)
        self.PCs = np.array(PCs)
        
    def Get_Input(self, gap_idx, tau_I, tau_A, delay, invert = False):
        def Get_w(N, tau):
            w = [np.exp(-i/tau) for i in range(N)]
            w = w/np.sum(w)
            return np.array(w)

        def Get_rI(S, tau_I):
            rI = np.zeros(len(S))
            N = round(5*tau_I)
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
            N = int(N)  # Convert to int for range
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
    
    def Init_Params(self, random = False):
        
        if random:
            self.Nt = np.random.uniform(0, 1, size=(3, 1))
            self.W = np.random.uniform(-1, 1, size=(3, 3))
            self.OnRe = np.random.uniform(-1, 1, size=(3, 1))
            self.OffRe = np.random.uniform(-1, 1, size=(3, 1))
            
        else:
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
                    [-0.06, -0.05, -0.04],
                    [-0.12, -0.18, -0.02],
                    [-0.22, -0.35, -0.2]
                ]
            ) 

            ## Response to Stimuli
            self.OnRe = np.array(
                [
                    [1.05],
                    [1.37],
                    [2.46]
                ]
            ) 
            self.OffRe = np.array(
                [
                    [-0.06],
                    [0.22],
                    [1.18]
                ]
            ) 
        
    def Optimize_Params(self):
        # Initialize the parameters as torch tensors with requires_grad=True
        W = torch.tensor(self.W, dtype=torch.float32, requires_grad=True)
        OnRe = torch.tensor(self.OnRe, dtype=torch.float32, requires_grad=True)
        OffRe = torch.tensor(self.OffRe, dtype=torch.float32, requires_grad=True)
        Nt = torch.tensor(self.Nt, dtype=torch.float32, requires_grad=True)
        
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
        ], lr=self.lr) #exclude Nt
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        
        # Number of iterations
        num_iterations = 1000
        
        # Set threshold for minimum change in loss
        patience = 5  # Number of iterations to wait
        min_delta = 5e-5  # Minimum change in loss to be considered significant
        best_loss = float('inf')
        patience_counter = 0
        self.opti_loss, self.full_loss = [], []

        # Optimization loop
        total_steps = num_iterations * (len(self.times))
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
                N = torch.zeros((3, len(self.times)))
                
                # Calculate N over time steps
                for t in range(1, len(self.times)):
                    #noise = torch.normal(0, np.sqrt(0.001), size=(3, 1))  # Add Gaussian noise
                    N[:, [t]] = N[:, [t-1]] + (W @ N[:, [t-1]] + OnRe * OnS[t-1] + OffRe * OffS[t-1]) * Nt
                    pbar.update(1)
                    
                # Calculate correlation with PCs (row-wise) using differentiable correlation in PyTorch
                target = torch.tensor(self.PCs[:3, self.opti_start:self.opti_end], dtype=torch.float32)
                predicted = N[:, self.opti_start:self.opti_end]
                
                # Compute MSE loss
                loss = torch.mean((predicted - target) ** 2)
                self.full_loss.append([iter, loss.item()])
                if iter % 50 == 0: print('Iter ' + str(iter), loss.item())

                # Backpropagate
                loss.backward()
                optimizer.step()  # Update parameters
                scheduler.step(loss) # Update learning rate based on loss

                # Keep track of the maximum correlation and best N
                if iter == 0 or loss.item() < min_loss:
                    min_loss = loss.item()
                    self.opti_loss.append([iter, min_loss])
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

                '''
                # Check for convergence (if correlation change is very small)
                if len(self.opti_loss) > 3 and abs(self.opti_loss[-1][1] - self.opti_loss[-2][1]) < 1e-3:
                    print("Convergence reached.")
                    break
                '''
                
                if abs(loss.item() - best_loss) < min_delta:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Converged: Loss hasn't improved by {min_delta} for {patience} iterations")
                        break
                else:
                    patience_counter = 0
                    best_loss = min(best_loss, loss.item())
            
        
        self.opti_loss = np.array(self.opti_loss)
        self.full_loss = np.array(self.full_loss)

        print('----------------Iter ' + str(iter) + '----------------')
        print('Integration for On-response = ' + str(self.tau_I_on * self.tau_I_on_coef) + 'ms')
        print('Adaptation for On-response = ' + str(self.tau_A_on * self.tau_A_on_coef) + 'ms')
        print('Delay for On-response = ' + str(self.delay_on * self.delay_on_coef) + 'ms')
        print("")
        print('Integration for Off-response = ' + str(self.tau_I_off * self.tau_I_off_coef) + 'ms')
        print('Adaptation for Off-response = ' + str(self.tau_A_off * self.tau_A_off_coef) + 'ms')
        print('Delay for Off-response = ' + str(self.delay_off * self.delay_off_coef) + 'ms')

    def Run(self): 
        def Normalize():
            for i in range(3):
                self.N[i] = (self.N[i]-np.mean(self.N[i][:100]))
                self.N[i] = self.N[i] / np.max(abs(self.N[i]))
                if self.Flip(self.N[i]): self.N[i] *= -1
            
        self.OnS = self.Get_Input(self.gap_idx, self.tau_I_on * self.tau_I_on_coef, self.tau_A_on* self.tau_A_on_coef, round(self.delay_on * self.delay_on_coef), invert = False)
        self.OffS = self.Get_Input(self.gap_idx, self.tau_I_off * self.tau_I_off_coef, self.tau_A_off * self.tau_A_off_coef, round(self.delay_off * self.delay_off_coef), invert = True)
        
        self.N = np.zeros((3, len(self.times)))
        for t in range(1,len(self.times)):
            noise = np.random.normal(0, np.sqrt(0.001), (3, 1))
            self.N[:,[t]] = self.N[:,[t-1]] +(self.W@self.N[:,[t-1]] + self.OnRe*self.OnS[t-1] + self.OffRe*self.OffS[t-1])*self.Nt + noise

        #Normalize()
        
        
        self.x, self.y, self.z = self.N[0], self.N[1], self.N[2]