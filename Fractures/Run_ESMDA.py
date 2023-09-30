import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from Materials.WGAN_Net import netG
from Materials import Forward_HT
import time

outf = './Results_ESMDA'
img_f = outf + '/image'      # output image folder
data_f = outf + '/data'    # output data folder

try:
    os.makedirs(img_f)
    os.makedirs(data_f)
except OSError:
    pass

''' Model Setup '''
Ly = 480; Lx = 480;  Lz = 1
nrow = 96; ncol = 96; nlay = 1
Q = -50.; Rch = 0.001
ztop = 0.; zbot = -10.


''' Location of observations '''
obs_locmat = np.zeros((nlay, nrow, ncol), np.bool_)

h_n_ob = 16      # number of wells
h_spacing = 24         # spacing between two wells
h_Start = 12
h_End = 84  

h_k = 1
for i in range(h_Start,h_End+h_spacing,h_spacing): 
    for j in range(h_Start,h_End+h_spacing,h_spacing): 
        obs_locmat[0, i, j] = 1
        h_k = h_k+1 

Q_locs_idx = np.where(obs_locmat == True)
Q_locs = []
for Q_loc in zip(Q_locs_idx[0], Q_locs_idx[1], Q_locs_idx[2]):
    Q_locs.append(Q_loc)

''' True field ''' 
s_true = np.loadtxt('./Materials/Ref_Fractures.txt')
s_true = s_true*2-1


''' Prepare WGAN generator '''
device = torch.device("cpu")
netG = netG()
netG.load_state_dict(torch.load('./Materials/Weights_Fractures.pth',map_location='cpu'))
netG.to(device)
netG.eval()
torch.set_grad_enabled(False)


''' Forward model parameters '''
mf_params = {'Lx': Lx, 'Ly': Ly,
          'Q': Q, 'Rch': Rch,
          'nlay': nlay, 'nrow': nrow, 'ncol': ncol,
          'zbot': zbot, 'ztop': ztop,
          'obs_locmat': obs_locmat, 'Q_locs': Q_locs}
Forward_Model = Forward_HT.Model(params = mf_params)


''' Observed data '''
obs_true = Forward_Model.run_model(s_true) 
obs_true = obs_true.reshape(-1,1)

err_std = 0.02
CD = np.eye(len(obs_true)) * err_std   # observation error matrix

CD_inv = np.eye(len(obs_true)) * (1/err_std)  # inverse
obs = obs_true + err_std*np.random.randn(len(obs_true),1)
np.savetxt(data_f+'/Observed_Head.txt', obs)


''' define plot function '''
def RMSE(obs,simul_obs):
  nobs = obs.shape[0]
  res = obs - simul_obs
  return np.sqrt(np.sum(res**2)/nobs)

def plot_func(Q_locs,K_mean,sim_mean,RMSE_mean,i=0):
  fig, ax = plt.subplots(1,3,figsize=(15,10))
  im = ax[0].pcolormesh(s_true, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1)
  for Q_loc in Q_locs:                                       # the location of wells
      ax[0].scatter(Q_loc[1],Q_loc[2], marker='o',c='None', edgecolors='yellow', s = 20)
  ax[0].set_aspect('equal', adjustable='box')
  ax[0].xaxis.set_visible(False)
  ax[0].yaxis.set_visible(False)
  ax[0].invert_yaxis()
  ax[0].set_title('True log10K',fontsize=18)
  #fig.colorbar(im, ax=ax[0])

  im = ax[1].pcolormesh(K_mean, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1)
  ax[1].set_aspect('equal', adjustable='box')
  #fig.colorbar(im, ax=ax[0,1])
  ax[1].xaxis.set_visible(False)
  ax[1].yaxis.set_visible(False)
  ax[1].invert_yaxis()
  if i == 0:
    ax[1].set_title('Initial logK',fontsize=18)
  elif i < 0:
    ax[1].set_title('Estimated logK',fontsize=18)
  else:
    ax[1].set_title('Estimated logK at iteration %d' % (i),fontsize=18)

  Min = -10
  Max = 10
  ax[2].plot(obs,sim_mean,'.', markersize=10, color='red')
  ax[2].plot([Min,Max],[Min,Max],'-', color='black')
  ax[2].set_aspect('equal', adjustable='box')
  ax[2].set_xlabel('Observed (m)')
  ax[2].set_ylabel('Simulated (m)')
  ax[2].legend(markerscale=0,handlelength=0,handletextpad=0,
                labels=['RMSE: '+ '%.3f' % RMSE_mean],
                loc='upper left',fontsize=12,edgecolor='k')
  ax[2].set_title('Fitting Error',fontsize=18)
  
  plt.tight_layout()
  #plt.show()
  fig.savefig(img_f+'/result_iter_'+str(i)+'.png',dpi=150, bbox_inches='tight')



''' Data Assimilation '''
zx, zy = 3,3
nr = 200       # number of realizations
itermax = 8   # iteration
alpha = np.ones(itermax)*itermax
latent_z = torch.randn(nr, 1, zx, zy, device=device)     # initial z

Time_start = time.time()
for pp in range(0, itermax):
    print('#################### Iteration %d #####################' % (pp))
    k_array = netG(latent_z).squeeze().numpy()
    np.save(data_f+'/k_array_'+str(pp)+'.npy', k_array)
    K_log = k_array*2-1
    Mean_K = np.array(K_log.mean(axis=0))                    # the mean of K
    Variance_K = np.array(np.var(K_log,axis = 0))               # the variance of K
    np.savetxt(data_f+'/Mean_K_' + str(pp) + '.txt', Mean_K)
    np.savetxt(data_f+'/Variance_K_' + str(pp) + '.txt', Variance_K)
    
    hss = np.zeros([h_n_ob*(h_n_ob-1),nr])
    RMSE_data = np.zeros([nr,1])
    RMSE_K = np.zeros([nr,1])
    
    for r in range(0,4):
        np.savetxt(data_f+'/First_4_K_Iter_' + str(pp) + '_' + str(r+1)+'.txt', K_log[r,:,:])
        plt.subplot(2,2,r+1)
        plt.imshow(K_log[r,:,:],cmap=plt.get_cmap('jet'), vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        #plt.colorbar()
    plt.savefig(img_f+'/First_4_K_' + str(pp)+ '.png')
    plt.close()
    
    hss = Forward_Model.run(K_log,par=True)  # parallelization 
    hss = hss.reshape([obs.shape[0],nr])
    for ri in range(nr):
        RMSE_data[ri,0] = RMSE(obs,hss[:,ri].reshape(-1,1))
        RMSE_K[ri,0] = RMSE(s_true.reshape(-1,1),K_log[ri,:,:].reshape(-1,1))
    
    
    np.save(data_f+'/data_'+str(pp)+'.npy', hss)
    RMSE_mean = RMSE_data.mean(axis=0)
    yf = latent_z.squeeze().reshape(nr, zx * zy).numpy()
    yf = yf.transpose()
    ym = np.array(yf.mean(axis=1))    # Mean of y_f
    ym = ym.reshape(ym.shape[0],1)    
    dmf = yf-ym

    df = hss
    dm = np.array(df.mean(axis=1)) 
    dm = dm.reshape(dm.shape[0],1)   
    ddf = df-dm
    
    
    Cmd_f = (np.dot(dmf,ddf.T))/(nr-1);  # The cros-covariance matrix
    Cdd_f = (np.dot(ddf,ddf.T))/(nr-1);  # The auto covariance of predicted data
    
    R = np.linalg.cholesky(CD) # Matriz triangular inferior
    U = R.T   # Matriz R transpose
    p , w = np.linalg.eig(CD)
    
    d_obs = obs.reshape(obs.shape[0],1)
    aux = np.repeat(d_obs,nr,axis=1)
    mean = 0*(d_obs.T)
    noise = np.random.multivariate_normal(mean[0], np.eye(len(obs)), nr).T

    d_uc = aux+math.sqrt(alpha[pp])*np.dot(U,noise)  

     # Analysis step
    varn = 1-1/math.pow(10,2)

    u, s, vh = np.linalg.svd(Cdd_f+alpha[pp]*CD); v = vh.T
    diagonal = s
    for i in range(len(diagonal)):
        if (sum(diagonal[0:i+1]))/(sum(diagonal)) > varn:
            diagonal = diagonal[0:i+1]
            break
    
    u = u[:,0:i+1]
    v = v[:,0:i+1]
    ess = np.diag(diagonal**(-1))
    K = np.dot(Cmd_f,(np.dot(np.dot(v,ess),(u.T))))

    ya = yf + (np.dot(K,(d_uc-df)))
    ya = ya.transpose()
    latent_z = torch.from_numpy(ya.reshape(nr, 1, zx, zy)).float()
    plot_func(Q_locs,Mean_K,dm,RMSE_mean,i=pp)
    np.savetxt(data_f+'/Simulated_Head_Mean_'+str(pp)+'.txt', dm)
    np.savetxt(data_f+'/RMSE_Data_'+str(pp)+'.txt', RMSE_data)
    np.savetxt(data_f+'/RMSE_K_'+str(pp)+'.txt', RMSE_K)



''' Final results '''
k_array = netG(latent_z).squeeze().numpy()
np.save(data_f+'/k_array_'+str(itermax)+'.npy', k_array)
K_log = k_array*2-1
Mean_K = np.array(K_log.mean(axis=0))                    # the mean of K
Variance_K = np.array(np.var(K_log,axis = 0))               # the variance of K
np.savetxt(data_f+'/Mean_K_' + str(itermax) + '.txt', Mean_K)
np.savetxt(data_f+'/Variance_K_' + str(itermax) + '.txt', Variance_K)
 
for r in range(0,4):
    np.savetxt(data_f+'/First_4_K_Iter_' + str(itermax) + '_' + str(r+1)+'.txt', K_log[r,:,:])
    plt.subplot(2,2,r+1)
    plt.imshow(K_log[r,:,:],cmap=plt.get_cmap('jet'), vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
plt.savefig(img_f+'/First_4_K_' + str(itermax)+ '.png')
plt.close()
    
hss = np.zeros([h_n_ob*(h_n_ob-1),nr])
RMSE_data = np.zeros([nr,1])
RMSE_K = np.zeros([nr,1])

hss = Forward_Model.run(k_array,par=True)  # parallelization 
hss = hss.reshape([obs.shape[0],nr])

np.save(data_f+'/data_'+str(itermax)+'.npy', hss)
for ri in range(nr):
    RMSE_data[ri,0] = RMSE(obs,hss[:,ri].reshape(-1,1))
    RMSE_K[ri,0] = RMSE(s_true.reshape(-1,1),K_log[ri,:,:].reshape(-1,1))
    
RMSE_mean = RMSE_data.mean(axis=0)
df = hss
dm = np.array(df.mean(axis=1)) 
dm = dm.reshape(dm.shape[0],1)  

R = np.linalg.cholesky(CD) # Matriz triangular inferior
U = R.T   # Matriz R transpose
p , w = np.linalg.eig(CD)
    
d_obs = obs.reshape(obs.shape[0],1)
aux = np.repeat(d_obs,nr,axis=1)
mean = 0*(d_obs.T)
noise = np.random.multivariate_normal(mean[0], np.eye(len(obs)), nr).T

plot_func(Q_locs,Mean_K,dm,RMSE_mean,i=-1)
np.savetxt(data_f+'/Simulated_Head_Mean_'+str(itermax)+'.txt', dm)
np.savetxt(data_f+'/RMSE_Data_'+str(itermax)+'.txt', RMSE_data)
np.savetxt(data_f+'/RMSE_K_'+str(itermax)+'.txt', RMSE_K)

Time_end = time.time()
time_elapsed = Time_end - Time_start
print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 