import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from WGAN_Net import netG
import matlab.engine
import time

eng = matlab.engine.start_matlab()

outf = './Results_Fractured_ESMDA'
img_f = outf + '/image'      # output image folder
data_f = outf + '/data'    # output data folder

wgan_pth = 'Weights_Fracture.pth'

try:
    os.makedirs(img_f)
    os.makedirs(data_f)
except OSError:
    pass

n_worker = 96    # number of workers for parallel processing

nrow = 96; ncol = 96; nlay = 1

''' Location of observations '''
obs_locmat = np.zeros((nlay, nrow, ncol), np.bool_)

h_n_ob = 16      # number of wells
h_spacing = 24         # spacing between two wells
h_Start = 11
h_End = 83  

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
s_true = np.loadtxt('Ref_Fracture.txt')


''' Prepare WGAN generator '''
device = 'cpu'
netG = netG()
netG.load_state_dict(torch.load(wgan_pth,map_location=device))
netG.to(device)
netG.eval()
torch.set_grad_enabled(False)


''' Observed data '''
saturation_true = eng.CO2_Simul(s_true,nargout=1)
# assume known error matrix R.  
saturation_true = np.array(saturation_true)

np.savetxt(data_f+'/Saturation_true.txt', saturation_true)
nobs = len(saturation_true)

std_saturation = saturation_true.mean() * 0.01   # 1% std error

err_saturation = std_saturation*np.random.randn(nobs,1)

CD = np.eye(nobs)   # observation error matrix
CD_inv = np.eye(nobs) # inverse

for p in range(nobs):
    CD[p,p] = std_saturation**2
    CD_inv[p,p] = 1/(std_saturation**2)
    
obs_true = saturation_true
err = err_saturation
obs = obs_true + err

np.savetxt(data_f+'/Observed_data.txt', obs)


''' define plot function '''
def RMSE(obs,simul_obs):
  nobs = obs.shape[0]
  res = obs - simul_obs
  return np.sqrt(np.sum(res**2)/nobs)

def plot_func(Q_locs,K_mean,sim_mean,rmse,i=0):
  fig, ax = plt.subplots(1,3,figsize=(15,10))
  im = ax[0].pcolormesh(s_true, cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
  for Q_loc in Q_locs:                                       # the location of wells
      ax[0].scatter(Q_loc[1],Q_loc[2], marker='o',c='None', edgecolors='yellow', s = 50,linewidth=2)
  ax[0].set_aspect('equal', adjustable='box')
  ax[0].xaxis.set_visible(False)
  ax[0].yaxis.set_visible(False)
  ax[0].invert_yaxis()
  ax[0].set_title('True logK',fontsize=18)
  #fig.colorbar(im, ax=ax[0])

  im = ax[1].pcolormesh(K_mean, cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
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

  Min = 0
  Max = 0.7
  ax[2].plot(obs,sim_mean,'.', markersize=10, color='red')
  ax[2].plot([Min,Max],[Min,Max],'-', color='black')
  ax[2].set_aspect('equal', adjustable='box')
  ax[2].set_xlabel('Observed Saturation')
  ax[2].set_ylabel('Simulated Saturation')
  ax[2].legend(markerscale=0,handlelength=0,handletextpad=0,
                labels=['RMSE: '+ '%.3f' % (rmse)],
                loc='upper left',fontsize=12,edgecolor='k')
  ax[2].set_title('Fitting Error',fontsize=18)
  
  plt.tight_layout()
  #plt.show()
  fig.savefig(img_f+'/result_iter_'+str(i)+'.png',dpi=150, bbox_inches='tight')
  plt.close()


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
    K_log = k_array
    Mean_K = np.array(K_log.mean(axis=0))                    # the mean of K
    Variance_K = np.array(np.var(K_log,axis = 0))               # the variance of K
    np.savetxt(data_f+'/Mean_K_' + str(pp) + '.txt', Mean_K)
    np.savetxt(data_f+'/Variance_K_' + str(pp) + '.txt', Variance_K)
    
    RMSE_data = np.zeros([nr,1])
    RMSE_K = np.zeros([nr,1])
    
    for r in range(0,4):
        np.savetxt(data_f+'/First_4_K_Iter_' + str(pp) + '_' + str(r+1)+'.txt', K_log[r,:,:])
        plt.subplot(2,2,r+1)
        plt.imshow(K_log[r,:,:],cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        #plt.colorbar()
    plt.savefig(img_f+'/First_4_K_' + str(pp)+ '.png')
    plt.close()
    
    K_mat = matlab.double(K_log.tolist())
    hss = eng.Parallel_CO2Simul(K_mat,n_worker,nr,nargout=1)
    hss = np.array(hss)
    
    for ri in range(nr):
        RMSE_data[ri,0] = RMSE(obs,hss[:,ri].reshape(-1,1))
        RMSE_K[ri,0] = RMSE(s_true.reshape(-1,1),K_log[ri,:,:].reshape(-1,1))
    
    
    np.save(data_f+'/data_'+str(pp)+'.npy', hss)
    RMSE_mean = RMSE_data.mean(axis=0).squeeze()
    print('RMSE: %.3f'%(RMSE_mean))
    
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
    
    np.save(data_f+'/Save_z.npy', ya)
    latent_z = torch.from_numpy(ya.reshape(nr, 1, zx, zy)).float().to(device)
    plot_func(Q_locs,Mean_K,dm,RMSE_mean,i=pp)
    np.savetxt(data_f+'/Simulated_Head_Mean_'+str(pp)+'.txt', dm)
    np.savetxt(data_f+'/RMSE_Data_'+str(pp)+'.txt', RMSE_data)
    np.savetxt(data_f+'/RMSE_K_'+str(pp)+'.txt', RMSE_K)



''' Final results '''
k_array = netG(latent_z).squeeze().numpy()
np.save(data_f+'/k_array_'+str(itermax)+'.npy', k_array)
K_log = k_array
Mean_K = np.array(K_log.mean(axis=0))                    # the mean of K
Variance_K = np.array(np.var(K_log,axis = 0))               # the variance of K
np.savetxt(data_f+'/Mean_K_' + str(itermax) + '.txt', Mean_K)
np.savetxt(data_f+'/Variance_K_' + str(itermax) + '.txt', Variance_K)
 
for r in range(0,4):
    np.savetxt(data_f+'/First_4_K_Iter_' + str(itermax) + '_' + str(r+1)+'.txt', K_log[r,:,:])
    plt.subplot(2,2,r+1)
    plt.imshow(K_log[r,:,:],cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
plt.savefig(img_f+'/First_4_K_' + str(itermax)+ '.png')
plt.close()
    
RMSE_data = np.zeros([nr,1])
RMSE_K = np.zeros([nr,1])

K_mat = matlab.double(K_log.tolist())
hss = eng.Parallel_CO2Simul(K_mat,n_worker,nr,nargout=1)
hss = np.array(hss)
      
np.save(data_f+'/data_'+str(itermax)+'.npy', hss)
for ri in range(nr):
    RMSE_data[ri,0] = RMSE(obs,hss[:,ri].reshape(-1,1))
    RMSE_K[ri,0] = RMSE(s_true.reshape(-1,1),K_log[ri,:,:].reshape(-1,1))
    
RMSE_mean = RMSE_data.mean(axis=0).squeeze()
print('RMSE: %.3f'%(RMSE_mean))

df = hss
dm = np.array(df.mean(axis=1)) 
dm = dm.reshape(dm.shape[0],1)  

plot_func(Q_locs,Mean_K,dm,RMSE_mean,i=-1)
np.savetxt(data_f+'/Simulated_Head_Mean_'+str(itermax)+'.txt', dm)
np.savetxt(data_f+'/RMSE_Data_'+str(itermax)+'.txt', RMSE_data)
np.savetxt(data_f+'/RMSE_K_'+str(itermax)+'.txt', RMSE_K)

Time_end = time.time()
time_elapsed = Time_end - Time_start
print('Time elapsed {:.0f}hr {:.0f}m {:.0f}s'.format( 
    (time_elapsed // 60) // 60,
    (time_elapsed // 60) % 60, 
    time_elapsed % 60) 
    ) 