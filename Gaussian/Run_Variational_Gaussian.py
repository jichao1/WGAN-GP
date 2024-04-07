import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from WGAN_Net import netG
import matlab.engine

eng = matlab.engine.start_matlab()

outf = './Results_Gaussian_Variational'
img_f = outf + '/image'      # output image folder
data_f = outf + '/data'    # output data folder

try:
    os.makedirs(img_f)
    os.makedirs(data_f)
except OSError:
    pass

n_worker = 96    # number of workers for parallel processing

itermax = 20  # iterations
alpha = 2    # step length
n_ls = 50   # number of realizations for line search

# prior covariance
# Sigma = 1.0**2
Sigma = 0.1**2
invSigma = 1./Sigma 

precision = 1e-5

zx, zy = 6,6   # latent dimension
m = zx * zy 

z_init = np.random.randn(m,1)

wgan_pth = 'Weights_Gaussian.pth'

''' True field ''' 
s_true = np.loadtxt('Ref_Gaussian.txt')

nrow = 96; ncol = 96; nlay = 1
# Location of observations
obs_locmat = np.zeros((nlay, nrow, ncol), np.bool_)

h_n_ob = 16      # number of wells
h_spacing = 24         # spacing between two wells
h_Start = 11
h_End = 83  


for i in range(h_Start,h_End+h_spacing,h_spacing): 
    for j in range(h_Start,h_End+h_spacing,h_spacing): 
        obs_locmat[0, i, j] = 1

Q_locs_idx = np.where(obs_locmat == True)
Q_locs = []
for Q_loc in zip(Q_locs_idx[0], Q_locs_idx[1], Q_locs_idx[2]):
    Q_locs.append(Q_loc)

''' Prepare WGAN generator '''
device = 'cpu'
netG = netG()
netG.load_state_dict(torch.load(wgan_pth,map_location=device))
netG.to(device)
netG.eval()
torch.set_grad_enabled(False)


''' Observed data '''
out_true = eng.CO2_Simul(s_true,nargout=1)
# assume known error matrix R.  
saturation_true = out_true
saturation_true = np.array(saturation_true)

nobs = len(saturation_true)

std_saturation = saturation_true.mean() * 0.01   # 1% std error

err_saturation = std_saturation*np.random.randn(nobs,1)

obs_true = saturation_true
err = err_saturation
obs = obs_true + err

np.savetxt(data_f+'/Observed_data.txt', obs)

invR = np.eye(nobs)
for p in range(nobs):
    invR[p,p] = 1/(std_saturation**2)
    
def Forward(z,num):
  z_tensor = z.reshape([num,1,zx,zy])
  z_tensor = torch.tensor(z_tensor).float().to(device)
  k_array = netG(z_tensor).cpu().squeeze().numpy()
  K_log = matlab.double(k_array.tolist())
  if num == 1:
      out_simul = eng.CO2_Simul(K_log,nargout=1)   
      saturation_simul = out_simul
      saturation_simul = np.array(saturation_simul)
      simul = saturation_simul
  elif num > 1:
      simul = eng.Parallel_CO2Simul(K_log,n_worker,num,nargout=1)
      simul = np.array(simul)
  return K_log, simul
    
def RMSE(obs,simul):
  nobs = obs.shape[0]
  res = obs - simul
  return np.sqrt(np.sum(res**2)/nobs)

''' define plot function '''
def plot_func(Q_locs,K_cur,simul,i=0):
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

  im = ax[1].pcolormesh(K_cur, cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
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
  ax[2].plot(obs,simul,'.', markersize=10, color='red')
  ax[2].plot([Min,Max],[Min,Max],'-', color='black')
  ax[2].set_aspect('equal', adjustable='box')
  ax[2].set_xlabel('Observed Saturation')
  ax[2].set_ylabel('Simulated Saturation')
  ax[2].legend(markerscale=0,handlelength=0,handletextpad=0,
                labels=['RMSE: '+ '%.3f' % RMSE(obs,simul)],
                loc='upper left',fontsize=12,edgecolor='k')
  ax[2].set_title('Fitting Error',fontsize=18)
  
  plt.tight_layout()
  #plt.show()
  fig.savefig(img_f+'/result_iter_'+str(i)+'.png',dpi=150, bbox_inches='tight')
  plt.close()


''' Inversion '''
nobs = obs.shape[0] 
z_cur = np.copy(z_init)

Time_start = time.time()
for i in range(itermax):
  print('#################### Iteration %d #####################' % (i))
  K_log, simul_cur = Forward(z_cur,num=1)
    
  np.savetxt(data_f+'/Log_K_iter_'+str(i)+'.txt', K_log)
  np.savetxt(data_f+'/Simulated_Head_iter_'+str(i)+'.txt', simul_cur)
  
  rmse = RMSE(obs,simul_cur).squeeze()
  print('RMSE: %.3f'%(rmse))
    
  plot_func(Q_locs,K_log,simul_cur,i=i)
  
  J_cur = np.zeros([nobs,m])
  delta = np.zeros([m,1])
  z_delta = np.zeros([m,m])
  for j in range(m):
    zerovec = np.zeros((m,1))
    zerovec[j] = 1.

    mag = np.dot(z_cur.T,zerovec)
    absmag = np.dot(abs(z_cur.T),abs(zerovec))
    if mag >= 0:
      signmag = 1.
    else:
      signmag = -1.
      
    delta[j] = signmag*np.sqrt(precision)*(max(abs(mag),absmag))/((np.linalg.norm(zerovec)+ np.finfo(float).eps)**2)
    if delta[j] == 0:
      delta[j] = np.sqrt(precision)
      
    z_delta[:,j] = (z_cur+zerovec*delta[j]).squeeze()
  
  K_log_delta, simul_delta = Forward(z_delta.transpose(),num=m)
  simul_delta = simul_delta.reshape([nobs,m])
  
  for k in range(m):
    temp = (simul_delta[:,k].reshape([-1,1])-simul_cur)/delta[k]
    J_cur[:,k] = temp.squeeze()

  solve_a = np.dot( np.dot(J_cur.T,invR),J_cur) + invSigma*np.eye(m)
  solve_b = -np.dot(invSigma,z_cur) + np.dot( np.dot(J_cur.T,invR), (obs - simul_cur) )
  dz = np.linalg.solve(solve_a,solve_b)

  z_ls = np.zeros([m,n_ls])
  for j in range(n_ls):
    z_ls[:,j] = (z_cur + float(j+1)*alpha/float(n_ls)*dz).squeeze()

  K_log_ls, simul_ls = Forward(z_ls.transpose(),num=n_ls)
  simul_ls = simul_ls.reshape([nobs,n_ls])
  
  obj_old = 1.E+10
  
  obj_best = 0.
  j_best = 0.

  for j in range(n_ls):
    res = obs - simul_ls[:,j].reshape([-1,1])
    obj = (np.dot(res.T,np.dot(invR,res)) + invSigma*np.dot(z_ls[:,j].T,z_ls[:,j])).reshape(-1)
    if obj < obj_old:
        j_best = j
        obj_best = obj
        obj_old = obj

  z_cur = np.copy(z_cur + float(j_best+1)*alpha/float(n_ls)*dz)
  

''' Final results '''
K_log, simul_cur = Forward(z_cur,num=1)
  
np.savetxt(data_f+'/Log_K_iter_'+str(itermax)+'.txt', K_log)
np.savetxt(data_f+'/Simulated_Head_iter_'+str(itermax)+'.txt', simul_cur)
  
rmse = RMSE(obs,simul_cur).squeeze()
print('RMSE: %.3f'%(rmse))

plot_func(Q_locs,K_log,simul_cur,i=-1)
  
Time_end = time.time()
time_elapsed = Time_end - Time_start
print('Time elapsed {:.0f}hr {:.0f}m {:.0f}s'.format( 
    (time_elapsed // 60) // 60,
    (time_elapsed // 60) % 60, 
    time_elapsed % 60) 
    ) 