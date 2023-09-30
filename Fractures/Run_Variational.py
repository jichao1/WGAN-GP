import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from Materials.WGAN_Net import netG
from Materials import Forward_HT
import scipy.optimize as opt
import time

outf = './Results_Variaional'
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
    

''' Pumping/monitoring well locations '''
N = np.array([ncol, nrow, nlay])
dx = np.array([Lx/ncol, Ly/nrow, Lz/nlay])

x = np.linspace(0. + dx[0] / 2., Lx - dx[0] / 2., N[0])
y = np.linspace(0. + dx[1] / 2., Ly - dx[1] / 2., N[1])

fig, ax = plt.subplots() 
idx = np.where(obs_locmat == 1)
for i,j in zip(idx[1],idx[2]):
    plt.plot(x[j],y[i],'ko')
plt.xlim(0,Lx)
plt.ylim(0,Ly)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
ax.set_aspect('equal', adjustable='box')
ax.invert_yaxis()
plt.title('well locations')


''' True field ''' 
s_true = np.loadtxt('./Materials/Ref_Fractures.txt')
s_true = s_true*2-1
x = np.linspace(0. + dx[0] / 2., Lx - dx[0] / 2., N[0])
y = np.linspace(0. + dx[1] / 2., Ly - dx[1] / 2., N[1])
XX, YY = np.meshgrid(x, y)

fig, ax = plt.subplots() 
plt.pcolormesh(XX,YY,s_true, cmap=plt.get_cmap('jet'))
ax.set_aspect('equal', adjustable='box')
ax.invert_yaxis()
plt.title('True logK')


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
# assume known error matrix R.  
std_err = 0.02
R = std_err**2
invR = 1./R
obs = obs_true + std_err*np.random.randn(len(obs_true),1)
np.savetxt(data_f+'/Observed_Head.txt', obs)

def Forward(z):
  z = z.reshape([1,1,zx,zy])
  z_tensor = torch.from_numpy(z).float()
  s_relz = netG(z_tensor)
  K_log = s_relz*2-1
  return Forward_Model.run_model(K_log[0,0,:,:].numpy())

def RMSE(z):
  res = obs - Forward(z)
  nobs = obs.shape[0]
  return np.sqrt(np.sum(res**2)/nobs)

def Obj_func(z):
  Sigma = 1.0**2
  res = obs - Forward(z)
  zvec = z.reshape(-1,1)
  return (np.dot(res.T,np.dot(invR,res)) + Sigma*np.dot(zvec.T,zvec)).reshape(-1)

def Obj_linesearch(alpha,*args):
  x0 = args[0]
  direction = args[1]
  #return Obj_func((1-alpha)*x0+alpha*direction) # for kalman 
  return Obj_func(x0+alpha*direction)

''' define plot function '''
def plot_func(z,Q_locs,i=0):
  fig, ax = plt.subplots(1,3,figsize=(15,10))
  im = ax[0].pcolormesh(s_true, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1)
  for Q_loc in Q_locs:                                       # the location of wells
      ax[0].scatter(Q_loc[1],Q_loc[2], marker='o',c='None', edgecolors='yellow', s = 20)
  ax[0].set_aspect('equal', adjustable='box')
  ax[0].xaxis.set_visible(False)
  ax[0].yaxis.set_visible(False)
  ax[0].invert_yaxis()
  ax[0].set_title('True log10K')
  #fig.colorbar(im, ax=ax[0])

  z_tensor = torch.from_numpy( z.reshape([1,1,zx,zy]) ).float()
  s_cur = netG(z_tensor)
  K_cur = s_cur*2-1
  np.savetxt(data_f+'/Log_K_iter_'+str(i)+'.txt', K_cur.squeeze().numpy())
  im = ax[1].pcolormesh(K_cur[0,0,:,:], cmap=plt.get_cmap('jet'), vmin=-1, vmax=1)
  ax[1].set_aspect('equal', adjustable='box')
  #fig.colorbar(im, ax=ax[0,1])
  ax[1].xaxis.set_visible(False)
  ax[1].yaxis.set_visible(False)
  ax[1].invert_yaxis()
  if i == 0:
    ax[1].set_title('Initial logK')
  elif i < 0:
    ax[1].set_title('Estimated logK')
  else:
    ax[1].set_title('Estimated logK at iteration %d' % (i))

  simul_obs = Forward(z)
  np.savetxt(data_f+'/Simulated_Head_iter_'+str(i)+'.txt', simul_obs)
  ax[2].plot(obs,simul_obs,'.', markersize=10, color='red')
  ax[2].plot(np.array( [np.min(simul_obs),np.max(simul_obs)]), np.array([np.min(simul_obs),np.max(simul_obs)] ),'-', color='black')
  ax[2].set_aspect('equal', adjustable='box')
  ax[2].set_xticks([])
  ax[2].set_yticks([])
  #fig.colorbar(im, ax=ax[2])
  #ax[2].xaxis.set_visible(False)
  #ax[2].yaxis.set_visible(False)
  ax[2].set_xlabel('obs')
  ax[2].set_ylabel('simul')
  ax[2].set_title('Observation Fitting (RMSE: %5.3e)' % RMSE(z))
  
  plt.tight_layout()
  plt.show()
  fig.savefig(img_f+'/result_iter_'+str(i)+'.png',dpi=150, bbox_inches='tight')


itermax = 20
# prior covariance
Sigma = 1.0**2
invSigma = 1./Sigma 
nobs = obs.shape[0] # read from obs
#precision = np.finfo(np.float64).eps # for optimal FD delta

precision = 1e-5

''' Data Assimilation '''
zx, zy = 3,3
m = zx * zy 
z_init = np.random.randn(1, m)

z_cur = np.copy(z_init.reshape([-1,1]))
plot_func(z_init,Q_locs=Q_locs)
print('objective and RMSE at z_init: %e, %e' % (Obj_func(z_init),RMSE(z_init)))

Time_start = time.time()
for i in range(itermax):
  print('#################### Iteration %d #####################' % (i+1))
  J_cur = np.zeros((nobs,m))
  for j in range(m):
    zerovec = np.zeros((m,1))
    zerovec[j] = 1.

    # following PCGA delta calculation, also see Brown and Saad, 1990
    mag = np.dot(z_cur.T,zerovec)
    absmag = np.dot(abs(z_cur.T),abs(zerovec))
    if mag >= 0:
      signmag = 1.
    else:
      signmag = -1.
    
    delta = signmag*np.sqrt(precision)*(max(abs(mag),absmag))/((np.linalg.norm(zerovec)+ np.finfo(float).eps)**2)

    if delta == 0:
      #print("%d-th delta: signmag %g, precision %g, max abs %g, norm %g" % (j,signmag, precision,(max(abs(mag),absmag)), (np.linalg.norm(z_cur)**2)))
      delta = np.sqrt(precision)
      #print("%d-th delta: assigned as sqrt(precision) - %g", delta)

    J_cur[:,j:j+1] = ( Forward(z_cur+zerovec*delta) - Forward(z_cur) )/delta

  
  Jz_cur = np.dot(J_cur,z_cur)

  # using Kalman (nobs x nobs system)
  #Cauto = Sigma*np.dot(J_cur,J_cur.T) + R*np.eye(nobs) # auto covariance - sparse matrix.. 
  #dz = Sigma*np.dot(J_cur.T, np.linalg.solve(Cauto, obs - forward(z_cur) + Jz_cur))  
  
  # Use dim(z) by dim(z) dimension formulation
  dz = np.linalg.solve((invR*np.dot(J_cur.T,J_cur) + invSigma*np.eye(m)), -np.dot(invSigma,z_cur) + invR*np.dot(J_cur.T,obs - Forward( z_cur )))
  
  res = opt.minimize_scalar(fun=Obj_linesearch,args=(z_cur,dz))
  alpha = res.x
  print('the step length alpha:',alpha)
  #print(np.linalg.norm(dz))
  # update (need linear search)

  # using Kalman (nobs x nobs system)
  #z_cur = np.copy((1-alpha)*z_cur + alpha*dz)
  
  # Use dim(z) by dim(z) dimension formulation
  z_cur = np.copy(z_cur + alpha*dz)
  plot_func( z_cur , Q_locs=Q_locs, i=i+1)
  
  print('objective and RMSE at iter %d: %e, %e' % (i+1, Obj_func(z_cur), RMSE(z_cur)))
  # if np.abs(alpha) < 1e-5:
  #   break


''' Final results '''
plot_func(z_cur, Q_locs=Q_locs, i=-1)
Time_end = time.time()
time_elapsed = Time_end - Time_start
print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 