function [data_all] = Parallel_CO2Simul(perm,NumWorkers,Ne)

myCluster = parcluster('local'); 
myCluster.NumWorkers = NumWorkers;                     
saveProfile(myCluster); 
parpool('local',NumWorkers); 

row = size(perm,2);
col = size(perm,3);
data_all=[];
parfor i=1:Ne    
    data_perm = perm(i,:,:);
    data_perm = reshape(data_perm,row,col)
    saturation_data = CO2_Simul(data_perm);
    data_all = [data_all,saturation_data];
end
poolobj = gcp('nocreate');
delete(poolobj);                                 % shut down parapool
end
