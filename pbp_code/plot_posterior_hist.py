import numpy as np 
import os
import matplotlib.pyplot as plt

cur_path = os.path.dirname(os.path.abspath(__file__))
print(cur_path)
file_path=os.path.join(cur_path, 'posterior_params')

data_name='year'
seed=5
n_hidden=100
n_iterations=40
#num_datapoints=1439    #wine dataset
num_datapoints=463715    #year dataset
num_datapoint=7372    #robot dataset
n_bins=10

list_mean_hidden=[]
list_var_hidden=[]
list_mean_output=[]
list_var_output=[]

for i in range(num_datapoints):
    #print(seed)
    
    file_mean="f_n_mean_{}_seed={}_n_iter={}_n_hidden={}_datapoint={}_0.csv".format(data_name, seed, n_iterations, n_hidden, i)
    mean_hidden = np.loadtxt(os.path.join(file_path,file_mean), delimiter=',')
    mean_hidden=np.array(mean_hidden)
    mean_hidden2=np.linalg.norm(mean_hidden)
    #print(mean_hidden.shape)
    #mean_hidden2=np.mean(mean_hidden, axis=1)
    #print(mean_hidden2.shape)
    list_mean_hidden.append(mean_hidden2)

    file_var="f_n_var_{}_seed={}_n_iter={}_n_hidden={}_datapoint={}_0.csv".format(data_name, seed, n_iterations, n_hidden, i)
    var_hidden = np.loadtxt(os.path.join(file_path,file_var), delimiter=',')
    var_hidden=np.array(var_hidden)
    #var_hidden2=np.mean(var_hidden, axis=1)
    var_hidden2=np.linalg.norm(var_hidden)
    list_var_hidden.append(var_hidden2)

    file_mean="f_n_mean_{}_seed={}_n_iter={}_n_hidden={}_datapoint={}_1.csv".format(data_name, seed, n_iterations, n_hidden, i)
    mean_output = np.loadtxt(os.path.join(file_path,file_mean), delimiter=',')
    mean_output=np.array(mean_output)
    mean_output2 =np.linalg.norm(mean_output)
    list_mean_output.append(mean_output2)

    file_var="f_n_var_{}_seed={}_n_iter={}_n_hidden={}_datapoint={}_1.csv".format(data_name, seed, n_iterations, n_hidden, i)
    var_output = np.loadtxt(os.path.join(file_path,file_var), delimiter=',')
    var_output=np.array(var_output)
    var_output2 =np.linalg.norm(var_output)
    list_var_output.append( var_output2)


fig, ax = plt.subplots(2, 2, sharey=True, tight_layout=True)

ax[0, 0].hist(list_mean_hidden, bins=n_bins)
ax[0, 0].set_title('norm mean hidden_layer')
#ax[0, 0].set_xlim(np.amin(mean_hidden2), np.amax(mean_hidden2))
ax[0, 1].hist(list_var_hidden, bins=n_bins)
ax[0, 1].set_title('norm var hidden_layer')
#ax[0, 1].set_xlim(np.amin(var_hidden2), np.amax(var_hidden2))

ax[1, 0].hist(list_mean_output, bins=n_bins)
ax[1, 0].set_title('mean output_layer')
#ax[1, 0].set_xlim(np.amin(mean_output), np.amax(mean_output))
ax[1, 1].hist(list_var_output, bins=n_bins)
ax[1, 1].set_title('var output_layer')
#ax[1, 1].set_xlim(np.amin(var_output), np.amax(var_output))


plt.show()





