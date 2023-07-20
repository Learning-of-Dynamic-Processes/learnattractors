# In[Preamble]:
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
#import ESN
from matplotlib import cm
import matplotlib.ticker as ticker
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Reshape,InputLayer,Dense,Flatten, Conv2D,Conv1D, Dropout, Input,ZeroPadding2D,ZeroPadding1D,MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# config = tf.config
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
K.set_floatx('float64')  

def lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

def sig_scaled(a,b,c,d):
    def sig_tmp(x):
        return a / (1 + K.exp(-b*(x-c)))+d
    return sig_tmp

# In[Lorenz trajectory]:
    
# Lorenz paramters and initial point
sigma, beta, rho = 10, 8/3, 28
u0, v0, w0 = 0, 1, 1.05

# Maximum time point and total number of time points
tmax, n = 12000, 1200001

plotlength = 10000
burnin = 100001
outofsample=1100001
t = np.linspace(0, tmax, n)
ls = np.linspace(0,1,plotlength)
cmap = cm.viridis(ls)

# Integrate
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
LS_x, LS_y, LS_z = f.T

# Plot the Lorenz trajectory using a Matplotlib 3D projection
fig = plt.figure(figsize = (10,10))
fig.suptitle("3D-scatter plot")
ax = fig.gca(projection='3d')
ax.scatter(LS_x[burnin:plotlength+burnin], LS_y[burnin:plotlength+burnin], LS_z[burnin:plotlength+burnin], color = cmap, s = 5)
plt.show()

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Components of trajectory")
ax1.scatter(t[burnin:plotlength+burnin],LS_x[burnin:plotlength+burnin], color = cmap, s = 5)
ax2.scatter(t[burnin:plotlength+burnin],LS_y[burnin:plotlength+burnin], color = cmap, s = 5)
ax3.scatter(t[burnin:plotlength+burnin],LS_z[burnin:plotlength+burnin], color = cmap, s = 5)
plt.show()

burnin = 115001
# Plot the Lorenz attractor using a Matplotlib 3D projection
fig = plt.figure(figsize = (10,10))
fig.suptitle("3D-scatter of a Lorenzcurve")
ax = fig.gca(projection='3d')
ax.scatter(LS_x[burnin:plotlength+burnin], LS_y[burnin:plotlength+burnin], LS_z[burnin:plotlength+burnin], color = cmap, s = 5)
plt.show()

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Lorenzcurve per Axis")
ax1.scatter(t[burnin:plotlength+burnin],LS_x[burnin:plotlength+burnin], color = cmap, s = 5)
ax2.scatter(t[burnin:plotlength+burnin],LS_y[burnin:plotlength+burnin], color = cmap, s = 5)
ax3.scatter(t[burnin:plotlength+burnin],LS_z[burnin:plotlength+burnin], color = cmap, s = 5)
plt.show()

# In[Noisy observations]:

burnin = 100001    
np.random.seed(42)
noise_x = np.random.normal(0, 0.5, np.shape(LS_x))
noise_y = np.random.normal(0, 0.5, np.shape(LS_y))
noise_z = np.random.normal(0, 0.5, np.shape(LS_z))
LS_noisy_x = LS_x + noise_x
LS_noisy_y = LS_y + noise_y
LS_noisy_z = LS_z + noise_z

data_target = [LS_x[burnin:outofsample],LS_y[burnin:outofsample],LS_z[burnin:outofsample]]
 
# Plot the Lorenz trajectory using a Matplotlib 3D projection
burnin = 115001
plotlength = 2000
ls = np.linspace(0,1,plotlength)
cmap = cm.viridis(ls)
fig = plt.figure(figsize = (10,10))
fig.suptitle("Lorenz trajectory")
ax = fig.gca(projection='3d')
ax.scatter(LS_noisy_x[burnin:plotlength+burnin], LS_noisy_y[burnin:plotlength+burnin], LS_noisy_z[burnin:plotlength+burnin], color = cmap, s = 5)
plt.show()

fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
fig.suptitle("Components of Lorenz trajectory")
ax1.scatter(t[burnin:plotlength+burnin],LS_noisy_x[burnin:plotlength+burnin], color = cmap, s = 5)
ax2.scatter(t[burnin:plotlength+burnin],LS_noisy_y[burnin:plotlength+burnin], color = cmap, s = 5)
ax3.scatter(t[burnin:plotlength+burnin],LS_noisy_z[burnin:plotlength+burnin], color = cmap, s = 5)
plt.show()

# In[Network]:

def buildNN(min_,max_):
    NN = Sequential()
    NN.add(InputLayer(input_shape=d))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = 'sigmoid',use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    NN.add(Dense(d,activation = "sigmoid",use_bias=True))
    # NN.add(Dense(1,activation = 'linear',use_bias=True))
    NN.add(Dense(1,activation = sig_scaled(max_-min_,1,0,min_),use_bias=True))
    # NN.add(Dense(1,activation ='sigmoid',use_bias=True))
    return NN

# network for dimension d=7
# def buildNN(min_,max_):
#     NN = Sequential()
#     NN.add(InputLayer(input_shape=d))
#     NN.add(Dense(10*d,activation = "sigmoid",use_bias=True))
#     NN.add(Dense(10*d,activation = "sigmoid",use_bias=True))
#     NN.add(Dense(10*d,activation = "sigmoid",use_bias=True))
#     NN.add(Dense(10*d,activation = "sigmoid",use_bias=True))
#     NN.add(Dense(10*d,activation = "sigmoid",use_bias=True))
#     # NN.add(Dense(1,activation = 'linear',use_bias=True))
#     NN.add(Dense(1,activation = sig_scaled(max_-min_,1,0,min_),use_bias=True))
#     return NN

# In[Reservoir with orthonormal matrix]:

d = 20

np.random.seed(40)
#Create reservoir A#
from scipy.stats import ortho_group  # Requires version 0.18 of scipy

Aorth = ortho_group.rvs(dim=d)
param = 0.9
Aorth /= np.real(np.max(np.linalg.eigvals(Aorth)))/param

#Create Reservoir C
Crc = np.random.uniform(low = -1.0, high = 1.0, size = d)
Crc /= np.linalg.norm(Crc)

burnin = 100001
outofsample=1100001
# n=1200001
#create trajectory
x_orth = np.zeros((d,n))  
for k in range(1,n):
    x_orth[:,k] = Aorth@x_orth[:,k-1] + Crc*LS_noisy_x[k]

data_input_rc_states  = x_orth[:,burnin-1:outofsample-1].T   

# In[Takens]:

d = 20
#Create Takens A
A= np.eye(d,k=-1)

#Create Takens C
C = np.zeros(d)
C[0] = 1

#create trajectory
x = np.zeros((d,n))  
error = np.zeros((d,n))
for k in range(1,n):
    x[:,k] = A@x[:,k-1] + C*LS_noisy_x[k]
 
data_input_tak_states  = x[:,burnin-1:outofsample-1].T

# In[RC training 1]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 1000 ,restore_best_weights=True)
NN1orth = buildNN(-20,20)
# NN1orth.load_weights("orth_noisy_filt1.h5")
NN1orth.compile(loss = "MSE", optimizer =Adam(learning_rate =5e-3),metrics=["MAPE"])
history1orth = NN1orth.fit(data_input_rc_states,data_target[0], batch_size=10000, epochs =7000, verbose = 1, shuffle=1,callbacks=[es])
NN1orth.save_weights("orth_noisy_filt1.h5")
np.save('historthfilt1.npy',history1orth.history)
np.min(data_target[0])

# In[Takens training 1]:

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500 ,restore_best_weights=True)
NN1tak = buildNN(-20,20)
NN1tak.compile(loss="MSE", optimizer=Adam(learning_rate=5e-3), metrics=["MAPE"])
history1tak = NN1tak.fit(data_input_tak_states, data_target[0], batch_size=10000, epochs=7000, verbose=1, shuffle=1, callbacks=[es])
NN1tak.save_weights("tak_noisy_filt1.h5")
np.save('histtaknoisefilt1.npy',history1tak.history)

# In[RC training 2]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1orth = buildNN(-20,20)
NN1orth.load_weights("orth_noisy_filt1.h5")
NN1orth.compile(loss = "MSE", optimizer =Adam(learning_rate =3e-3),metrics=["MAPE"])
history1orth = NN1orth.fit(data_input_rc_states,data_target[0], batch_size=10000, epochs =7000, verbose = 0, shuffle=1,callbacks=[es])
NN1orth.save_weights("orth_noisy_filt2.h5")
np.save('historthnoisefilt2.npy',history1orth.history)

# In[Takens training 2]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500 ,restore_best_weights=True)
NN1tak = buildNN(-20,20)
NN1tak.load_weights("tak_noisy_filt1.h5")
NN1tak.compile(loss="MSE", optimizer=Adam(learning_rate=3e-3), metrics=["MAPE"])
history1tak = NN1tak.fit(data_input_tak_states, data_target[0], batch_size=10000, epochs=7000, verbose=0, shuffle=1, callbacks=[es])
NN1tak.save_weights("tak_noisy_filt2.h5")
np.save('histtaknoisefilt2.npy',history1tak.history)

# In[RC training 3]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1orth = buildNN(-20,20)
NN1orth.load_weights("orth_noisy_filt2.h5")
NN1orth.compile(loss = "MSE", optimizer =Adam(learning_rate =1e-3),metrics=["MAPE"])
history1orth = NN1orth.fit(data_input_rc_states,data_target[0], batch_size=10000, epochs =7000, verbose = 1, shuffle=1,callbacks=[es])
NN1orth.save_weights("orth_noisy_filt3.h5")
np.save('historthnoise3.npy',history1orth.history)

# In[Takens training 3]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500 ,restore_best_weights=True)
NN1tak = buildNN(-20,20)
NN1tak.load_weights("tak_noisy_filt2.h5")
NN1tak.compile(loss="MSE", optimizer=Adam(learning_rate=1e-3), metrics=["MAPE"])
history1tak = NN1tak.fit(data_input_tak_states, data_target[0], batch_size=10000, epochs=7000, verbose=1, shuffle=1, callbacks=[es])
NN1tak.save_weights("tak_noisy_filt3.h5")
np.save('histtaknoisefilt3.npy',history1tak.history)

# In[RC training 4]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1orth = buildNN(-20,20)
NN1orth.load_weights("orth_noisy_filt3.h5")
NN1orth.compile(loss = "MSE", optimizer =Adam(learning_rate =9e-4),metrics=["MAPE"])
history1orth = NN1orth.fit(data_input_rc_states,data_target[0], batch_size=10000, epochs =7000, verbose = 1, shuffle=1,callbacks=[es])
NN1orth.save_weights("orth_noisy_filt4.h5")
np.save('historthnoisefilt4.npy',history1orth.history)

# In[Takens training 4]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500 ,restore_best_weights=True)
NN1tak = buildNN(-20,20)
NN1tak.load_weights("tak_noisy_filt3.h5")
NN1tak.compile(loss="MSE", optimizer=Adam(learning_rate=9e-4), metrics=["MAPE"])
history1tak = NN1tak.fit(data_input_tak_states, data_target[0], batch_size=10000, epochs=7000, verbose=0, shuffle=1, callbacks=[es])
NN1tak.save_weights("tak_noisy_filt4.h5")
np.save('histtaknoisefilt4.npy',history1tak.history)

# In[RC training 5]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1orth = buildNN(-20,20)
NN1orth.load_weights("orth_noisy_filt4.h5")
NN1orth.compile(loss = "MSE", optimizer =Adam(learning_rate =7e-4),metrics=["MAPE"])
history1orth = NN1orth.fit(data_input_rc_states,data_target[0], batch_size=10000, epochs =7000, verbose = 0, shuffle=1,callbacks=[es])
NN1orth.save_weights("orth_noisy_filt5.h5")
np.save('historthnoisefilt5.npy',history1orth.history)

# In[Takens training 5]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500 ,restore_best_weights=True)
NN1tak = buildNN(-20,20)
NN1tak.load_weights("tak_noisy_filt4.h5")
NN1tak.compile(loss="MSE", optimizer=Adam(learning_rate=7e-4), metrics=["MAPE"])
history1tak = NN1tak.fit(data_input_tak_states, data_target[0], batch_size=10000, epochs=7000, verbose=0, shuffle=1, callbacks=[es])
NN1tak.save_weights("tak_noisy_filt5.h5")
np.save('histtaknoisefilt5.npy',history1tak.history)

# In[RC training 6]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1orth = buildNN(-20,20)
NN1orth.load_weights("orth_noisy_filt5.h5")
NN1orth.compile(loss = "MSE", optimizer =Adam(learning_rate =5e-4),metrics=["MAPE"])
history1orth = NN1orth.fit(data_input_rc_states,data_target[0], batch_size=10000, epochs =7000, verbose = 1, shuffle=1,callbacks=[es])
NN1orth.save_weights("orth_noisy_filt6.h5")
np.save('historthnoisefilt6.npy',history1orth.history)

# In[Takens training 6]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500 ,restore_best_weights=True)
NN1tak = buildNN(-20,20)
NN1tak.load_weights("tak_noisy_filt5.h5")
NN1tak.compile(loss="MSE", optimizer=Adam(learning_rate=5e-4), metrics=["MAPE"])
history1tak = NN1tak.fit(data_input_tak_states, data_target[0], batch_size=10000, epochs=7000, verbose=0, shuffle=1, callbacks=[es])
NN1tak.save_weights("tak_noisy_filt6.h5")
np.save('histtaknoise6.npy',history1tak.history)

# In[RC training 7]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1orth = buildNN(-20,20)
NN1orth.load_weights("orth_noisy_filt6.h5")
NN1orth.compile(loss = "MSE", optimizer =Adam(learning_rate =5e-5),metrics=["MAPE"])
history1orth = NN1orth.fit(data_input_rc_states,data_target[0], batch_size=10000, epochs =7000, verbose = 1, shuffle=1,callbacks=[es])
NN1orth.save_weights("orth_noisy_filt7.h5")
np.save('historthnoisefilt7.npy',history1orth.history)

# In[Takens training 7]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500 ,restore_best_weights=True)
NN1tak = buildNN(-20,20)
NN1tak.load_weights("tak_noisy_filt6.h5")
NN1tak.compile(loss="MSE", optimizer=Adam(learning_rate=5e-5), metrics=["MAPE"])
history1tak = NN1tak.fit(data_input_tak_states, data_target[0], batch_size=10000, epochs=7000, verbose=0, shuffle=1, callbacks=[es])
NN1tak.save_weights("tak_noisy_filt7.h5")
np.save('histtaknoisefilt7.npy',history1tak.history)

# In[RC training 8]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience = 500 ,restore_best_weights=True)
NN1orth = buildNN(-20,20)
NN1orth.load_weights("orth_noisy_filt7.h5")
NN1orth.compile(loss = "MSE", optimizer =Adam(learning_rate =3e-5),metrics=["MAPE"])
history1orth = NN1orth.fit(data_input_rc_states,data_target[0], batch_size=10000, epochs =7000, verbose = 1, shuffle=1,callbacks=[es])
NN1orth.save_weights("orth_noisy_filt8.h5")
np.save('historthnoisefilt8.npy',history1orth.history)

# In[Takens training 8]

## Training
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500 ,restore_best_weights=True)
NN1tak = buildNN(-20,20)
NN1tak.load_weights("tak_noisy_filt7.h5")
NN1tak.compile(loss="MSE", optimizer=Adam(learning_rate=3e-5), metrics=["MAPE"])
history1tak = NN1tak.fit(data_input_tak_states, data_target[0], batch_size=10000, epochs=7000, verbose=0, shuffle=1, callbacks=[es])
NN1tak.save_weights("tak_noisy_filt8.h5")
np.save('histtaknoisefilt8.npy',history1tak.history)

