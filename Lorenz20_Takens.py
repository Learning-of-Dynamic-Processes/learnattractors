# In[Takens multi-step 1]
filename_tak = "tak_noisy_filt8.h5"
NN1tak = buildNN(-20,20)
NN1tak.load_weights(filename_tak)
fc_length = 1000
start_point = 3500
start_time = burnin + start_point
fc_tak_x = np.zeros(fc_length+start_time,)
for i in range(fc_length):
    if i==0:
        xt = data_input_tak_states[i + start_time - burnin,:]
        fc_tak_x[i+start_time] = NN1tak.predict(xt.reshape(1,d)).flatten()
    else:
        xt = A@xt + C*fc_tak_x[i-1+start_time]
        fc_tak_x[i+start_time] = NN1tak.predict(xt.reshape(1,d)).flatten()
   
mse_fc_tak_x_1 =(fc_tak_x[start_time:fc_length+start_time]-data_target[0][start_time - burnin:fc_length+start_time - burnin])**2
mmse_fc_tak_x_1 = np.mean(mse_fc_tak_x_1)

fc_length = 1000
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.set_size_inches(20, 21)
fig.suptitle("Takens NN: Pseudo-Multistep Insample Forecast x axis", fontsize=30)
ax1.plot(data_target[0][start_time-burnin:fc_length+start_time-burnin],linewidth=3)
ax1.set_xlabel("Lorenz x-axis", fontsize=30)
ax2.plot(fc_tak_x[start_time:fc_length+start_time],linewidth=3)
ax2.set_xlabel("x-forecast", fontsize=30)
ax3.plot(mse_fc_tak_x_1,linewidth=3)
ax3.set_xlabel("mse", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

print(mmse_fc_tak_x_1)

# In[Takens one-step ahead 1]

NN1tak = buildNN(-20,20)
NN1tak.load_weights(filename_tak)
fc_length = 3000
start_point = 2000
burnin = 100001
start_time = burnin + start_point
fc_tak_x = np.zeros(fc_length+start_time,)
for i in range(fc_length):
    if i==0:
        xt = data_input_tak_states[i + start_time - burnin,:]
        fc_tak_x[i+start_time] = NN1tak.predict(xt.reshape(1,d)).flatten()
    else:
        xt = A@xt + C*LS_noisy_x[i - 1 + start_time]
        fc_tak_x[i+start_time] = NN1tak.predict(xt.reshape(1,d)).flatten()
   
mse_1stepfc_tak_x_1 =(fc_tak_x[start_time:fc_length+start_time]-data_target[0][start_time-burnin:fc_length+start_time-burnin])**2
mmse_1stepfc_tak_x_1 = np.mean(mse_1stepfc_tak_x_1)

plotlength = 1200
ls = np.linspace(0,1,plotlength)
cmap = cm.viridis(ls)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
basewidth=12

fig.set_size_inches(basewidth, basewidth*2.4)
fontsizeval=1.8*basewidth
labelfontsizeval=2.5*basewidth
midfontsizeval=2*basewidth
#fig.suptitle("RC NN: Pseudo-Onestep Insample Forecast x axis", fontsize=30)
#ax1.plot(data_target[0][start_time-burnin:plotlength+start_time-burnin], linewidth=3)
ax1.scatter(t[start_time:plotlength+start_time],data_target[0][start_time-burnin:plotlength+start_time-burnin],color = cmap, s = 5,linewidth=3)
ax1.title.set_size(labelfontsizeval)
ax1.title.set_text(r"$u$-component of a trajectory of the Lorenz system")
ax1.tick_params(axis='both', which='major', labelsize=fontsizeval)
ax2.scatter(t[start_time:plotlength+start_time], LS_noisy_x[start_time:plotlength+start_time],color = cmap, s = 5,linewidth=3)
ax2.title.set_size(labelfontsizeval)
ax2.title.set_text(r"$u$-component with additive $N(0,0.25)$ noise")
ax2.tick_params(axis='both', which='major', labelsize=fontsizeval)
#ax3.plot(fc_orth1_x[start_time:plotlength+start_time],'g',linewidth=3)
ax3.scatter(t[start_time:plotlength+start_time],fc_tak_x[start_time:plotlength+start_time],color = cmap, s = 5,linewidth=3)
ax3.title.set_size(labelfontsizeval)
ax3.title.set_text(r"one-step ahead forecast of the $u$-component")
ax3.tick_params(axis='both', which='major', labelsize=fontsizeval)
ax4.plot(mse_1stepfc_tak_x_1,linewidth=3)
ax4.title.set_size(labelfontsizeval)
ax4.title.set_text("squared error")#, fontsize=labelfontsizeval)
ax4.set_xlabel("points corresponding to time steps $t\in [1020,1032]$", fontsize=midfontsizeval)
ax4.tick_params(axis='both', which='major', labelsize=fontsizeval)
plt.xticks(fontsize=fontsizeval)
plt.yticks(fontsize=fontsizeval)
plt.show()
fig.savefig('Lorenz_traj_ucomponent_Takens.png', format='png', dpi=300)

print(mmse_1stepfc_tak_x_1)

# In[Plot results]

# Plot the Lorenz attractor using a Matplotlib 3D projection
burnin = 100001
start_time = burnin + 2000
plotlength = 3000
ls = np.linspace(0,1,plotlength)
cmap = cm.viridis(ls)
basewidth=8
fig.set_size_inches(basewidth, basewidth*2.4)
fontsizeval=1.8*basewidth
labelfontsizeval=2.5*basewidth
midfontsizeval=2*basewidth
# Plot trajectory of the Lorenz system
fig = plt.figure(figsize = (10,10))
#fig.suptitle("A trajectory of the Lorenz system")
ax = fig.gca(projection='3d')
ax.scatter(LS_x[start_time:plotlength+start_time], LS_y[start_time:plotlength+start_time], LS_z[start_time:plotlength+start_time], color = cmap, s = 5)
ax.set_xlabel("$u$", fontsize=midfontsizeval)
ax.set_ylabel("$v$", fontsize=midfontsizeval)
ax.set_zlabel("$w$", fontsize=midfontsizeval)
ax.tick_params(axis='both', which='major', labelsize=fontsizeval)
ax.view_init(15, -50)
fig.savefig('Lorenz_traj.png', format='png', dpi=300)
plt.show()

# Plot trajectory of the Lorenz system with noisy u and original v, w
fig = plt.figure(figsize = (10,10))
#fig.suptitle("A trajectory of the Lorenz system with u-component contaminated with N(0,0,25) noise")
ax = fig.gca(projection='3d')
ax.scatter(LS_noisy_x[start_time:plotlength+start_time], LS_y[start_time:plotlength+start_time], LS_z[start_time:plotlength+start_time], color = cmap, s = 5)
ax.set_xlabel("$u$", fontsize=midfontsizeval)
ax.set_ylabel("$v$", fontsize=midfontsizeval)
ax.set_zlabel("$w$", fontsize=midfontsizeval)
ax.tick_params(axis='both', which='major', labelsize=fontsizeval)
ax.view_init(15, -50)
fig.savefig('Lorenz_traj_ucompnoisy.png', format='png', dpi=300)
plt.show()

# Plot the result of Takens+NN
fig = plt.figure(figsize = (10,10))
#fig.suptitle("A filtered and one-step forecasted trajectory")
ax = fig.gca(projection='3d')
ax.scatter(fc_tak_x[start_time:plotlength+start_time], LS_y[start_time:plotlength+start_time], LS_z[start_time:plotlength+start_time], color = cmap, s = 5)
ax.set_xlabel("$u$", fontsize=midfontsizeval)
ax.set_ylabel("$v$", fontsize=midfontsizeval)
ax.set_zlabel("$w$", fontsize=midfontsizeval)
ax.tick_params(axis='both', which='major', labelsize=fontsizeval)
ax.view_init(15, -50)
fig.savefig('Lorenz_traj_filtfrcst_Takens.png', format='png', dpi=300)
plt.show()
# In[in-sample training 1]:

#in-sample
NN1tak = buildNN(-20,20)
NN1tak.load_weights(filename_tak)
LS_tak_x_is = NN1tak.predict(data_input_tak_states)
mse_is_tak_x_1 =(LS_tak_x_is[start_time - burnin:fc_length+start_time - burnin].T-data_target[0][start_time - burnin:fc_length+start_time - burnin])**2
mse_is_tak_x_1 = mse_is_tak_x_1[0]
mmse_is_tak_x_1 = np.mean(mse_is_tak_x_1)

start_time = 3000
plotlength = 1000
fc_length = plotlength
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.set_size_inches(20, 21)
fig.suptitle("Takens NN: Insample x axis", fontsize=30)
ax1.plot(data_target[0][start_time:fc_length+start_time],linewidth=3)
ax1.set_xlabel("Lorenz x-axis", fontsize=30)
ax2.plot(LS_tak_x_is[start_time:fc_length+start_time],linewidth=3)
ax2.set_xlabel("x-fit", fontsize=30)
ax3.plot(mse_is_tak_x_1,linewidth=3)
ax3.set_xlabel("Lorenz x-axis", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

print(mmse_is_tak_x_1)


# In[Takens multi-step 1]
NN1tak = buildNN(-20,20)
NN1tak.load_weights(filename_tak)
fc_length = 1000
ind = 0
mmse_fc_tak_x_1 = np.zeros(700,)
for t in range(3000,10000,10):
    
    print(t)
    start_point = t
    start_time = burnin + start_point
    fc_tak_x = np.zeros(fc_length+start_time,)
    for i in range(fc_length):
        if i==0:
            xt = data_input_tak_states[i + start_time - burnin,:]
            fc_tak_x[i+start_time] = NN1tak.predict(xt.reshape(1,d)).flatten()
        else:
            xt = A@xt + C*fc_tak_x[i-1+start_time]
            fc_tak_x[i+start_time] = NN1tak.predict(xt.reshape(1,d)).flatten()
       
    mse_fc_tak_x_1 =(fc_tak_x[start_time:fc_length+start_time]-data_target[0][start_time - burnin:fc_length+start_time - burnin])**2
    mmse_fc_tak_x_1[ind] = np.mean(mse_fc_tak_x_1)
    ind += 1

plt.plot(mmse_fc_tak_x_1)
mean_mmse_fc_tak_x_1 = np.mean(mmse_fc_tak_x_1)
print(mean_mmse_fc_tak_x_1)

# In[Takens whole system]
NN1tak_x = buildNN(-20,20)
filename_tak_x = "tak_noisy_filt7.h5"
NN1tak_x.load_weights(filename_tak_x)
NN1tak_y = buildNN(-28,29)
filename_tak_y = "tak_noisy_filty1.h5"
NN1tak_y.load_weights(filename_tak_y)
# NN1tak_z = buildNN(-20,20)
# filename_tak_z = "tak_noisy_filtz1.h5"
# NN1tak_z.load_weights(filename_tak_z)
fc_length = 1000
start_point = 3000
start_time = burnin + start_point
fc_tak_x = np.zeros(fc_length+start_time,)
fc_tak_y = np.zeros(fc_length+start_time,)
fc_tak_z = np.zeros(fc_length+start_time,)
for i in range(fc_length):
    if i==0:
        xt = data_input_tak_states[i + start_time - burnin,:]
        fc_tak_x[i+start_time] = NN1tak_x.predict(xt.reshape(1,d)).flatten()
        fc_tak_y[i+start_time] = NN1tak_y.predict(xt.reshape(1,d)).flatten()
        #fc_tak_z[i+start_time] = NN1tak_z.predict(xt.reshape(1,d)).flatten()
    else:
        xt = A@xt + C*fc_tak_x[i-1+start_time]
        fc_tak_x[i+start_time] = NN1tak_x.predict(xt.reshape(1,d)).flatten()
        fc_tak_y[i+start_time] = NN1tak_y.predict(xt.reshape(1,d)).flatten()
        #fc_tak_z[i+start_time] = NN1tak_z.predict(xt.reshape(1,d)).flatten()
    
mse_fc_tak_x_1 =(fc_tak_x[start_time:fc_length+start_time]-data_target[0][start_time - burnin:fc_length+start_time - burnin])**2
mmse_fc_tak_x_1 = np.mean(mse_fc_tak_x_1)
mse_fc_tak_y_1 =(fc_tak_y[start_time:fc_length+start_time]-data_target[1][start_time - burnin:fc_length+start_time - burnin])**2
mmse_fc_tak_y_1 = np.mean(mse_fc_tak_y_1)
# mse_fc_tak_z_1 =(fc_tak_z[start_time:fc_length+start_time]-data_target[2][start_time - burnin:fc_length+start_time - burnin])**2
# mmse_fc_tak_z_1 = np.mean(mse_fc_tak_z_1)
fc_length = 1000
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.set_size_inches(20, 21)
fig.suptitle("Takens NN: Pseudo-Multistep Insample Forecast x axis", fontsize=30)
ax1.plot(data_target[0][start_time-burnin:fc_length+start_time-burnin],linewidth=3)
ax1.set_xlabel("Lorenz x-axis", fontsize=30)
ax2.plot(fc_tak_x[start_time:fc_length+start_time],linewidth=3)
ax2.set_xlabel("x-forecast", fontsize=30)
ax3.plot(mse_fc_tak_x_1,linewidth=3)
ax3.set_xlabel("mse", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.set_size_inches(20, 21)
fig.suptitle("Takens NN: Pseudo-Multistep Insample Forecast x axis", fontsize=30)
ax1.plot(data_target[1][start_time-burnin:fc_length+start_time-burnin],linewidth=3)
ax1.set_xlabel("Lorenz y-axis", fontsize=30)
ax2.plot(fc_tak_y[start_time:fc_length+start_time],linewidth=3)
ax2.set_xlabel("x-forecast", fontsize=30)
ax3.plot(mse_fc_tak_y_1,linewidth=3)
ax3.set_xlabel("mse", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# fig.set_size_inches(20, 21)
# fig.suptitle("Takens NN: Pseudo-Multistep Insample Forecast x axis", fontsize=30)
# ax1.plot(data_target[0][start_time-burnin:fc_length+start_time-burnin],linewidth=3)
# ax1.set_xlabel("Lorenz x-axis", fontsize=30)
# ax2.plot(fc_tak_x[start_time:fc_length+start_time],linewidth=3)
# ax2.set_xlabel("x-forecast", fontsize=30)
# ax3.plot(mse_fc_tak_x_1,linewidth=3)
# ax3.set_xlabel("mse", fontsize=30)
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# plt.show()
print(mmse_fc_tak_x_1)
print(mmse_fc_tak_y_1)
#print(mmse_fc_tak_z_1)