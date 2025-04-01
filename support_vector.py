from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

x_tr_data = pd.read_excel(r'C:\Users\Gaming\Desktop\Ders\Damar Tıkanıklığı\Damar_AI\750_new\Initial_parameter_training_x.xlsx', header = None)
X_Train = np.array(x_tr_data)

x_veri_data = pd.read_excel(r'C:\Users\Gaming\Desktop\Ders\Damar Tıkanıklığı\Damar_AI\750_new\Initial_parameter_test_x.xlsx', header = None) 
X_Veri = np.array(x_veri_data)

y_train_data = pd.read_excel(r'C:\Users\Gaming\Desktop\Ders\Damar Tıkanıklığı\Damar_AI\750_new\Initial_parameter_training_y.xlsx', header = None) 
y_Train = np.array(y_train_data)

y_veri_data = pd.read_excel(r'C:\Users\Gaming\Desktop\Ders\Damar Tıkanıklığı\Damar_AI\750_new\Initial_parameter_test_y.xlsx', header = None) 
y_Veri = np.array(y_veri_data)


confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)

regr = svm.SVR()
xx = np.arange(len(y_Veri))

#######SHEAR##########
#Standartize(normalize) the data
scaler_y_shear = StandardScaler()
y_Train_shear_scaled = scaler_y_shear.fit_transform(y_Train[:, 0].reshape(-1, 1)).ravel()

regr.fit(X_Train,y_Train_shear_scaled)
y_p_shear_scaled = regr.predict(X_Veri)

# Predict and rescale the predictions back to the original scale
y_p_shear_scaled = regr.predict(X_Veri)
y_p_shear = scaler_y_shear.inverse_transform(y_p_shear_scaled.reshape(-1, 1)).ravel()

relative_shear = (np.absolute((y_Veri[:,0] - y_p_shear) / y_Veri[:,0])) * 100 
error_shear =  (np.absolute((y_Veri[:,0] - y_p_shear) / y_Veri[:,0])) * 100 
mean_error_shear = np.mean(error_shear)
std_error_shear = np.std(error_shear)
sigma_shear = std_error_shear / np.sqrt(len(y_Veri))
shear_lower = mean_error_shear - z_score * sigma_shear
shear_upper = mean_error_shear + z_score * sigma_shear


#plt.figure(0)
#plt.scatter(xx,relative_shear)
#plt.ylim(0, 40)
#plt.title("Network Shear, Y limit 40")

#now remove values higher than the condition
threshold_shear = 100000
mask_shear = relative_shear <= threshold_shear

y_p_filtered_shear = y_p_shear[mask_shear]
y_Veri_filtered_shear = y_Veri[mask_shear]


#recalculate error for better variables & visualation
relative_filtered_shear = np.mean((np.absolute((y_Veri_filtered_shear[:,0] - y_p_filtered_shear) / y_Veri_filtered_shear[:,0])) * 100)



#######VELOCITY##########
#Standartize(normalize) the data
scaler_y_velo = StandardScaler()
y_Train_velo_scaled = scaler_y_velo.fit_transform(y_Train[:, 1].reshape(-1, 1)).ravel()

regr.fit(X_Train,y_Train_velo_scaled)
y_p_velo_scaled = regr.predict(X_Veri)

# Predict and rescale the predictions back to the original scale
y_p_velo_scaled = regr.predict(X_Veri)
y_p_velo = scaler_y_velo.inverse_transform(y_p_velo_scaled.reshape(-1, 1)).ravel()


relative_velo = (np.absolute((y_Veri[:,1] - y_p_velo) / y_Veri[:,1])) * 100 
error_velo =  (np.absolute((y_Veri[:,1] - y_p_velo) / y_Veri[:,1])) * 100 
mean_error_velo = np.mean(error_velo)
std_error_velo = np.std(error_velo)
sigma_velo = std_error_velo / np.sqrt(len(y_Veri))
velo_lower = mean_error_velo - z_score * sigma_velo
velo_upper = mean_error_velo + z_score * sigma_velo


#plt.figure(1)
#plt.scatter(xx,relative_velo)
#plt.ylim(0, 25)
#plt.title("Network Velociy, Y limit 25")

#now remove values higher than the condition
threshold_velo = 50000
mask_velo = relative_velo <= threshold_velo

y_p_filtered_velo = y_p_velo[mask_velo]
y_Veri_filtered_velo = y_Veri[mask_velo]


#recalculate error for better variables & visualation
relative_filtered_velo = np.mean((np.absolute((y_Veri_filtered_velo[:,1] - y_p_filtered_velo) / y_Veri_filtered_velo[:,1])) * 100)


relative_shear = np.mean(relative_shear)
relative_velo = np.mean(relative_velo)

print(f"Error for Shear: {relative_shear}")
print(f"Error for Velo: {relative_velo}")
total_error = (relative_shear + relative_velo) / 2
print(f"Total error (average): {total_error}")

print(f"Mean prediction error (shear): {mean_error_shear:.4f}")
print(f"95% CI of model error (shear): ({shear_lower:.4f}, {shear_upper:.4f})")
print(f"Mean prediction error (velo): {mean_error_velo:.4f}")
print(f"95% CI of model error (velo): ({velo_lower:.4f}, {velo_upper:.4f})")














