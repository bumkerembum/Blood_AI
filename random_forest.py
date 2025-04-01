import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

# Import data
x_tr_data = pd.read_excel(r'C:\Users\Gaming\Desktop\Ders\Damar Tıkanıklığı\Damar_AI\450_new\Initial_parameter_training_x.xlsx', header = None)
X_Train = np.array(x_tr_data)

x_veri_data = pd.read_excel(r'C:\Users\Gaming\Desktop\Ders\Damar Tıkanıklığı\Damar_AI\450_new\Initial_parameter_test_x.xlsx', header = None) 
X_Veri = np.array(x_veri_data)

y_train_data = pd.read_excel(r'C:\Users\Gaming\Desktop\Ders\Damar Tıkanıklığı\Damar_AI\450_new\Initial_parameter_training_y.xlsx', header = None) 
y_Train = np.array(y_train_data)

y_veri_data = pd.read_excel(r'C:\Users\Gaming\Desktop\Ders\Damar Tıkanıklığı\Damar_AI\450_new\Initial_parameter_test_y.xlsx', header = None) 
y_Veri = np.array(y_veri_data)


confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)


# Initialize scalers for each target column
scaler_shear = StandardScaler()
scaler_velo = StandardScaler()

# Standardize the target variables (scaling both columns independently)
y_Train_scaled = np.column_stack((
    scaler_shear.fit_transform(y_Train[:, 0].reshape(-1, 1)).ravel(),
    scaler_velo.fit_transform(y_Train[:, 1].reshape(-1, 1)).ravel(),
))

y_Veri_scaled = np.column_stack((
    scaler_shear.transform(y_Veri[:, 0].reshape(-1, 1)).ravel(),
    scaler_velo.transform(y_Veri[:, 1].reshape(-1, 1)).ravel(),
))

# Initialize Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # n_estimators=100 means 100 trees in the forest

# Train the Random Forest model
rf_regressor.fit(X_Train, y_Train_scaled)

# Predict on the validation set
y_pred_scaled = rf_regressor.predict(X_Veri)

# Inverse scale the predictions back to original scale
y_pred = np.column_stack((
    scaler_shear.inverse_transform(y_pred_scaled[:, 0].reshape(-1, 1)).ravel(),
    scaler_velo.inverse_transform(y_pred_scaled[:, 1].reshape(-1, 1)).ravel()
))

# Calculate Mean Percentage Error (MPE) for each column
relative_shear = np.mean((np.abs((y_Veri[:, 0] - y_pred[:, 0]) / y_Veri[:, 0])) * 100)
relative_velo = np.mean((np.abs((y_Veri[:, 1] - y_pred[:, 1]) / y_Veri[:, 1])) * 100)

# Calculate total MPE (average of MPE for both columns)
total_error = (relative_shear + relative_velo) / 2

# Print the results
print(f"Error for Shear: {relative_shear}")
print(f"Error for Velo: {relative_velo}")
print(f"Total error (average): {total_error}")


# CI Calculation

error_shear =  (np.absolute((y_Veri[:,0] - y_pred[:,0]) / y_Veri[:,0])) * 100 
mean_error_shear = np.mean(error_shear)
std_error_shear = np.std(error_shear)
sigma_shear = std_error_shear / np.sqrt(len(y_Veri))
shear_lower = mean_error_shear - z_score * sigma_shear
shear_upper = mean_error_shear + z_score * sigma_shear

error_velo =  (np.absolute((y_Veri[:,1] - y_pred[:,1]) / y_Veri[:,1])) * 100 
mean_error_velo = np.mean(error_velo)
std_error_velo = np.std(error_velo)
sigma_velo = std_error_velo / np.sqrt(len(y_Veri))
velo_lower = mean_error_velo - z_score * sigma_velo
velo_upper = mean_error_velo + z_score * sigma_velo

print(f"Mean prediction error (shear): {mean_error_shear:.4f}")
print(f"95% CI of model error (shear): ({shear_lower:.4f}, {shear_upper:.4f})")
print(f"Mean prediction error (velo): {mean_error_velo:.4f}")
print(f"95% CI of model error (velo): ({velo_lower:.4f}, {velo_upper:.4f})")
