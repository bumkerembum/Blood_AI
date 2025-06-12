import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

tf.compat.v1.reset_default_graph()

# Import Files
test_x_data = pd.read_excel('location', header=None)
Test_x = np.array(test_x_data)
                                                                                                                                        
test_y_data = pd.read_excel('location', header=None)
Test_y = np.array(test_y_data)


confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)


#Get scaling data 

y_tr_data = pd.read_excel('location', header = None)
y_Train = np.array(y_tr_data)


scaler_shear = StandardScaler()
scaler_velo = StandardScaler()

y_Train_scaled = np.column_stack((
    scaler_shear.fit_transform(y_Train[:, 0].reshape(-1, 1)).ravel(),
    scaler_velo.fit_transform(y_Train[:, 1].reshape(-1, 1)).ravel(),

))

y_Test_scaled = np.column_stack((
    scaler_shear.transform(Test_y[:, 0].reshape(-1, 1)).ravel(),
    scaler_velo.transform(Test_y[:, 1].reshape(-1, 1)).ravel(),
))


# Start a session
with tf.compat.v1.Session() as sess:
    # Load the meta graph and weights
    saver = tf.compat.v1.train.import_meta_graph('/location/model.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint('location'))

    # Get the default graph
    graph = tf.compat.v1.get_default_graph()
    
    # Get the placeholders and operation by name
    X = graph.get_tensor_by_name("X:0")
    prediction = graph.get_tensor_by_name("prediction:0")  # Adjust this tensor name to match the actual output tensor from your model

    # Prepare the feed dictionary
    feed_dict = {X: Test_x}
    
    # Run the session to get the predictions
    Test_prediction_scaled = sess.run(prediction, feed_dict=feed_dict)
    
    # Print the results
    #print(f'Test Predictions: {Test_prediction}')


# Invert scaled Predictions

Test_prediction = np.column_stack((
    scaler_shear.inverse_transform(Test_prediction_scaled[:, 0].reshape(-1, 1)).ravel(),
    scaler_velo.inverse_transform(Test_prediction_scaled[:, 1].reshape(-1, 1)).ravel() 
    ))


# Calculations of error

shear_1 = np.mean(np.power(Test_prediction[:,0] - Test_y[:,0], 2))
velo_1 = np.mean(np.power(Test_prediction[:,1] - Test_y[:,1], 2))
loss_1 = (shear_1 + velo_1 ) / 2


shear_2 = np.mean((np.absolute((Test_y[:,0] - Test_prediction[:,0]) / Test_y[:,0])) * 100 )
velo_2 = np.mean((np.absolute((Test_y[:,1] - Test_prediction[:,1]) / Test_y[:,1])) * 100 )
loss_2 = (shear_2 + velo_2 ) / 2

print(f'For Root Mean Square Eror: \nShear: {shear_2}, Velocity: {velo_2}, Total: {loss_2} \n\n')


# CI Calculation

error_shear = np.absolute((Test_y[:,0] - Test_prediction[:,0]) / Test_y[:,0]) * 100 
std_error_shear = np.std(error_shear)
sigma_shear = std_error_shear / np.sqrt(len(Test_y))
shear_lower = shear_2 - z_score * sigma_shear
shear_upper = shear_2 + z_score * sigma_shear

error_velo = np.absolute((Test_y[:,1] - Test_prediction[:,1]) / Test_y[:,1]) * 100 
std_error_velo = np.std(error_velo)
sigma_velo = std_error_velo / np.sqrt(len(Test_y))
velo_lower = velo_2 - z_score * sigma_velo
velo_upper = velo_2 + z_score * sigma_velo

print(f"Mean prediction error (shear): {shear_2:.4f}")
print(f"95% CI of model error (shear): ({shear_lower:.4f}, {shear_upper:.4f})")
print(f"Mean prediction error (velo): {velo_2:.4f}")
print(f"95% CI of model error (velo): ({velo_lower:.4f}, {velo_upper:.4f})")



