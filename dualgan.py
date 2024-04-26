"""DualGAN: Dual Adversarial Time Series Generation with Generative Adversarial Networks and Autoencoders

"""

# Necessary Packages
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from metrics.discriminative_metrics import discriminative_score_metrics

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator

def dualgan (ori_data, parameters, num_samples):
  """DualGAN function.
  
  Use original data as training set to generater synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: DualGAN network parameters
    
  Returns:
    - generated_data: generated time-series data
  """
  # Initialization on the Graph
  tf.compat.v1.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  
  def MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
      
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)
              
  ## Build a RNN networks          
  
  # Network Parameters
  if parameters['hidden_dim'] == 'same':
     hidden_dim = dim
  else:  
     hidden_dim   = parameters['hidden_dim'] 
        
  num_layers   = parameters['num_layer']
  iterations   = parameters['iterations']
  batch_size   = parameters['batch_size']
  z_dim        = dim
  gamma        = 1
  beta         = 1

  module_first  = 'gru'
  module_second  = 'lstm'

    
  # Input place holders
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name = "myinput_z")
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")
    
  final_generated = []
  saver = None
  global_summing = 100
  
  def embedder (X, T):
    """Embedding network between original feature space to latent space.
    
    Args:
      - X: input time-series features
      - T: input time information
      
    Returns:
      - H: embeddings
    """
    with tf.compat.v1.variable_scope("embedder", reuse = tf.compat.v1.AUTO_REUSE):
        
      e_cell_first = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_first, hidden_dim) for _ in range(num_layers)])
      e_outputs_first, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell_first, X, dtype=tf.float32, sequence_length = T)
    
      e_cell_second = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_second, hidden_dim) for _ in range(num_layers)])
      e_outputs_second, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell_second, X, dtype=tf.float32, sequence_length = T)
            
      combined = tf.concat([e_outputs_first, e_outputs_second], axis=-1)

      # Dimensionality reduction to match input attribute size
      H = tf.compat.v1.layers.dense(combined, dim, activation=tf.nn.sigmoid)
        
    return H
      
  def recovery (H, T):   
    """Recovery network from latent space to original space.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - X_tilde: recovered data
    """     
    with tf.compat.v1.variable_scope("recovery", reuse = tf.compat.v1.AUTO_REUSE):       
        
      r_cell_first = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_first, hidden_dim) for _ in range(num_layers)])
      r_outputs_first, r_last_states = tf.compat.v1.nn.dynamic_rnn(r_cell_first, H, dtype=tf.float32, sequence_length = T)
    
      r_cell_second = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_second, hidden_dim) for _ in range(num_layers)])
      r_outputs_second, r_last_states = tf.compat.v1.nn.dynamic_rnn(r_cell_second, H, dtype=tf.float32, sequence_length = T)
    
      combined = tf.concat([r_outputs_first, r_outputs_second], axis=-1)

      # Dimensionality reduction to match input attribute size
      X_tilde = tf.compat.v1.layers.dense(combined, dim, activation=tf.nn.sigmoid)
    
    return X_tilde
    
  def generator (Z, T):  
    """Generator function: Generate time-series data in latent space.
    
    Args:
      - Z: random variables
      - T: input time information
      
    Returns:
      - E: generated embedding
    """        
    with tf.compat.v1.variable_scope("generator", reuse = tf.compat.v1.AUTO_REUSE): 
    
      g_cell_first = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_first, hidden_dim) for _ in range(num_layers)])
      g_outputs_first, g_last_states = tf.compat.v1.nn.dynamic_rnn(g_cell_first, Z, dtype=tf.float32, sequence_length = T)
    
      g_cell_second = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_second, hidden_dim) for _ in range(num_layers)])
      g_outputs_second, g_last_states = tf.compat.v1.nn.dynamic_rnn(g_cell_second, Z, dtype=tf.float32, sequence_length = T)
            
      combined = tf.concat([g_outputs_first, g_outputs_second], axis=-1)

      # Dimensionality reduction to match input attribute size
      E = tf.compat.v1.layers.dense(combined, dim, activation=tf.nn.sigmoid)
    
    return E
      
  def supervisor (H, T): 
    """Generate next sequence using the previous sequence.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """          
    with tf.compat.v1.variable_scope("supervisor", reuse = tf.compat.v1.AUTO_REUSE):   
    
      s_cell_first = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_first, hidden_dim) for _ in range(num_layers)])
      s_outputs_first, s_last_states = tf.compat.v1.nn.dynamic_rnn(s_cell_first, H, dtype=tf.float32, sequence_length = T)
    
      s_cell_second = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_second, hidden_dim) for _ in range(num_layers)])
      s_outputs_second, s_last_states = tf.compat.v1.nn.dynamic_rnn(s_cell_second, H, dtype=tf.float32, sequence_length = T)
            
      combined = tf.concat([s_outputs_first, s_outputs_second], axis=-1)

      # Dimensionality reduction to match input attribute size
      S = tf.compat.v1.layers.dense(combined, dim, activation=tf.nn.sigmoid)
    
    return S
          
  def discriminator (H, T):
    """Discriminate the original and synthetic time-series data.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """        
    with tf.compat.v1.variable_scope("discriminator", reuse = tf.compat.v1.AUTO_REUSE):
    
      d_cell_first = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_first, hidden_dim) for _ in range(num_layers)])
      d_outputs_first, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell_first, H, dtype=tf.float32, sequence_length = T)
    
      d_cell_second = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_second, hidden_dim) for _ in range(num_layers)])
      d_outputs_second, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell_second, H, dtype=tf.float32, sequence_length = T)
            
      combined = tf.concat([d_outputs_first, d_outputs_second], axis=-1)

      # Dimensionality reduction to match input attribute size
      Y_hat = tf.compat.v1.layers.dense(combined, 1, activation=None)
    
    return Y_hat   



  def ae_discriminator (X, T):
    """Discriminate the original and synthetic time-series data.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """        
    with tf.compat.v1.variable_scope("ae_discriminator", reuse = tf.compat.v1.AUTO_REUSE):
    
      d_cell_first = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_first, hidden_dim) for _ in range(num_layers)])
      d_outputs_first, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell_first, X, dtype=tf.float32, sequence_length = T)
    
      d_cell_second = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_second, hidden_dim) for _ in range(num_layers)])
      d_outputs_second, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell_second, X, dtype=tf.float32, sequence_length = T)
            
      combined = tf.concat([d_outputs_first, d_outputs_second], axis=-1)

      # Dimensionality reduction to match input attribute size
      Y_hat_ae = tf.compat.v1.layers.dense(combined, 1, activation=None)
    
    return Y_hat_ae 

    
  # Embedder & Recovery
  H = embedder(X, T)
  X_tilde = recovery(H, T)

  Y_ae_fake = ae_discriminator(X_tilde, T)
  Y_ae_real = ae_discriminator(X, T) 
    
  # Generator
  E_hat = generator(Z, T)
  H_hat = supervisor(E_hat, T)
  H_hat_supervise = supervisor(H, T)
    
  # Synthetic data
  X_hat = recovery(H_hat, T)
    
  Y_ae_fake_e = ae_discriminator(X_hat, T)
  X_tilde_fake_second = recovery(E_hat, T)
  Y_ae_fake_e_second = ae_discriminator(X_tilde_fake_second, T)
    
  # Discriminator
  Y_fake = discriminator(H_hat, T)
  Y_real = discriminator(H, T)     
  Y_fake_e = discriminator(E_hat, T)
    
    
  # Variables        
  e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]
  d_ae_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('ae_discriminator')]

    
  # Discriminator loss
  D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
  D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
  D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
  D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e


  # AE Discriminator loss
  D_ae_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_ae_real), Y_ae_real)
  D_ae_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_ae_fake), Y_ae_fake)
  D_ae_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_ae_fake_e), Y_ae_fake_e)
  D_ae_loss_fake_e_second = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_ae_fake_e_second), Y_ae_fake_e_second)

  D_ae_loss = D_ae_loss_real + D_ae_loss_fake  

  D_ae_loss_real_second = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_ae_fake), Y_ae_fake)
  D_ae_loss_second = D_ae_loss_real + D_ae_loss_real_second + beta * (D_ae_loss_fake_e + gamma * D_ae_loss_fake_e_second)
    
            
  # Generator loss
  # 1. Adversarial loss
  G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
  G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    
  G_loss_U_ae = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_ae_fake_e), Y_ae_fake_e)
  G_loss_U_ae_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_ae_fake_e_second), Y_ae_fake_e_second)
    
  # 2. Supervised loss
  G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
    
  # 3. Two Momments
  G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
  G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    
  G_loss_V = G_loss_V1 + G_loss_V2

  #---------
  # 4. Time Series Characteristics
   
  W = tf.range(1, seq_len + 1, dtype=tf.float32)
  W = tf.reshape(W, (1, seq_len, 1))  # Reshape to match dimensions
  W = tf.broadcast_to(W, (batch_size, seq_len, dim))  # Expand dimensions to match X and X_hat

  W_sum = tf.reduce_sum(W, axis=1, keepdims=True)  # Sum weights along the seq_len
  W_normalized = W / W_sum  # Normalize weights

  weighted_average_X = tf.reduce_sum(W_normalized * X, axis=1)
  weighted_average_X_hat = tf.reduce_sum(W_normalized * X_hat, axis=1)

  mean_weighted_average_X = tf.reduce_mean(weighted_average_X, axis=0)
  mean_weighted_average_X_hat = tf.reduce_mean(weighted_average_X_hat, axis=0)

  std_weighted_average_X = tf.math.reduce_std(weighted_average_X, axis=0)
  std_weighted_average_X_hat = tf.math.reduce_std(weighted_average_X_hat, axis=0)

  mean_weighted_average_mse = tf.compat.v1.losses.mean_squared_error(mean_weighted_average_X, mean_weighted_average_X_hat)
  std_weighted_average_mse = tf.compat.v1.losses.mean_squared_error(std_weighted_average_X, std_weighted_average_X_hat)

    
  #----

  x= tf.range(seq_len, dtype=tf.float32)

  # Calculate sums needed for the slope formula
  sum_x = tf.reduce_sum(x)
  sum_x2 = tf.reduce_sum(tf.square(x))
  N = seq_len

  # Function to calculate the slope
  def calculate_slope(Y):
    sum_y = tf.reduce_sum(Y, axis=1)
    sum_xy = tf.reduce_sum(x[:, tf.newaxis] * Y, axis=1)
    numerator = N * sum_xy - sum_x * sum_y
    denominator = N * sum_x2 - tf.square(sum_x)
    slope = numerator / denominator
    return slope

  # Calculate slopes for X and X_hat
  slope_X = calculate_slope(X)
  slope_X_hat = calculate_slope(X_hat)
    
  mean_slope_X = tf.reduce_mean(slope_X, axis=0)
  mean_slope_X_hat = tf.reduce_mean(slope_X_hat, axis=0)
    
  std_slope_X = tf.math.reduce_std(slope_X, axis=0)
  std_slope_X_hat = tf.math.reduce_std(slope_X_hat, axis=0)
    
  mean_slope_mse = tf.compat.v1.losses.mean_squared_error(mean_slope_X, mean_slope_X_hat)
  std_slope_mse = tf.compat.v1.losses.mean_squared_error(std_slope_X, std_slope_X_hat)


  #----
    
  def calculate_skewness(data, axis=1):
    N = tf.cast(tf.shape(data)[axis], tf.float32)
    mean = tf.reduce_mean(data, axis=axis, keepdims=True)
    std_dev = tf.math.reduce_std(data, axis=axis, keepdims=True)
    skewness = tf.reduce_sum(((data - mean) / std_dev)**3, axis=axis) * (N / ((N - 1) * (N - 2)))
    return skewness


  
  skew_X = calculate_skewness(X, axis=1)
  skew_X_hat = calculate_skewness(X_hat, axis=1)
  
  mean_skew_X = tf.reduce_mean(skew_X, axis=0)
  mean_skew_X_hat = tf.reduce_mean(skew_X_hat, axis=0)

  std_skew_X = tf.math.reduce_std(skew_X, axis=0)
  std_skew_X_hat = tf.math.reduce_std(skew_X_hat, axis=0)

    
  mean_skew_mse = tf.compat.v1.losses.mean_squared_error(mean_skew_X, mean_skew_X_hat)
  std_skew_mse = tf.compat.v1.losses.mean_squared_error(std_skew_X, std_skew_X_hat)


  #----
  
  def median(data):
    time_size = data.shape[1]
    if time_size % 2 == 1:
      median = data[:, time_size // 2, :]
    else:
      median = (data[:, (time_size // 2) - 1, :] + data[:, time_size // 2, :]) / 2.0
    return median

  median_X = median(X)
  median_X_hat = median(X_hat)
    
  mean_median_X = tf.reduce_mean(median_X, axis=0)
  mean_median_X_hat = tf.reduce_mean(median_X_hat, axis=0)
    
  std_median_X = tf.math.reduce_std(median_X, axis=0)
  std_median_X_hat = tf.math.reduce_std(median_X_hat, axis=0)
    
  mean_median_mse = tf.compat.v1.losses.mean_squared_error(mean_median_X, mean_median_X_hat)
  std_median_mse = tf.compat.v1.losses.mean_squared_error(std_median_X, std_median_X_hat)
    
       
  #---------


  ts_structure = mean_weighted_average_mse + std_weighted_average_mse + mean_slope_mse + std_slope_mse + mean_median_mse + std_median_mse + mean_skew_mse + std_skew_mse

  # 4. Summation
  G_loss = (G_loss_U_ae + gamma * G_loss_U_ae_e) + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V  + 50 * ts_structure
            
  # Embedder network loss
  lambda_c = 0.01
  E_loss_T00 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
  E_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_ae_fake), Y_ae_fake)
  
  E_loss0 = 10*tf.sqrt(E_loss_T00 + lambda_c*E_loss_U)
  E_loss = 10*tf.sqrt(E_loss_T00 + lambda_c * 0.1 *E_loss_U)  + 0.1*G_loss_S
    
  # optimizer
  E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
  E_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
  D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
  D_ae_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_ae_loss, var_list = d_ae_vars)
  D_ae_solver_second = tf.compat.v1.train.AdamOptimizer().minimize(D_ae_loss_second, var_list = d_ae_vars)
  G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)      
  GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)   
        
  ## DualGAN training   
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
    
  # 1. Embedding network training
  print('Start Embedding Network Training')
    
  for itt in range(int(iterations*0.5)):
    for kk in range(2):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
      # Train embedder        
      _, step_e_loss = sess.run([E0_solver, E_loss0], feed_dict={X: X_mb, T: T_mb})        
      # Checkpoint
    
    check_d_ae_loss = sess.run(D_ae_loss, feed_dict={X: X_mb, T: T_mb})
    # Train discriminator (only when the discriminator does not work well)
    if (check_d_ae_loss > 0.15):        
      _, step_d_ae_loss = sess.run([D_ae_solver, D_ae_loss], feed_dict={X: X_mb, T: T_mb})
    
    if itt % 500 == 0 or itt==int(iterations*0.5)-1:
      print('step: '+ str(itt*2) + '/' + str(iterations) + ', AE_loss: ' + str(np.round(step_e_loss,4))
           + ', AE_D_loss: ' + str(np.round(step_d_ae_loss,4))) 
      
  print('Finish Embedding Network Training')
    
  # 2. Training only with supervised loss
  print('Start Training with Supervised Loss Only')
    
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)    
    # Random vector generation   
    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    # Train generator       
    _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})       
    # Checkpoint
    if itt % 1000 == 0 or itt==iterations-1:
      print('step: '+ str(itt)  + '/' + str(iterations) +', S_loss: ' + str(np.round(step_g_loss_s,4)) )
      
  print('Finish Training with Supervised Loss Only')


  print('Start Joint Training')
  
  for itt in range(iterations):
    # Generator training (twice more than discriminator training)
    for kk in range(2):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)               
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Train generator
      _, step_g_loss_u, step_g_loss_u_s, step_g_loss_s, step_g_loss_v, step_g_loss, step_g_loss_ts_structure = sess.run([G_solver, G_loss_U_ae, G_loss_U_ae_e, G_loss_S, G_loss_V, G_loss, ts_structure], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
       # Train embedder        
      _, step_e_loss_t0 = sess.run([E_solver, E_loss], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})   
           
    # Discriminator training        
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
    # Random vector generation
    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    # Check discriminator loss before updating
    
    check_d_ae_loss = sess.run(D_ae_loss_second, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    # Train discriminator (only when the discriminator does not work well)
    if (check_d_ae_loss > 0.15):        
      _, step_d_ae_loss_second, step_d_ae_loss = sess.run([D_ae_solver_second, D_ae_loss_second, D_ae_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
    # Print multiple checkpoints
    if itt % 1000 == 0 or itt==iterations-1:
      print('step: '+ str(itt) + '/' + str(iterations) + 
            ', D_loss: ' + str(np.round(step_d_ae_loss_second,4)) +  
            ', G_loss_u_g: ' + str(np.round(step_g_loss_u,4)) + 
            ', G_loss_u_s: ' + str(np.round(step_g_loss_u_s,4)) + 
            ', G_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
            ', G_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
            ', G_loss_ts: ' + str(np.round(step_g_loss_ts_structure,4)) + 
            ', AE_loss: ' + str(np.round(step_e_loss_t0,4)) +
            ', AE_D_loss: ' + str(np.round(step_d_ae_loss,4))
           )
    

  if itt >= 5000 and (itt % 500 == 0 or itt==iterations-1):
        
        saver = tf.compat.v1.train.Saver()
        Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
        generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})    
        generated_data = list()
        for i in range(no):
            temp = generated_data_curr[i,:ori_time[i],:]
            generated_data.append(temp)
        # Renormalization
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val
        
        
        metric_iteration = 5
        discriminative_score = list()
        for _ in range(metric_iteration):
            temp_disc = discriminative_score_metrics(ori_data, generated_data)
            discriminative_score.append(temp_disc)
            
        mean_real = np.mean(ori_data, axis=0)
        mean_synthetic = np.mean(generated_data, axis=0)
        mse_mean = np.mean((mean_real - mean_synthetic) ** 2)
        
        variance_real = np.var(ori_data, axis=0)
        variance_synthetic = np.var(generated_data, axis=0)
        mse_variance = np.mean((variance_real - variance_synthetic) ** 2)
        mean_dis_score = np.round(np.mean(discriminative_score), 4)
        
        summing = mean_dis_score
            
        if summing <= global_summing:
            global_summing = summing
            final_generated = generated_data
          
        
        
        
        
  print('Finish Joint Training')

  #-------------------------------------------------------------------
    
  if num_samples == "same" and global_summing != 0.5:
    return final_generated
  
  elif num_samples == "same" and global_summing == 0.5:
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})    
    generated_data = list()
    for i in range(no):
        temp = generated_data_curr[i,:ori_time[i],:]
        generated_data.append(temp)
    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val
    
    return generated_data

  else:
    count = int(num_samples / no)
  
    all_generated_data = None
    for c in range(count):
        Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
        generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})    
        generated_data = []
        for i in range(no):
          temp = generated_data_curr[i,:ori_time[i],:]
          generated_data.append(temp)

        # Renormalization
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val

        all_generated_data = np.concatenate((all_generated_data, generated_data))

    return all_generated_data
                
  #-------------------------------------------------------------------
