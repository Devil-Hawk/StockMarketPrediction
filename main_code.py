import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, Attention, MultiHeadAttention
from tensorflow.keras.layers import Flatten, TimeDistributed, RepeatVector, Concatenate, Add, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Multiply, Lambda
import datetime
import pickle
import json
from tqdm import tqdm
import warnings
import multiprocessing

# ====== Enhanced GPU Configuration for TensorFlow Nightly ======
# Print TensorFlow version to confirm we're using nightly
print(f"TensorFlow version: {tf.__version__}")

# Configure GPU with CUDA 12.4
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f" - {gpu}")
        
        # Set memory growth to avoid grabbing all VRAM at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set TensorFlow to use the GPU
        tf.config.set_visible_devices(gpus, 'GPU')
        
        # Set up mixed precision policy for faster training
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled")
        
        # Test GPU with a simple operation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print(f"Test matrix multiplication result device: {c.device}")
        
        # Configure for optimal GPU usage
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Each GPU gets exclusive thread
        os.environ['TF_GPU_THREAD_COUNT'] = '4'  # Multiple threads per GPU
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Another way to enable memory growth
        
        print("GPU successfully configured for TensorFlow")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        raise RuntimeError("GPU configuration failed - cannot continue with training")
else:
    print("No GPU found. This code is optimized for GPU training.")
    raise RuntimeError("GPU is required for this model. Please check CUDA installation.")

# Warning suppressions
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = "final_aggregated_sentiment_stock.csv"
MODEL_PATH = "models/"
RESULTS_PATH = "results/"
LOOKBACK = 120  # Increased lookback window for more context
FORECAST_HORIZON = 1  # Days ahead to predict
BATCH_SIZE = 128  # Increased batch size for better GPU utilization
EPOCHS = 500  # More epochs for longer training
PATIENCE = 50  # Increased patience
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
TARGET_COLUMN = 'Open'  # We predict next day's opening price
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'avg_sentiment']

# Create directories
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Custom metrics - updated for TF nightly with explicit casting
def custom_root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def custom_mean_absolute_percentage_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-7, float('inf')))
    return 100.0 * tf.reduce_mean(diff)

def directional_accuracy(y_true, y_pred):
    """Measures whether prediction direction matches actual direction"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    direction_true = tf.sign(y_true[:, 0] - y_true[:, 1])
    direction_pred = tf.sign(y_pred[:, 0] - y_true[:, 1])
    return tf.reduce_mean(tf.cast(tf.equal(direction_true, direction_pred), 'float32'))

class StockPredictor:
    def __init__(self):
        self.data = None
        self.model = None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self.scalers = {}
        self.history = None
        self.predictions = None
        self.evaluation_metrics = None
        self.feature_columns = FEATURE_COLUMNS
        self.target_column = TARGET_COLUMN
        self.lookback = LOOKBACK
        self.forecast_horizon = FORECAST_HORIZON
    
    def load_data(self):
        """Load and preprocess data"""
        print("Loading data...")
        self.data = pd.read_csv(DATA_PATH)
        
        # Convert date to datetime and set as index
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.data.sort_index(inplace=True)
        
        # Display info about the data
        print(f"Data shape: {self.data.shape}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Features: {', '.join(self.data.columns)}")
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(f"Missing values:\n{missing[missing > 0]}")
            print("Filling missing values with forward fill...")
            self.data.fillna(method='ffill', inplace=True)
            
        # Convert sentiment to float if needed
        if 'avg_sentiment' in self.data.columns:
            self.data['avg_sentiment'] = self.data['avg_sentiment'].astype(float)
            
        # Add technical indicators as features
        self.add_technical_indicators()
        
        # Feature selection
        self.feature_engineering()
        
        print(f"Final data shape: {self.data.shape}")
        
    def add_technical_indicators(self):
        """Add technical analysis indicators as features"""
        print("Adding technical indicators...")
        df = self.data.copy()
        
        # Moving Averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Price Momentum
        df['momentum_1d'] = df['Close'].pct_change(periods=1)
        df['momentum_5d'] = df['Close'].pct_change(periods=5)
        df['momentum_10d'] = df['Close'].pct_change(periods=10)
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        mean_dev = abs(typical_price - typical_price.rolling(window=20).mean()).rolling(window=20).mean()
        df['CCI'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_dev)
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_ma10'] = df['Volume'].rolling(window=10).mean()
            df['volume_ma10_ratio'] = df['Volume'] / df['volume_ma10']
        
        # Sentiment moving average
        if 'avg_sentiment' in df.columns:
            df['sentiment_ma5'] = df['avg_sentiment'].rolling(window=5).mean()
            df['sentiment_ma10'] = df['avg_sentiment'].rolling(window=10).mean()
            df['sentiment_change'] = df['avg_sentiment'].diff()
        
        # Price change prediction target (for classification)
        df['price_up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Replace inf and NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        
        # Drop rows with NaN due to moving windows
        df = df.iloc[50:]
        
        self.data = df
    
    def feature_engineering(self):
        """Feature selection and engineering"""
        print("Performing feature engineering...")
        
        # Calculate feature correlations with target
        correlation_with_target = self.data.corrwith(self.data[self.target_column].shift(-1)).abs().sort_values(ascending=False)
        print(f"Top 10 features correlated with target:\n{correlation_with_target.head(10)}")
        
        # Extend feature columns with engineered features
        engineered_features = [
            'MA5', 'MA20', 'EMA12', 'MACD', 'RSI', 'BB_width', 
            'momentum_1d', 'momentum_5d', '%K', 'ATR', 'CCI'
        ]
        
        # Add sentiment features if present
        if 'avg_sentiment' in self.data.columns:
            engineered_features.extend(['avg_sentiment', 'sentiment_ma5', 'sentiment_change'])
        
        # Update feature columns
        self.feature_columns = self.feature_columns + engineered_features
        print(f"Selected {len(self.feature_columns)} features: {', '.join(self.feature_columns)}")
    
    def prepare_sequences(self):
        """Prepare training sequences for time series prediction with GPU optimization"""
        print("Preparing sequences optimized for GPU...")
        
        # Split into train, validation, and test sets
        n = len(self.data)
        train_end = int(n * TRAIN_SPLIT)
        val_end = train_end + int(n * VAL_SPLIT)
        
        train_data = self.data.iloc[:train_end]
        val_data = self.data.iloc[train_end:val_end]
        test_data = self.data.iloc[val_end:]
        
        print(f"Train data: {len(train_data)} rows ({TRAIN_SPLIT*100:.0f}%)")
        print(f"Validation data: {len(val_data)} rows ({VAL_SPLIT*100:.0f}%)")
        print(f"Test data: {len(test_data)} rows ({TEST_SPLIT*100:.0f}%)")
        
        # Scale features
        for feature in self.feature_columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_data[feature] = scaler.fit_transform(train_data[[feature]])
            val_data[feature] = scaler.transform(val_data[[feature]])
            test_data[feature] = scaler.transform(test_data[[feature]])
            self.scalers[feature] = scaler
        
        # Scale target
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        train_data[[self.target_column]] = target_scaler.fit_transform(train_data[[self.target_column]])
        val_data[[self.target_column]] = target_scaler.transform(val_data[[self.target_column]])
        test_data[[self.target_column]] = target_scaler.transform(test_data[[self.target_column]])
        self.scalers['target'] = target_scaler
        
        # Create sequences
        self.X_train, self.y_train = self._create_sequences(train_data)
        self.X_val, self.y_val = self._create_sequences(val_data)
        self.X_test, self.y_test = self._create_sequences(test_data)
        
        # Convert to float16 for mixed precision training
        self.X_train = tf.cast(self.X_train, tf.float16)
        self.X_val = tf.cast(self.X_val, tf.float16)
        self.X_test = tf.cast(self.X_test, tf.float16)
        
        # Create optimized TensorFlow datasets for GPU
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        self.train_dataset = self.train_dataset.shuffle(10000, seed=RANDOM_SEED)
        self.train_dataset = self.train_dataset.batch(BATCH_SIZE)
        self.train_dataset = self.train_dataset.prefetch(tf.data.AUTOTUNE)
        
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        self.val_dataset = self.val_dataset.batch(BATCH_SIZE)
        self.val_dataset = self.val_dataset.prefetch(tf.data.AUTOTUNE)
        
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        self.test_dataset = self.test_dataset.batch(BATCH_SIZE)
        self.test_dataset = self.test_dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"Training sequences: {self.X_train.shape}")
        print(f"Validation sequences: {self.X_val.shape}")
        print(f"Test sequences: {self.X_test.shape}")
        print("Created GPU-optimized TensorFlow datasets")
    
    def _create_sequences(self, data):
        """Helper function to create sequences for LSTM"""
        X, y = [], []
        
        # Get feature and target data
        feature_data = data[self.feature_columns].values
        target_data = data[self.target_column].values
        
        # Create sequences
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            X.append(feature_data[i:i+self.lookback])
            y.append(target_data[i+self.lookback:i+self.lookback+self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def build_hybrid_model(self):
        """Build an extremely computationally intensive neural network model optimized for GPU"""
        print("Building ultra-heavy neural network model for GPU...")
        
        # Force model to be built on GPU
        with tf.device('/GPU:0'):
            # Determine input shape
            input_shape = (self.lookback, len(self.feature_columns))
            
            # Input layer
            inputs = Input(shape=input_shape)
            
            # 1. Multi-level Convolutional path for pattern extraction
            # First conv block
            conv1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(inputs)
            conv1 = BatchNormalization()(conv1)
            conv2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(conv1)
            conv2 = BatchNormalization()(conv2)
            conv3 = Conv1D(filters=128, kernel_size=7, padding='same', activation='relu')(conv2)
            conv3 = BatchNormalization()(conv3)
            max_pool1 = MaxPooling1D(pool_size=2)(conv3)
            
            # Second conv block
            conv4 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(max_pool1)
            conv4 = BatchNormalization()(conv4)
            conv5 = Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')(conv4)
            conv5 = BatchNormalization()(conv5)
            conv6 = Conv1D(filters=256, kernel_size=7, padding='same', activation='relu')(conv5)
            conv6 = BatchNormalization()(conv6)
            max_pool2 = MaxPooling1D(pool_size=2)(conv6)
            
            # 2. Multi-layer Bidirectional LSTM path with residual connections
            # First LSTM block
            lstm1 = Bidirectional(LSTM(200, return_sequences=True))(inputs)
            lstm1 = BatchNormalization()(lstm1)
            lstm1 = Dropout(0.3)(lstm1)
            
            lstm2 = Bidirectional(LSTM(200, return_sequences=True))(lstm1)
            lstm2 = BatchNormalization()(lstm2)
            lstm2 = Dropout(0.3)(lstm2)
            
            # Residual connection for first block
            lstm_res1 = Conv1D(filters=400, kernel_size=1, padding='same')(inputs)  # Match dimensions
            lstm2 = Add()([lstm2, lstm_res1])
            
            # Second LSTM block
            lstm3 = Bidirectional(LSTM(300, return_sequences=True))(lstm2)
            lstm3 = BatchNormalization()(lstm3)
            lstm3 = Dropout(0.3)(lstm3)
            
            lstm4 = Bidirectional(LSTM(300, return_sequences=True))(lstm3)
            lstm4 = BatchNormalization()(lstm4)
            lstm4 = Dropout(0.3)(lstm4)
            
            # Residual connection for second block
            lstm_res2 = Conv1D(filters=600, kernel_size=1, padding='same')(lstm2)  # Match dimensions
            lstm4 = Add()([lstm4, lstm_res2])
            
            # 3. Deep Transformer path with multi-head attention
            transformer = inputs
            
            # Create 6 transformer blocks for extreme depth
            for i in range(6):  # 6 transformer blocks
                # Multi-head self-attention (more heads = more computation)
                attention_output = MultiHeadAttention(
                    num_heads=16, key_dim=64, dropout=0.1
                )(transformer, transformer, transformer)
                
                # Skip connection and normalization
                transformer = Add()([transformer, attention_output])
                transformer = LayerNormalization(epsilon=1e-6)(transformer)
                
                # Wider feed-forward network
                ffn = Dense(512, activation='relu')(transformer)
                ffn = Dropout(0.1)(ffn)
                ffn = Dense(input_shape[1])(ffn)
                
                # Skip connection and normalization
                transformer = Add()([transformer, ffn])
                transformer = LayerNormalization(epsilon=1e-6)(transformer)
            
            # 4. Cross-attention between CNN and LSTM paths
            cross_attention = MultiHeadAttention(
                num_heads=8, key_dim=64, dropout=0.1
            )(lstm4, max_pool2, max_pool2)
            
            cross_attention = BatchNormalization()(cross_attention)
            
            # Combine all paths - first match sequence lengths and dimensions
            max_pool_flat = Flatten()(max_pool2)
            max_pool_expanded = RepeatVector(lstm4.shape[1])(max_pool_flat)
            
            # Concatenate the paths
            combined = Concatenate()([lstm4, transformer, max_pool_expanded, cross_attention])
            
            # 5. Multiple attention mechanisms
            # Self attention
            self_attention = Dense(1, activation='tanh')(combined)
            self_attention = Flatten()(self_attention)
            self_attention = Lambda(lambda x: tf.transpose(x, [0, 2, 1]))(self_attention)
            
            # Apply attention weights using explicit Lambda layer
            context = Lambda(lambda x: x[0] * x[1])([combined, self_attention])
            context = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
            
            # 6. Deep fully connected layers with residual connections
            # First dense block with residual
            dense1 = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(context)
            dense1 = BatchNormalization()(dense1)
            dense1 = Dropout(0.5)(dense1)
            
            dense2 = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense1)
            dense2 = BatchNormalization()(dense2)
            dense2 = Dropout(0.5)(dense2)
            
            # Residual connection
            dense2_res = Add()([dense1, dense2])
            
            # Second dense block with residual
            dense3 = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense2_res)
            dense3 = BatchNormalization()(dense3)
            dense3 = Dropout(0.4)(dense3)
            
            dense4 = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense3)
            dense4 = BatchNormalization()(dense4)
            dense4 = Dropout(0.4)(dense4)
            
            # Residual connection
            dense4_res = Add()([dense3, dense4])
            
            # Third dense block
            dense5 = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense4_res)
            dense5 = BatchNormalization()(dense5)
            dense5 = Dropout(0.3)(dense5)
            
            dense6 = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense5)
            dense6 = BatchNormalization()(dense6)
            dense6 = Dropout(0.3)(dense6)
            
            # Output layer
            outputs = Dense(self.forecast_horizon)(dense6)
            
            # Build and compile model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile with custom metrics
            model.compile(
                optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for better convergence
                loss='mean_squared_error',
                metrics=[
                    custom_root_mean_squared_error,
                    custom_mean_absolute_percentage_error
                ]
            )
            
            # Print model summary
            model.summary()
            
            self.model = model
            return model
    
    def train_model(self):
        """Train the model with early stopping and learning rate reduction"""
        print("Training model on GPU...")
        
        # Verify GPU is being used
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError("No GPU available for training!")
        
        # Create a GPU memory monitoring callback
        class GPUMemoryMonitor(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                try:
                    import subprocess
                    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,nounits,noheader'])
                    for i, line in enumerate(result.decode('utf-8').strip().split('\n')):
                        memory_used, memory_total, gpu_util = map(int, line.split(','))
                        print(f"\nGPU {i}: Memory {memory_used}/{memory_total} MB ({memory_used/memory_total*100:.1f}%), Utilization {gpu_util}%")
                except:
                    pass
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=20,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(MODEL_PATH, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(RESULTS_PATH, 'logs'),
                histogram_freq=1,
                profile_batch='500,520'
            ),
            GPUMemoryMonitor()
        ]
        
        # Use strategy for GPU training
        strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
        with strategy.scope():
            # Train model with optimized settings for TF nightly
            history = self.model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
        
        self.history = history.history
        
        # Save history
        with open(os.path.join(RESULTS_PATH, 'training_history.pkl'), 'wb') as f:
            pickle.dump(self.history, f)
        
        # Plot training history
        self.plot_training_history()
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot custom metric (RMSE)
        plt.subplot(1, 2, 2)
        plt.plot(self.history['custom_root_mean_squared_error'], label='Training RMSE')
        plt.plot(self.history['val_custom_root_mean_squared_error'], label='Validation RMSE')
        plt.title('Root Mean Squared Error')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'training_history.png'))
        plt.close()
    
    def predict_and_evaluate(self):
        """Make predictions and evaluate the model"""
        print("Making predictions and evaluating model...")
        
        # Predict on test data using the optimized dataset
        predictions = self.model.predict(self.test_dataset)
        
        # Inverse transform predictions and actual values
        target_scaler = self.scalers['target']
        
        # Reshape predictions and actual values for inverse transform
        pred_reshaped = predictions.reshape(-1, 1)
        actual_reshaped = self.y_test.reshape(-1, 1)
        
        # Inverse transform
        pred_inv = target_scaler.inverse_transform(pred_reshaped)
        actual_inv = target_scaler.inverse_transform(actual_reshaped)
        
        # Reshape back
        pred_inv = pred_inv.reshape(predictions.shape)
        actual_inv = actual_inv.reshape(self.y_test.shape)
        
        # Store predictions
        self.predictions = {
            'actual': actual_inv,
            'predicted': pred_inv,
            'dates': self.data.index[self.lookback + len(self.X_train) + len(self.X_val):].values[:len(pred_inv)]
        }
        
        # Calculate metrics
        mse = mean_squared_error(actual_inv, pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_inv, pred_inv)
        r2 = r2_score(actual_inv, pred_inv)
        
        mape = np.mean(np.abs((actual_inv - pred_inv) / actual_inv)) * 100
        
        # Calculate directional accuracy
        direction_actual = np.sign(np.diff(actual_inv.flatten()))
        direction_pred = np.sign(np.diff(pred_inv.flatten()))
        directional_acc = np.mean(direction_actual == direction_pred)
        
        # Store metrics
        self.evaluation_metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R^2': r2,
            'Directional Accuracy': directional_acc
        }
        
        # Print metrics
        print("Model Evaluation:")
        for metric, value in self.evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save metrics
        with open(os.path.join(RESULTS_PATH, 'evaluation_metrics.json'), 'w') as f:
            json.dump(self.evaluation_metrics, f, indent=4)
        
        # Plot predictions
        self.plot_predictions()
    
    def plot_predictions(self):
        """Plot predictions vs actual values"""
        print("Plotting predictions...")
        
        actual = self.predictions['actual'].flatten()
        predicted = self.predictions['predicted'].flatten()
        dates = pd.to_datetime(self.predictions['dates'])
        
        plt.figure(figsize=(14, 7))
        
        # Plot the predictions and actual values
        plt.subplot(2, 1, 1)
        plt.plot(dates, actual, label='Actual', color='blue')
        plt.plot(dates, predicted, label='Predicted', color='red', linestyle='--')
        plt.title('Stock Price Prediction (Actual vs Predicted)')
        plt.xlabel('Date')
        plt.ylabel(f'{self.target_column} Price')
        plt.legend()
        plt.grid(True)
        
        # Plot the error
        plt.subplot(2, 1, 2)
        error = predicted - actual
        plt.plot(dates, error, color='green')
        plt.axhline(y=0, color='red', linestyle='-')
        plt.title('Prediction Error')
        plt.xlabel('Date')
        plt.ylabel('Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'predictions.png'))
        plt.close()
        
        # Create scatter plot of predicted vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_PATH, 'scatter_plot.png'))
        plt.close()
    
    def run_pipeline(self):
        """Run the complete machine learning pipeline"""
        # 1. Load and preprocess data
        self.load_data()
        
        # 2. Prepare sequences
        self.prepare_sequences()
        
        # 3. Build model
        self.build_hybrid_model()
        
        # 4. Train model
        self.train_model()
        
        # 5. Predict and evaluate
        self.predict_and_evaluate()
        
        print("Pipeline complete!")

# Create trading strategy simulation
def simulate_trading_strategy(predictions, initial_capital=10000, transaction_cost=0.001):
    """Simulate a simple trading strategy based on model predictions"""
    print("Simulating trading strategy...")
    
    actual = predictions['actual'].flatten()
    predicted = predictions['predicted'].flatten()
    dates = pd.to_datetime(predictions['dates'])
    
    # Create a DataFrame for the simulation
    df = pd.DataFrame({
        'Date': dates[:-1],  # One less due to calculating daily change
        'Actual': actual[:-1],
        'Predicted': predicted[:-1],
        'Next_Actual': actual[1:],
        'Predicted_Change': np.diff(predicted),
        'Actual_Change': np.diff(actual)
    })
    
    # Strategy: Buy if predicted change is positive, sell if negative
    df['Position'] = np.where(df['Predicted_Change'] > 0, 1, -1)
    
    # Calculate returns
    df['Market_Return'] = df['Actual_Change'] / df['Actual']
    df['Strategy_Return'] = df['Position'] * df['Market_Return']
    
    # Apply transaction costs when position changes
    df['Position_Change'] = df['Position'].diff().fillna(0)
    df.loc[df['Position_Change'] != 0, 'Strategy_Return'] -= transaction_cost
    
    # Calculate cumulative returns
    df['Cumulative_Market_Return'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    # Calculate portfolio values
    df['Market_Portfolio'] = initial_capital * df['Cumulative_Market_Return']
    df['Strategy_Portfolio'] = initial_capital * df['Cumulative_Strategy_Return']
    
    # Calculate metrics
    total_trades = (df['Position_Change'] != 0).sum()
    profitable_trades = ((df['Position'] * df['Actual_Change']) > 0).sum()
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    final_market_value = df['Market_Portfolio'].iloc[-1]
    final_strategy_value = df['Strategy_Portfolio'].iloc[-1]
    
    market_return_pct = (final_market_value / initial_capital - 1) * 100
    strategy_return_pct = (final_strategy_value / initial_capital - 1) * 100
    
    # Annualized returns (assuming 252 trading days)
    n_days = len(df)
    market_annual_return = ((final_market_value / initial_capital) ** (252 / n_days) - 1) * 100
    strategy_annual_return = ((final_strategy_value / initial_capital) ** (252 / n_days) - 1) * 100
    
    # Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)
    strategy_daily_returns = df['Strategy_Return']
    sharpe_ratio = np.sqrt(252) * strategy_daily_returns.mean() / strategy_daily_returns.std()
    
    # Maximum drawdown
    cumulative_peaks = df['Strategy_Portfolio'].cummax()
    drawdowns = (df['Strategy_Portfolio'] - cumulative_peaks) / cumulative_peaks
    max_drawdown = drawdowns.min() * 100
    
    # Print results
    print("\nTrading Strategy Results:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Market Value: ${final_market_value:.2f} ({market_return_pct:.2f}%)")
    print(f"Final Strategy Value: ${final_strategy_value:.2f} ({strategy_return_pct:.2f}%)")
    print(f"Market Annual Return: {market_annual_return:.2f}%")
    print(f"Strategy Annual Return: {strategy_annual_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Plot performance
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Market_Portfolio'], label='Buy & Hold', color='blue')
    plt.plot(df['Date'], df['Strategy_Portfolio'], label='Model Strategy', color='green')
    plt.title('Trading Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_PATH, 'trading_strategy.png'))
    plt.close()
    
    # Save trading data
    df.to_csv(os.path.join(RESULTS_PATH, 'trading_simulation.csv'), index=False)
    
    # Return key metrics
    return {
        'win_rate': win_rate,
        'total_trades': total_trades,
        'market_return': market_return_pct,
        'strategy_return': strategy_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

if __name__ == "__main__":
    # Clear any previous session
    tf.keras.backend.clear_session()
    
    print("*** Stock Price Prediction with Advanced Neural Networks on GPU ***")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Verify GPU is available
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    if not gpu_available:
        print("ERROR: No GPU detected. This model requires GPU acceleration.")
        exit(1)
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file '{DATA_PATH}' not found.")
        print("Please run preprocessing first to generate the sentiment data.")
        exit(1)
    
    # Run the prediction pipeline
    try:
        predictor = StockPredictor()
        predictor.run_pipeline()
        
        # Simulate trading strategy
        trading_metrics = simulate_trading_strategy(predictor.predictions)
        
        # Save all metrics
        with open(os.path.join(RESULTS_PATH, 'trading_metrics.json'), 'w') as f:
            json.dump(trading_metrics, f, indent=4)
        
        print("\nAll results saved to the 'results' directory.")
        print("Script execution complete!")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc() 