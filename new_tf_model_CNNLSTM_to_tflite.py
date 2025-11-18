import logging
import math
import os

import tensorflow as tf
# -------------------------------------------------------------------
# IMPORT ALL NECESSARY LAYERS AND MODELS
# -------------------------------------------------------------------
from tensorflow.keras.layers import (LSTM, BatchNormalization, Conv1D, Dense,
                                     Dropout, GlobalAveragePooling1D, Input,
                                     MaxPooling1D, ReLU)
from tensorflow.keras.models import Model, Sequential

# Configure logging
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------
# MODEL DEFINITION (REFACTORED TO FUNCTIONAL API)
# -------------------------------------------------------------------

def create_functional_model(
    input_channels,
    seq_len,
    conv_filters,
    conv_kernel_size,
    pool_size,
    lstm_units,
    dense_units,
    dropout_rate=0.5
):
    """Creates the CNN-LSTM model using the Keras Functional API."""

    # --- Input Validation ---
    if input_channels <= 0:
        input_channels = 1
    if seq_len <= 0:
        seq_len = 1
    if not conv_filters:
        conv_filters = [32]

    kernel_size = max(1, conv_kernel_size)
    current_pool_size = max(1, pool_size)

    # --- Shape Validation ---
    try:
        final_seq_len = seq_len
        for _ in conv_filters:
            final_seq_len = math.floor(final_seq_len / current_pool_size)
        
        if final_seq_len <= 0:
            raise ValueError(
                f"Calculated LSTM input sequence length is zero or negative ({final_seq_len}). "
                f"Check CNN/Pooling parameters relative to segment length ({seq_len})."
            )
    except Exception as e:
        logging.error(
            f"Error calculating layer output size during model init: {e}"
        )
        raise e

    # --- Define Model Architecture ---
    
    # 1. Input Layer
    # Keras format: (batch_size, seq_len, channels)
    inputs = Input(shape=(seq_len, input_channels), name="input_layer")

    # 2. CNN Block
    x = inputs
    for i, out_channels in enumerate(conv_filters):
        x = Conv1D(
            filters=out_channels,
            kernel_size=kernel_size,
            padding='same',
            name=f"conv1d_{i}"
        )(x)
        x = BatchNormalization(name=f"batchnorm_{i}")(x)
        x = ReLU(name=f"relu_{i}")(x)
        x = MaxPooling1D(pool_size=current_pool_size, name=f"maxpool_{i}")(x)
        x = Dropout(dropout_rate, name=f"dropout_{i}")(x)

    # 3. LSTM Block
    # Output is (batch, new_seq_len, features), which is perfect for LSTM
    x = LSTM(
        units=lstm_units,
        return_sequences=True,  # Return all time steps
        name="lstm"
    )(x)
    x = Dropout(dropout_rate, name="lstm_dropout")(x)

    # 4. Aggregation
    # This replaces torch.mean(dim=1)
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)
    
    # 5. Dense Block
    x = Dense(dense_units, activation='relu', name="dense_1")(x)
    outputs = Dense(1, activation='sigmoid', name="output_sigmoid")(x)

    # Create and return the model
    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_TF_Functional")
    return model

# -------------------------------------------------------------------
# CONVERSION SCRIPT
# -------------------------------------------------------------------

# --- 1. Define Model Parameters ---
MODEL_INPUT_CHANNELS = 4
MODEL_SEQ_LEN = 30
MODEL_CONV_FILTERS = [128, 256]
MODEL_CONV_KERNEL_SIZE = 5
MODEL_POOL_SIZE = 2
MODEL_LSTM_UNITS = 128
MODEL_DENSE_UNITS = 128
MODEL_DROPOUT_RATE = 0.0

SAVED_MODEL_DIR = "saved_temp_tf_model"
TFLITE_MODEL_PATH = "cnn_lstm_model.tflite"

def convert_to_tflite():
    print("--- 1. Initializing TensorFlow Model ---")
    try:
        model = create_functional_model(
            input_channels=MODEL_INPUT_CHANNELS,
            seq_len=MODEL_SEQ_LEN,
            conv_filters=MODEL_CONV_FILTERS,
            conv_kernel_size=MODEL_CONV_KERNEL_SIZE,
            pool_size=MODEL_POOL_SIZE,
            lstm_units=MODEL_LSTM_UNITS,
            dense_units=MODEL_DENSE_UNITS,
            dropout_rate=MODEL_DROPOUT_RATE,
        )
        
        model.summary()

        print("\n--- 2. Saving as SavedModel format ---")
        # Use the simpler Keras save method. This is more robust.
        model.export(SAVED_MODEL_DIR)
        print(f"Model saved successfully to {SAVED_MODEL_DIR}")

        # --- 3. Convert to TFLite ---
        print(f"\n--- 3. Converting SavedModel to TFLite ---")
        
        # Point the converter to the saved model directory
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
        
        # # Enable necessary ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable default TFLite ops.
            tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops.
        ]
        
        # (Optional) Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        print("TFLite conversion successful.")

        # --- 4. Save the TFLite Model ---
        print(f"\n--- 4. Saving TFLite model to {TFLITE_MODEL_PATH} ---")
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"Successfully saved TFLite model to {TFLITE_MODEL_PATH}")
        print(f"File size: {os.path.getsize(TFLITE_MODEL_PATH) / 1024:.2f} KB")

    except ValueError as e:
        logging.error(f"Error during model initialization: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during conversion: {e}", exc_info=True)

if __name__ == "__main__":
    convert_to_tflite()