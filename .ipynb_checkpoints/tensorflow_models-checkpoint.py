def basic_cnn_binary(period):
        
    inputs = tf.keras.Input(shape=(period,1,))
    
    # x = layers.Conv1D(16, kernel_size=5, strides=3, activation='relu')(inputs)
    x = layers.Conv1D(8, kernel_size=3, strides=1, activation='relu')(inputs)
    x = layers.Conv1D(1, kernel_size=1, strides=1, activation='relu')(x)
    
    # x = layers.LSTM(16, return_sequences=True, return_state=True)(x)
    x = layers.LSTM(16, return_sequences=False, return_state=False)(x)
    
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.Dense(8)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs) 
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.binary_accuracy])
    
    return model