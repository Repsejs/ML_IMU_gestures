import tensorflow as tf
import subprocess
import os

def convert_h5_to_tflite(h5_model_path, tflite_model_path):
    # Load the Keras model
    model = tf.keras.models.load_model(h5_model_path)
    
    # Convert the Keras model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the TensorFlow Lite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

def convert_tflite_to_tmdl(tflite_model_path, tmdl_model_path):
    # Command to convert TFLite model to TMdl using the TinyMaix tool
    command = [
        'tinymaix/tm_tools/tflite2tmdl',  # Path to the tflite2tmdl tool
        tflite_model_path,
        tmdl_model_path
    ]
    subprocess.run(command, check=True)

def convert_tmdl_to_c_header(tmdl_model_path, header_file_path):
    # Command to convert TMdl model to C header file using TinyMaix tool
    command = [
        'tinymaix/tm_tools/tmdl2c',  # Path to the tmdl2c tool
        tmdl_model_path,
        header_file_path
    ]
    subprocess.run(command, check=True)

def main():
    h5_model_path = 'h5/mnist_valid.h5'
    tflite_model_path = 'tflite/mnist_valid.tflite'
    tmdl_model_path = 'tmdl/mnist_valid.tmdl'
    header_file_path = 'tmdl/mnist_valid.h'
    
    # Convert H5 model to TFLite format
    convert_h5_to_tflite(h5_model_path, tflite_model_path)
    
    # Convert TFLite model to TMdl format
    convert_tflite_to_tmdl(tflite_model_path, tmdl_model_path)
    
    # Convert TMdl model to C header file
    convert_tmdl_to_c_header(tmdl_model_path, header_file_path)
    
    print(f"Model successfully converted to C header file: {header_file_path}")

if __name__ == '__main__':
    main()
