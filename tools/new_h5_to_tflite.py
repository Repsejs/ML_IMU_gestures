import tensorflow as tf
import numpy as np
import csv
import sys

def parse_csv(file_path, sequence_length=100):
    data = {}
    current_label = None
    current_data = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        
        for row in reader:
            if row == ['X-acc', 'Y-acc', 'Z-acc', 'label']:
                if current_label is not None and current_data:
                    # Ensure data length matches the expected sequence length
                    if len(current_data) < sequence_length:
                        # Pad with zeros if shorter
                        padding = sequence_length - len(current_data)
                        current_data.extend([[0, 0, 0]] * padding)
                    elif len(current_data) > sequence_length:
                        # Truncate if longer
                        current_data = current_data[:sequence_length]
                    
                    if current_label in data:
                        data[current_label].append(np.array(current_data))
                    else:
                        data[current_label] = [np.array(current_data)]
                    current_data = []
                continue

            x_acc, y_acc, z_acc, label = row
            x_acc, y_acc, z_acc = float(x_acc), float(y_acc), float(z_acc)

            if current_label is None:
                current_label = label

            if label != current_label:
                if current_data:
                    # Ensure data length matches the expected sequence length
                    if len(current_data) < sequence_length:
                        # Pad with zeros if shorter
                        padding = sequence_length - len(current_data)
                        current_data.extend([[0, 0, 0]] * padding)
                    elif len(current_data) > sequence_length:
                        # Truncate if longer
                        current_data = current_data[:sequence_length]
                    
                    if current_label in data:
                        data[current_label].append(np.array(current_data))
                    else:
                        data[current_label] = [np.array(current_data)]
                    current_data = []

                current_label = label

            current_data.append([x_acc, y_acc, z_acc])

        # Append the last set of data
        if current_label is not None and current_data:
            # Ensure data length matches the expected sequence length
            if len(current_data) < sequence_length:
                # Pad with zeros if shorter
                padding = sequence_length - len(current_data)
                current_data.extend([[0, 0, 0]] * padding)
            elif len(current_data) > sequence_length:
                # Truncate if longer
                current_data = current_data[:sequence_length]
            
            if current_label in data:
                data[current_label].append(np.array(current_data))
            else:
                data[current_label] = [np.array(current_data)]

    return data

def representative_dataset_gen(data, sequence_length=100):
    for label, sequences in data.items():
        for seq in sequences:
            yield [np.array(seq, dtype=np.float32).reshape(1, sequence_length, 3)]

def h5_to_tflite(h5_name, tflite_name, is_quant=0, csv_file=None):
    # Load the Keras model
    model = tf.keras.models.load_model(h5_name)

    # Set up the converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if is_quant:
        # Parse the CSV to get representative data
        data = parse_csv(csv_file)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset_gen(data)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.int8
        converter.inference_output_type = tf.int8  # or tf.int8

    # Convert the model to TFLite
    tflite_model = converter.convert()

    # Save the converted model to a file
    with open(tflite_name, 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <input_h5> <output_tflite> <is_quant> <csv_file>")
        sys.exit(1)

    h5_name = sys.argv[1]
    tflite_name = sys.argv[2]
    is_quant = int(sys.argv[3])
    csv_file = sys.argv[4]

    h5_to_tflite(h5_name, tflite_name, is_quant, csv_file)
