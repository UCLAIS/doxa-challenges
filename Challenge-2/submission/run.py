import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np


def main():
    # load TF model
    model = tf.keras.models.load_model('model')

    print("READY")
    # read and load a test set from stdin
    test_set_path = input()
    x_test = np.load(test_set_path)
    x_test = x_test['data']


    # run model on the test set
    y_preds = model.predict(x_test, verbose=0)
    y_pred_classes = tf.math.argmax(y_preds, axis=1)
    
    # write predicted classes to stdout
    for pred in y_pred_classes:
        print(int(pred))
    
if __name__ == "__main__":
    main()
