import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf


def main():
    # load the Tensorflow model
    model = tf.keras.models.load_model("model")

    # synchronise with DOXA
    print("READY")

    # read the test set file path from stdin
    test_set_path = input()

    # load the test set
    x_test = np.load(test_set_path)["data"]

    # run the model on the test set
    predictions = tf.math.argmax(model.predict(x_test, verbose=0), axis=1)

    # write the class predictions to stdout
    for prediction in predictions:
        print(int(prediction))


if __name__ == "__main__":
    main()
