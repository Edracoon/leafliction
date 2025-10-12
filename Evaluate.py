import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import utils
from keras.models import load_model, Sequential


MODEL_PATH = 'model/model.keras'
TEST_DATASET_PATH = './images/'


def evaluate():
    """
    Load a trained model and evaluate
    its performance on a test dataset.
    """
    model: Sequential = load_model(MODEL_PATH)

    test_generator = utils.image_dataset_from_directory(
        TEST_DATASET_PATH,
        image_size=(128, 128),
        batch_size=32,
        label_mode='categorical',
        shuffle=False
    )
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Accuracy : {accuracy}  |  Loss : {loss}")


def main():
    evaluate()


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print(f"evaluate.py: error: {e}")