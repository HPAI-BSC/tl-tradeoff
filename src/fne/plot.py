import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


class Plot:
    def __init__(self):
        pass

    @staticmethod
    def plot_learning_curves(history):
        # Accuracy plot
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.title('Training and validation accuracy')
        plt.savefig('../outputs/fine_tuning_accuracy.pdf')
        plt.close()

        # Loss plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.title('Training and validation loss')
        plt.savefig('../outputs/fine_tuning_loss.pdf')

        return
