from matplotlib import pyplot as plt
import matplotlib.animation as anim
import numpy as np

def create_trace_animation(predictions: np.ndarray, ground_truths: np.ndarray):
    """
    Create a 3d-animation containing the ground truth trace and prediction trace of the lowest cable point.
    """

    # y is given as the height dimension, plt expecting z to be the height. swapping the axis 
    predictions.T[[1, 2]] = predictions.T[[2, 1]]
    ground_truths.T[[1, 2]] = ground_truths.T[[2, 1]]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    prediction_line = ax.plot([], [], [], label="predictions")[0]
    ground_truth_line = ax.plot([], [], [], label="ground truth")[0]
    lines = [prediction_line, ground_truth_line]
    
    min_pred, max_pred = predictions.min(axis=0), predictions.max(axis=0)
    min_true, max_true = ground_truths.min(axis=0), ground_truths.max(axis=0)
    min_x, min_y, min_z = min(min_pred[0], min_true[0]), min(min_pred[1], min_true[1]), min(min_pred[2], min_true[2])
    max_x, max_y, max_z = max(max_pred[0], max_true[0]), max(max_pred[1], max_true[1]), max(max_pred[2], max_true[2])
    ax.set(xlim3d=(min_x, max_x), xlabel='X')
    ax.set(ylim3d=(min_y, max_y), ylabel='Z')
    ax.set(zlim3d=(min_z, max_z), zlabel='Y')
    
    animation = anim.FuncAnimation(
        fig, update_lines, predictions.shape[0], fargs=(predictions, ground_truths, lines), interval=300
    )
    ax.legend()
    return animation


def update_lines(num, predictions, ground_truths, lines):
    for line, function in zip(lines, [predictions, ground_truths]):
        line.set_data(function[:num, :2].T)
        line.set_3d_properties(function[:num, 2])
    return lines


def create_3d_point_trace(data: np.ndarray):

    # y is given as the height dimension, plt expecting z to be the height. swapping the axis 
    x = data[:, 0]
    y = data[:, 2]
    z = data[:, 1]

    ax = plt.figure().add_subplot(projection='3d')

    min_x, min_y, min_z = min(x), min(y), min(z)
    max_x, max_y, max_z = max(x), max(y), max(z)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])

    ax.plot(x, y, z, c='red', label='trace')
    ax.legend()
    return plt