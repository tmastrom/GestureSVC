# Credits to https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
Initial REsult
confusion_matrix = [[639,  15,  32,  20],
 [  3, 697,   5,   4],
 [  6,  22, 640,  78],
 [ 38,  75,   6, 640]]
 '''

# db1 feature vector, no optimization
confusion_matrix = [[638,  14,  34,  20],
 [  2, 697,   5,   5],
 [  6 , 23, 638 , 79],
 [ 37,  64,   9 ,649]]



class_names = ['Rock', 'Scissors', 'Paper', 'Ok']




def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    


print_confusion_matrix(confusion_matrix, class_names);

