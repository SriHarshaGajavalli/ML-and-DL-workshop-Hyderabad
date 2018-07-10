import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from time import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


class KNearestNeighbor(object):


    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):

        """Predict labels for test data using this classifier."""
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):

        """Compute the distance between each test point in X and each training point."""

        now = time()
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        train_square = np.sum((self.X_train ** 2).T, axis=0)  # a^2
        test_square = np.sum(X ** 2, axis=1)  # b^2
        mid_val = 2 * np.dot(X, self.X_train.T)  # -2ab
        dists = np.sqrt(((-1 * mid_val.T) + test_square).T + train_square)
        print("Time taken to compute the distances")

        print(time() - now)
        return dists


    def predict_labels(self, dists, k=1):

        """Predict labels for samples."""

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = [self.y_train[x] for x in np.argpartition(dists[i, :], k)[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred




def findFeatureScoreList(kmeans,images):

    featureScoreList = []
    for image in images:
        featureScore = np.zeros(CLUSTER_SIZE, dtype=int)
        predictedClusterList = kmeans.predict(image)
        for predictedCluster in predictedClusterList:
            featureScore[predictedCluster] += 1
        featureScoreList.append(featureScore)
    return featureScoreList


def readData(TRAIN_FOLDERS, TEST_FOLDERS, NO_OF_TRAIN, NO_OF_TEST):
    trainImages = []
    images = []
    imageno = 0
    testImages = []
    testImageCount = 0
    print("Reading Data")
    for fileSuffix in range(1, NO_OF_TEST + 1):
        filename = TEST_FOLDERS + str(fileSuffix) + "_test_sift.csv"
        filedata = pd.read_csv(filename, index_col=None, header=None).values.tolist()
        testImages.append([])
        for row in filedata:
            siftFeature = row[4:]
            testImages[testImageCount].append(siftFeature)
        testImageCount += 1
    # print("Reading TRAIN images")


    for fileSuffix in range(1, NO_OF_TRAIN + 1):
        filename = TRAIN_FOLDERS + str(fileSuffix) + "_train_sift.csv"

        filedata = pd.read_csv(filename, index_col=None, header=None).values.tolist()
        images.append([])
        for row in filedata:
            siftFeature = row[4:]
            trainImages.append(siftFeature)
            images[imageno].append(siftFeature)
        imageno += 1
    return trainImages, images, testImages


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    CLUSTER_SIZE = 50
    TEST_FOLDERS = "/home/donglee/Documents/Computer-Vision/Assignment2/data/test_sift_features/"
    TRAIN_FOLDERS = "/home/donglee/Documents/Computer-Vision/Assignment2/data/train_sift_features/"
    NO_OF_TEST = 800
    NO_OF_TRAIN = 1888
    y_train = \
        pd.read_csv("/home/donglee/Documents/Computer-Vision/Assignment2/data/train_labels.csv", index_col=None,
                    header=None).values.tolist()[0]
    y_test = pd.read_csv("/home/donglee/Documents/Computer-Vision/Assignment2/data/test_labels.csv", index_col=None,
                         header=None).values.tolist()[0]

    X_Train, TrainImages, X_Test = readData(TRAIN_FOLDERS, TEST_FOLDERS, NO_OF_TRAIN, NO_OF_TEST)
    print("Initiated Kmeans Clustering")
    now = time()
    kmeans = KMeans(n_clusters=CLUSTER_SIZE, random_state=0, batch_size=256).fit(X_Train)
    print("Yay, Kmeans Clustering done successfully")
    print("Time Taken for K-Means Clustering: " + str(time() - now) + " sec")
    featureScoreList = findFeatureScoreList(kmeans, TrainImages)

    print("Classification with KNN:")
    now = time()
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(featureScoreList, y_train)
    featureScoreListTest = findFeatureScoreList(kmeans, X_Test)
    predictions = neigh.predict(featureScoreListTest)
    accuracy = accuracy_score(predictions, y_test)
    print("Accuracy of KNN: " + str(accuracy * 100) + " percent")
    print("Time Taken for KNN algorithm: " + str(time() - now) + " sec")
    cnf_matrix = confusion_matrix(y_test, predictions)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=y_test,
                          title='Confusion matrix, without normalization')
    print("Classifying with SVM:")
    now = time()
    clf = LinearSVC(random_state=0)
    clf.fit(featureScoreList, y_train)
    pred = clf.predict(featureScoreListTest)
    print("Accuracy using SVM:")
    print(accuracy_score(pred, y_test) * 100)
    print("Time Taken for SVM algorithm: " + str(time() - now) + " sec")
    cnf_matrix = confusion_matrix(y_test, pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=y_test,
                          title='Confusion matrix, without normalization')