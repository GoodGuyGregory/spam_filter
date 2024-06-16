import pandas as pd
import numpy as np
# used for confusion matrix metric
from sklearn.metrics import confusion_matrix
# used for building and displaying the confusion matrix
import matplotlib.pyplot as plt
# used for displaying the heatmap properties for the confusion matrix
import seaborn as sns


class NaiveBayesEmailClassifier:
    def __init__(self, trainingData):

        self.trainingDataSet = trainingData

        # training spambase class priors
        self.trainingSpamClassPrior = self.determineClassPrior(self.trainingDataSet, 1)
        self.trainingHamClassPrior = self.determineClassPrior(self.trainingDataSet, 0)

        # defaults for each spam and ham mean and std
        self.spamsMean = []
        self.spamsStd = []

        self.hamsMean = []
        self.hamsStd = []

        self.determineClassFeatureMeansStds(trainingData)

        # performance metrics:
        self.totalEmailsClassified = 0

        self.truePositives = 0
        self.trueNegatives = 0
        self.falsePositives = 0
        self.falseNegatives = 0

        self.confusionMatrix = None


    #  supply a 0 for ham and 1 for spam
    def determineClassPrior(self, dataSet, targetClass):
        return ((dataSet.iloc[:, -1] == targetClass).sum()) / len(dataSet)

    def determineClassFeatureMeansStds(self, trainingData):

        # calculates the mean for each feature within the training data
        foundSpams = trainingData[trainingData.iloc[:, -1] == 1]
        foundHam = trainingData[trainingData.iloc[:, -1] == 0]

        #  pull the std and mean for each class respectively.
        foundSpamsMean = foundSpams.iloc[:, :-1].mean(axis=0)
        foundSpamsStd = foundSpams.iloc[:, :-1].std(axis=0)

        foundHamsMean = foundHam.iloc[:, :-1].mean(axis=0)
        foundHamsStd = foundHam.iloc[:, :-1].std(axis=0)

        # replace 0 with a small number for underflow and prevent division by zero
        # inplace=True swaps the current df without need for reassignment of the df

        foundSpamsStd.replace(0.0, 0.0001, inplace=True)

        foundHamsStd.replace(0.0, 0.0001, inplace=True)

        self.spamsMean = np.array(foundSpamsMean.to_list())
        self.spamsStd = np.array(foundSpamsStd.to_list())

        self.hamsMean = np.array(foundHamsMean.to_list())
        self.hamsStd = np.array(foundHamsStd.to_list())

    def classifyEmails(self, emailFeatures, emailTargets):
        predictedValues = []

        for email in range(len(emailFeatures)):
            self.totalEmailsClassified += 1
            emailFeatureVector = emailFeatures[email]
            targetClass = emailTargets[email]
            # adds a posteriors list
            posteriors = []

            # classify ham
            hamPosterior = np.sum(np.log(self.gaussianNB(emailFeatureVector, self.hamsMean, self.hamsStd)) + np.log(self.trainingHamClassPrior))
            posteriors.append(hamPosterior)

            # classify spam
            spamPosterior = np.sum(np.log(self.gaussianNB(emailFeatureVector, self.spamsMean, self.spamsStd)) + np.log(self.trainingSpamClassPrior))
            posteriors.append(spamPosterior)

            prediction = np.argmax(posteriors, axis=0)

            predictedValues.append(prediction)

            if prediction == targetClass:
                # classify prediction for accuracy metrics:
                # case where the target is classified as spam
                # case where the email is correctly classified as spam
                if prediction == 1 and targetClass == 1:
                    self.truePositives += 1
                # case where classified as not spam and is non-spam
                if prediction == 0 and targetClass == 0:
                    self.trueNegatives += 1
            # case where it wasn't correctly predicted.
            else:
                # case where the email is falsely classified as spam
                if prediction == 1 and targetClass == 0:
                    self.falsePositives += 1
                # case where classified as non spam but is spam email
                if prediction == 0 and targetClass == 1:
                    self.falseNegatives += 1

        self.confusionMatrix = confusion_matrix(emailTargets, predictedValues)



    def coefficientEFraction(self, classStd):
        return 1 / np.sqrt((2 * np.pi) * classStd )


    def exponentialFractionTerm(self, featureVector, classMean, classStd):
        eps = 1e-4
        return (((featureVector - classMean) ** 2) / (2 * (classStd ** 2) + eps))

    def gaussianNB(self, emailFeatureVector, classMean, classStd):
        result = self.coefficientEFraction(classStd) * (np.exp(-1 * self.exponentialFractionTerm(emailFeatureVector, classMean, classStd)))
        return result

    def determineAccuracy(self):
        print("Accuracy Results:")
        print("====================================")
        print("Out of " + str(self.totalEmailsClassified) + " emails:")
        accuracy = (self.truePositives + self.trueNegatives) / (self.truePositives + self.falsePositives + self.trueNegatives + self.falseNegatives) * 100
        print( "Accuracy of : " + str(round(accuracy,2)) + "%")
        print("====================================")

    def determinePrecision(self):
        print("Precision Results:")
        print("====================================")
        print("Out of " + str(self.totalEmailsClassified) + " emails:")
        precision = (self.truePositives) / (self.truePositives + self.falsePositives) * 100
        print("Precison of : " + str(round(precision, 2)))
        print("====================================")

    def determineRecall(self):
        print("Recall Results:")
        print("====================================")
        print("Out of " + str(self.totalEmailsClassified) + " emails:")
        recall = (self.truePositives) / (self.truePositives + self.falseNegatives) * 100
        print("Recall of : " + str(round(recall, 2)))
        print("====================================")

    def displayConfusionMatrix(self):

        if len(self.confusionMatrix) > 0:
            labels = ['Not Spam', 'Spam']
            sns.heatmap(self.confusionMatrix, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()
        else:
            return "run a classification in order to display a confusion matrix"



# build 2300 instances of spam and ham
# split 40% spam 60% ham
def buildTrainingTestData(spamSplit, hamSplit):
    spambase = pd.read_csv("spambase.data", header=None)

    # get the number of spams and hams required
    spamPulls = int(2300 * spamSplit)
    hamPulls = int(2300 * hamSplit)

    #  shuffle original data
    spambase = spambase.sample(frac=1)


    # Training Data Preparations:
    # =======================================================

    #  pull only the desired number of spams
    trainingSpams = spambase[spambase.iloc[:, -1] == 1][:spamPulls]
    trainingHams = spambase[spambase.iloc[:, -1] == 0][:hamPulls]

    # removes data so test can't pull it into its own DF
    #  this is done with a boolean mask to remove the corresponding index values from the original df
    spambase = spambase[~spambase.index.isin(trainingSpams.index)]
    spambase = spambase[~spambase.index.isin(trainingHams.index)]


    training_spambase = pd.concat([trainingHams, trainingSpams])

    #  Testing Data
    # ==============================================================

    #  pull only the desired number of spams
    testingSpams = spambase[spambase.iloc[:, -1] == 1][:spamPulls]
    testingHams = spambase[spambase.iloc[:, -1] == 0][:hamPulls]

    # removes data so test can't pull it into its own DF
    # ~ negates the boolean mask to select the rows that are not present in spambase and filter them
    spambase = spambase[~spambase.index.isin(testingSpams.index)]
    spambase = spambase[~spambase.index.isin(testingHams.index)]

    testing_spambase = pd.concat([testingHams, testingSpams])

    # return the data from split
    return training_spambase, testing_spambase

def prepareTestData(testingData):
    # shuffle the test data.
    testingData = testingData.sample(frac=1)

    # split the testing features from the label.
    inputFeatures = np.array(testingData.iloc[:, :-1])

    # pull classifications and for comparison.
    inputTargets = np.array(testingData.iloc[:, -1].to_list())

    return inputFeatures, inputTargets






def main():

    trainingSpambaseData, testingSpambaseData = buildTrainingTestData(.40,.60)

    emailClassifier = NaiveBayesEmailClassifier(trainingSpambaseData)


    #  split target and features for testing.
    inputFeature, inputTargets = prepareTestData(testingSpambaseData)

    emailClassifier.classifyEmails(inputFeature, inputTargets)

    emailClassifier.determineAccuracy()
    emailClassifier.determinePrecision()
    emailClassifier.determineRecall()

    emailClassifier.displayConfusionMatrix()

    # used for visual correlation inspection of the data.
    # determine correlation for the features:
    correlationMatrix = testingSpambaseData.corr()
    # display.max_columns lets all of the 57 features be displayed.
    pd.set_option('display.max_columns', None)
    print(correlationMatrix.head())












main()


