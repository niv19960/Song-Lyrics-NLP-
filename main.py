import mlBasicModel
import preparingData
import LogisticRegressionFineTuning
import NNModel


def main():
    preparingData.handlingData()
    mlBasicModel.basicModelPipeline()
    LogisticRegressionFineTuning.fineTuning()
    # NNModel.neuralNetwork()
    print("OK!")


if __name__ == '__main__':
    main()
