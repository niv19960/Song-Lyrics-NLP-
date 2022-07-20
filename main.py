import mlBasicModel
import preparingData
import RandomForestFineTuning
import NNModel


def main():
    preparingData.handlingData()
    mlBasicModel.basicModelPipeline()
    RandomForestFineTuning.fineTuning()
    NNModel.neuralNetwork()
    print("OK!")


if __name__ == '__main__':
    main()
