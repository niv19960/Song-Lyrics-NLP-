import standAloneModel
import mlBasicModel
import preparingData
import RandomForestFineTuning
import NNModel


def main():
    preparingData.handlingData()
    mlBasicModel.basicModelPipeline()
    RandomForestFineTuning.fineTuning()
    standAloneModel.FineTunedModel()
    NNModel.neuralNetwork()


if __name__ == '__main__':
    main()
