package main

import (
	"ann/utils"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type Network struct {
	inputLayers    int        // number of input layers
	hiddenLayers   int        // number of hidden layers
	outputLayers   int        // number of output layers
	hiddenWeights  *mat.Dense // weights of input to first hidden layer
	outputsWeights *mat.Dense // weights of last hidden layer to output
	learningRate   float64    // learning rate of the algorithm
}

func CreateNetwork(_inputLayers, _hiddenLayers, _outputLayers int, _learningRate float64) (net Network) {
	net = Network{
		inputLayers:  _inputLayers,
		hiddenLayers: _hiddenLayers,
		outputLayers: _outputLayers,
		learningRate: _learningRate,
	}
	net.hiddenWeights = mat.NewDense(net.hiddenLayers, net.inputLayers, randomArray(net.hiddenLayers*net.inputLayers, float64(net.inputLayers)))
	net.outputsWeights = mat.NewDense(net.outputLayers, net.hiddenLayers, randomArray(net.hiddenLayers*net.outputLayers, float64(net.outputLayers)))
	return net
}

func (net Network) Predict(inputData []float64) mat.Matrix {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := utils.Dot(net.hiddenWeights, inputs)
	hiddenOutputs := utils.Apply(utils.Sigmoid, hiddenInputs)
	finalInputs := utils.Dot(net.outputsWeights, hiddenOutputs)
	finalOutputs := utils.Apply(utils.Sigmoid, finalInputs)
	return finalOutputs
}

func (net *Network) Train(inputData []float64, targetData []float64) {
	// forward propogation:
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := utils.Dot(net.hiddenWeights, inputs)
	hiddenOutputs := utils.Apply(utils.Sigmoid, hiddenInputs)
	finalInputs := utils.Dot(net.outputsWeights, hiddenOutputs)
	finalOutputs := utils.Apply(utils.Sigmoid, finalInputs)

	// finding errors:
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := utils.Subtract(targets, finalOutputs)
	hiddenErrors := utils.Dot(net.outputsWeights.T(), outputErrors)

	// back propogation:
	M := utils.Scale(
		net.learningRate,
		utils.Dot(
			utils.Multiply(outputErrors, utils.SigmoidPrime(finalOutputs)),
			hiddenOutputs.T(),
		),
	)
	net.outputsWeights = utils.Add(net.outputsWeights, M).(*mat.Dense)

	M = utils.Scale(
		net.learningRate,
		utils.Dot(
			utils.Multiply(hiddenErrors, utils.SigmoidPrime(hiddenOutputs)),
			inputs.T(),
		),
	)
	net.hiddenWeights = utils.Add(net.hiddenWeights, M).(*mat.Dense)
}

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}
	data = make([]float64, size)
	for i := 0; i < len(data); i++ {
		data[i] = dist.Rand()
	}
	return
}
