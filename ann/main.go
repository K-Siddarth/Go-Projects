package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func mnist_train(net *Network) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()
	line_num := 0
	for epochs := range 5 {
		fmt.Println("epochs: ", epochs)
		train_file, _ := os.Open("data/mnist_train.csv")
		read := csv.NewReader(bufio.NewReader(train_file))
		line_num = 0
		for {
			line_num++
			record, err := read.Read()
			if err == io.EOF {
				// file done
				break
			}

			inputs := make([]float64, net.inputLayers)
			for i := 0; i < net.inputLayers; i++ {
				x, _ := strconv.ParseFloat(record[i+1], 64)
				inputs[i] = (x / 255.0 * 0.99) + 0.01
			}

			targets := make([]float64, 10) // contains the probability of the image being the number i
			for i := range targets {
				targets[i] = 0.01
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.99

			net.Train(inputs, targets)
			fmt.Println(line_num, "Training done")
		}
		train_file.Close()
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train : %v\n", elapsed)
}

func mnist_predict(net *Network) {
	t1 := time.Now()
	test_file, _ := os.Open(("data/mnist_test.csv"))
	defer test_file.Close()
	line_num := 0
	read := csv.NewReader((bufio.NewReader(test_file)))
	score := 0
	for {
		line_num++
		record, err := read.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, net.inputLayers)
		for i := 0; i < net.inputLayers; i++ {
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.99) + 0.01
		}
		outputs := net.Predict(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < net.outputLayers; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
		fmt.Println(line_num, "Prediction done")
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime for testing : %v\n", elapsed)
	fmt.Printf("\nScore: %v\n", score)

}

func save(net Network) {
	h, err := os.Create("model/hidden_weights.model")
	if err != nil {
		defer h.Close()
	}

	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	out, err := os.Create("model/output_weights.model")
	if err != nil {
		defer out.Close()
	}
	if err == nil {
		net.outputsWeights.MarshalBinaryTo(out)
	}
	fmt.Println("saved")
}

func load(net *Network) {
	h, err := os.Open("model/hidden_weights.model")

	defer h.Close()

	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	out, err := os.Create("model/output_weights.model")
	defer out.Close()

	if err == nil {
		net.outputsWeights.Reset()
		net.outputsWeights.UnmarshalBinaryFrom(out)
	}
	fmt.Println("Loaded")
}

func main() {
	net := CreateNetwork(784, 200, 10, 0.01)

	mnist := flag.String("mnist", "", "Training/Prediction")
	flag.Parse()

	switch *mnist {
	case "train":
		mnist_train(&net)
		save(net)
	case "predict":
		load(&net)
		mnist_predict(&net)
	default:
		panic("[Error]: Invalid argument. Either run in train mode or predict")
	}
}
