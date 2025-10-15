package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := m.Dims()
	if r != c {
		panic("Invalid input : rows and columns do not align for a dot product")
	}
	res := mat.NewDense(r, c, nil)
	res.Product(m, n)
	return res
}

func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Apply(fn, res)
	return res
}

func Sigmoid(r, c int, z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func SigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	res := make([]float64, rows)
	for i := range res {
		res[i] = 1
	}
	ones := mat.NewDense(rows, 1, res)
	return Multiply(m, Subtract(ones, m))
}

func Scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Scale(s, m)
	return res
}

func Multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.MulElem(m, n)
	return res
}

func Subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Sub(m, n)
	return res
}

func Add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Add(m, n)
	return res
}
