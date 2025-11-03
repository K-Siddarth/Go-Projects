[ANN]:
Training and testing data set downloaded from https://github.com/sausheong/gonn/blob/master/mnist_dataset/mnist.zip
ensure that data/ dir is a subdir of ann
To run the ann:
    go build
    ./ann -mnist train # for running the ann on training data set
    ./ann -mnist predict # for running the ann on testing data set

[SERVER]:
To run the server:
    go build 
    run the executable(main) in the directory (Contest directory) in which you want to create your problem directory
