# funn

A tiny header-only C++20 feedforward neural network library.

## Features

- Single header (`funn.hpp`), no dependencies beyond the standard library
- Templated on any numeric type satisfying the `Number` concept (defaults to `double`)
- Configurable per-layer activation and weight-initialization functions
- Configurable loss function
- Forward pass (`NN::calc`) and training via backpropagation + mini-batch gradient descent (`NN::learn`)

## Usage

```cpp
#include "funn.hpp"

int main(){
    // input size 1, one hidden layer of 3, output size 1
    funn::NN<> nn({1, 3, 1});

    funn::Data<> data{{5.0}, {3.0}}; // input vector + expected output vector

    nn.calc(data);
    auto out = nn.getOutput();
    std::cout << out.output.front() << '\n';
}
```

The constructor also takes optional loss and activation functions:

```cpp
funn::NN<> nn({1, 200, 1},
    funn::diffSquared<>, funn::diffSquared_dx<>,   // loss + derivative
    funn::Sigmoid2<>,    funn::Sigmoid2_dx<>);     // default activation + derivative
```

Per-layer activation can be overridden after construction (e.g. a linear output layer):

```cpp
nn.layers.back().activationFunction    = Linear;
nn.layers.back().activationFunction_dx = Linear_dx;
```

### Training

`learn` takes a batch of `Data` and a learning rate, accumulates gradients over
the batch, and applies one gradient-descent step:

```cpp
std::vector<funn::Data<>> batch = /* ... */;
nn.learn(batch, 0.1);
```

See `example.cpp` for a full run that trains the network to approximate `sin(x)`.

Build with a C++20 compiler:

```sh
g++ -std=c++20 example.cpp -o example
```

## Layout

- `funn.hpp` — the library
- `example.cpp` — trains a network to approximate `sin(x)`
