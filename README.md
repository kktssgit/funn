# funn

A tiny header-only C++20 feedforward neural network library.

## Features

- Single header (`funn.hpp`), no dependencies beyond the standard library
- Templated on any numeric type satisfying the `Number` concept (defaults to `double`)
- Configurable per-layer activation and weight-initialization functions
- Forward pass (`NN::calc`) only — no training/backpropagation yet

## Usage

```cpp
#include "funn.hpp"

int main(){
    funn::NN<> nn({1, 3, 1}); // input size 1, one hidden layer of 3, output size 1

    funn::Data data{{5.0}, 3.0}; // input vector + expected output
    auto out = nn.calc(data);

    std::cout << out.output.front() << '\n';
}
```

Build with a C++20 compiler:

```sh
g++ -std=c++20 example.cpp -o example
```

## Layout

- `funn.hpp` — the library
- `example.cpp` — minimal usage example
