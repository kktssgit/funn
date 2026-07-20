#include "funn.hpp"

int main(){
    funn::NN<> nn({1,3,1});
    std::cout<<"layer 1: \n"; nn.layers[0].printNeurons();
    std::cout<<"layer 2: \n"; nn.layers[1].printNeurons();
    std::cout<<"layer 3: \n"; nn.layers[2].printNeurons();

    funn::Data data{{5.0},3.0};
    auto out = nn.calc(data);

    std::cout<<out.output.front()<<'\n';

    std::cout<<"layer 1: \n"; nn.layers[0].printNeurons();
    std::cout<<"layer 2: \n"; nn.layers[1].printNeurons();
    std::cout<<"layer 3: \n"; nn.layers[2].printNeurons();
}