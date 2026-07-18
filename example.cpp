#include "funn.hpp"

int main(){
    funn::NN<> nn({1,50,1});
    nn.layers[1].printNeurons();

}