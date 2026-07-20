#pragma once
#include <vector>
#include <concepts>
#include <cmath>
#include <memory>
#include <random>
#include <cassert>

#include <iostream>
#include <iomanip>

namespace funn{

template <typename T>
concept Number =
    std::copyable<T> &&
    std::totally_ordered<T> &&
    requires(T a, T b) {
        {a+b} -> std::convertible_to<T>;
        {a-b} -> std::convertible_to<T>;
        {a*b} -> std::convertible_to<T>;
        {a/b} -> std::convertible_to<T>;
        {a+=b} -> std::convertible_to<T&>;
        {a-=b} -> std::convertible_to<T&>;
        {a*=b} -> std::convertible_to<T&>;
        {a/=b} -> std::convertible_to<T&>;
    };

template<Number num = double>
num ReLU(num x){
    return (x>num(0))?x:num(0);
}

// example random initialization function for weights

template<Number num = double>
num RandInitWeights(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(-1.0,1.0);
    return num(dist(gen));
}

template<Number num = double>
struct Data{
    std::vector<num>input;
    num expected;
};

template<Number num = double>
struct Output{
    std::vector<num>output;
};

template<Number num = double>
struct NN{
    struct Neuron{
        num value;
        num bias;
        std::vector<num>weights;
        Neuron(
            size_t wsize,
            num(*initFunc)(),
            num v,
            num b)
        : value(v), bias(b){
            weights.resize(wsize);
            for(auto& i:weights) i = initFunc();
        }
    };
    
    struct Layer{
        std::vector<Neuron>neurons;
        size_t lsize;
        size_t ins,outs{};

        num(*activationFunction)(num);
        num(*neuronInitFunction)();
        
        Layer(
            size_t lsize,
            size_t in,
            num(*activFunc)(num) = ReLU<num>,
            num(*initFunc)() = RandInitWeights<num>,
            num defaultNeuronBias = num{},
            num defaultNeuronValue = num{}
        ) : lsize(lsize), ins(in),
        activationFunction(activFunc),
        neuronInitFunction(initFunc) {
            for(size_t i{};i<lsize;i++)
                neurons.emplace_back(Neuron(in, initFunc, defaultNeuronValue, defaultNeuronBias));
        }

        void printNeurons(){
            int i{};
            for(auto& n:neurons){
                i++;
                std::cout<<"Neuron "<<i<<" value: "<<n.value<<", bias: "<<n.bias<<",\nweights: { ";
                for(auto w:n.weights) std::cout<<std::fixed<<std::setprecision(3)<<w<<' ';
                std::cout<<"}\n\n";
            }
        }
    };

    size_t layers_amount{};
    size_t input_size{};
    size_t output_size{};
    std::vector<Layer>layers;

    // make another constructor which accepts all the other params

    NN( // will fail if layer_sizes is empty i think
        const std::vector<size_t>&layer_sizes
    ) : 
        input_size(layer_sizes.front()),
        output_size(layer_sizes.back()),
        layers_amount(layer_sizes.size())
    {
        for(size_t i{};i<layer_sizes.size();i++){
            layers.emplace_back(Layer(
                layer_sizes[i],
                (i==0) ? 0 : layer_sizes[i-1] 
            ));
        }
    }

    Output<num> calc(const Data<num>& data){
        auto& in = data.input;

        assert(in.size() == input_size);

        Output out{std::vector<num>()};

        for(auto layer_p = layers.begin(); layer_p != layers.end(); layer_p++){
            if(layer_p == layers.begin()){
                size_t n_i{};
                for(auto& neuron : layer_p->neurons) neuron.value = in[n_i++];
                continue;
            }

            auto prev_layer = layer_p; prev_layer--;

            for(auto& neuron : layer_p->neurons){
                neuron.value = neuron.bias;
                for(size_t i{};i<prev_layer->neurons.size();i++)
                    neuron.value += prev_layer->neurons[i].value * neuron.weights[i];
                neuron.value = layer_p->activationFunction(neuron.value);
            }
        }

        for(const auto& neuron : layers.back().neurons) out.output.push_back(neuron.value);

        return out;
    }
};



}
