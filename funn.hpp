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

// concept for dynamic typing

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



// example functions

template<Number num = double>
num ReLU(num x){
    return x>0?x:0;
}

template<Number num = double>
num ReLU_dx(num x){
    return x>0?1:0;
}

template<Number num = double>
num Sigmoid2(num x){
    return (2.0 / (1.0 + std::exp(-x))) - 1.0;
}

template<Number num = double>
num Sigmoid2_dx(num x){
    num y = Sigmoid2(x);
    return (1.0 - y * y) / 2.0;
}

template<Number num = double>
num derive(num(*fun)(num), num x, num h = num(0.000001)){
    return (fun(x+h)-fun(x))/h;
}

template<Number num = double>
num RandInitWeights(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(-0.5,0.5);
    return num(dist(gen));
}

template<Number num = double>
num diffSquared(num predicted, num expected){
    return (0.5)*(predicted-expected)*(predicted-expected);
}

template<Number num = double>
num diffSquared_dx(num predicted, num expected){
    return (predicted-expected);
}

// data structures

template<Number num = double>
struct Data{
    std::vector<num>input{};
    std::vector<num>expected{};
};

template<Number num = double>
struct Output{
    std::vector<num>output;
};

template<Number num = double>
struct NN{
    struct Neuron{
        num value;
        num value_pa{}; // value pre activation function
        num bias;
        std::vector<num>weights;

        std::vector<num>weight_gradients={};
        num bias_gradient={};

        num delta{};

        Neuron(
            size_t wsize,
            num(*initFunc)() ) {
            weights.resize(wsize);
            weight_gradients.assign(wsize,num(0.0));
            for(auto& i:weights) i = initFunc();
            bias = initFunc();
        }
    };
    
    struct Layer{
        std::vector<Neuron>neurons;
        size_t lsize;
        size_t ins,outs{};

        num(*activationFunction)(num);
        num(*activationFunction_dx)(num); // activation function derivative
        // num(*neuronInitFunction)();
        
        Layer(
            size_t lsize,
            size_t in,
            num(*activFunc)(num) = ReLU<num>,
            num(*activFuncDx)(num) = ReLU_dx<num>,
            num(*initFunc)() = RandInitWeights<num>
        ) : lsize(lsize), ins(in),
        activationFunction(activFunc),
        activationFunction_dx(activFuncDx)
        /* neuronInitFunction(initFunc) */ {
            for(size_t i{};i<lsize;i++)
                neurons.emplace_back(Neuron(in, initFunc));
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
    std::vector<num>loss;

    num(*lossFunction)(num,num);
    num(*lossFunction_dx)(num,num);

    // make another constructor which accepts all the other params

    NN( // will fail if layer_sizes is empty i think
        const std::vector<size_t>&layer_sizes,
        num(*lossFun)(num,num) = diffSquared<>,
        num(*lossFunDx)(num,num) = diffSquared_dx<>,
        num(*activFunDefault)(num) = ReLU<>,
        num(*activFunDefaultDx)(num) = ReLU_dx<>
    ) : 
        input_size(layer_sizes.front()),
        output_size(layer_sizes.back()),
        layers_amount(layer_sizes.size()),
        lossFunction(lossFun),
        lossFunction_dx(lossFunDx)
    {
        for(size_t i{};i<layer_sizes.size();i++){
            layers.emplace_back(Layer(
                layer_sizes[i],
                (i==0) ? 0 : layer_sizes[i-1],
                activFunDefault,
                activFunDefaultDx
            ));
        }
        for(auto& neuron:layers.front().neurons)
            neuron.value = neuron.value_pa = neuron.bias = num(0.0);

        loss.assign(output_size,num(0.0));
    }

    void calc(const Data<num>&data, bool learning = false){
        auto& in = data.input;

        assert(in.size() == input_size);

        for(auto layer_p = layers.begin(); layer_p != layers.end(); layer_p++){
            if(layer_p == layers.begin()){
                size_t n_i{};
                for(auto& neuron : layer_p->neurons) neuron.value_pa = neuron.value = in[n_i++];
                continue;
            }

            auto prev_layer = layer_p; prev_layer--;

            for(auto& neuron : layer_p->neurons){
                neuron.value_pa = neuron.bias;
                for(size_t i{};i<prev_layer->neurons.size();i++)
                    neuron.value_pa += prev_layer->neurons[i].value * neuron.weights[i];
                neuron.value = layer_p->activationFunction(neuron.value_pa);
            }
        }

        if(learning)
            for(size_t l_i{};l_i<layers.back().neurons.size();l_i++)
                loss[l_i]=lossFunction(layers.back().neurons[l_i].value,data.expected[l_i]);
    }

    Output<num> getOutput(){
        Output out{std::vector<num>()};
        for(const auto& neuron : layers.back().neurons) out.output.push_back(neuron.value);
        return out;
    }

    void learn(const std::vector<Data<num>>&dataset, double learning_rate = 0.01){
        // clear all gradients
        for(auto& layer:layers)
            for(auto& neuron:layer.neurons)
                std::fill(neuron.weight_gradients.begin(),neuron.weight_gradients.end(),num(0.0)), 
                neuron.bias_gradient = num(0.0);

        for(auto& dataPoint:dataset){
            calc(dataPoint, 1);

            for(auto layer_p = layers.rbegin(); layer_p != layers.rend()-1; layer_p++){
                
                auto prev_layer = layer_p+1;

                if(layer_p == layers.rbegin()){
                    size_t d_i{};
                    for(auto& neuron:layer_p->neurons){
                        neuron.delta = lossFunction_dx(neuron.value, dataPoint.expected[d_i++]) *
                                       layer_p->activationFunction_dx(neuron.value_pa);

                        neuron.bias_gradient += neuron.delta;

                        for(size_t w_i{}; w_i<neuron.weights.size();w_i++){
                            neuron.weight_gradients[w_i] += neuron.delta * prev_layer->neurons[w_i].value;
                        }
                    }
                } else {
                    auto next_layer = layer_p-1;

                    size_t k_i{};
                    for(auto& neuron:layer_p->neurons){
                        
                        num error_sum = num(0.0);
                        for(auto& next_neuron:next_layer->neurons){
                            error_sum += next_neuron.delta * next_neuron.weights[k_i];
                        }
                        neuron.delta = error_sum * layer_p->activationFunction_dx(neuron.value_pa);

                        neuron.bias_gradient += neuron.delta;

                        for(size_t w_i{}; w_i < neuron.weights.size();w_i++)
                            neuron.weight_gradients[w_i] += neuron.delta * prev_layer->neurons[w_i].value;

                        k_i++;
                    }
                }
            }
        }

        for(auto& layer:layers){
            for(auto& neuron:layer.neurons){
                for(size_t w_i{}; w_i<neuron.weights.size();w_i++)
                    neuron.weights[w_i] -= (neuron.weight_gradients[w_i]/dataset.size())*learning_rate;
                neuron.bias -= (neuron.bias_gradient/dataset.size())*learning_rate;
            }
        }
    }

};



}
