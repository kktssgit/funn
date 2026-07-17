#pragma once
#include <vector>
#include <concepts>

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

template<Number num = double>
struct NN{
    struct Neuron{
        num value;
        num bias;
        Neuron(num v = num{}, num b = num{}) : value(v), bias(b) {}
    };
    
    struct Layer{
        
    };
};



}
