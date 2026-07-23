#include "funn.hpp"

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cmath>

double Linear(double x){
    return x;
}
double Linear_dx(double x){
    return 1;
}

int main(){

    funn::NN<> nn({1,64,64,64,1},
    funn::diffSquared<>,funn::diffSquared_dx<>,
    funn::Sigmoid2<>,   funn::Sigmoid2_dx<>);

    nn.layers.back().activationFunction = Linear;
    nn.layers.back().activationFunction_dx = Linear_dx;

    srand(getpid());
    std::vector<funn::Data<>>dataset;

    for(int i{};i<1024;i++){
        double r = (double((rand()%1256636)-628318)/100000);

        funn::Data<>dp;
        dp.input.push_back(r / 6.28);
        dp.expected.push_back(sin(r));
        dataset.push_back(dp);
    }

    for(int i{};i<5000;i++){
    std::vector<funn::Data<>>batch;
        for(int l{};l<1024;l++){
            batch.push_back(dataset[l]);
            if((l+1)%32==0){
                nn.learn(batch,0.1);
                std::cout<<"learned | epoch: "<<i<<" | batch: "<<(l+1)/32<<'\n';
                batch.clear();
            }
        }
    }

    double error = 0;
    for(int i{};i<100;i++){
        double r = (double((rand()%1256636)-628318)/100000);

        funn::Data<>dp;
        dp.input.push_back(r / 6.28);
        dp.expected.push_back(sin(r));

        nn.calc(dp);
        auto out = nn.getOutput();
        double output = out.output.front();

        error += fabs(output-sin(r));
        std::cout<<"Output for x = "<<r<<" -> "<<output<<'\n';
    }
    error /= 100;
    std::cout<<"Average error: "<<error<<'\n';


}