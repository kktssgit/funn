#include "funn.hpp"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <cmath>

double Linear(double x){
    return x;
}
double Linear_dx(double x){
    return 1;
}

double Sin(double x){
    return sin(x);
}

double Sin_dx(double x){
    return cos(x);
}

int main(){

    // Fourier features: hand the net high-frequency basis functions of x directly,
    // instead of asking tanh units to synthesize them (beats spectral bias).
    const int K = 32;
    std::vector<double> freqs;
    for(int j{};j<K;j++) freqs.push_back(std::pow(2.0, 8.0*j/(K-1)));  // 1 .. 256
    auto feat = [&](double x){
        std::vector<double> v{ x };
        for(double f : freqs){ v.push_back(std::sin(f*x)); v.push_back(std::cos(f*x)); }
        return v;
    };
    const size_t IN = 1 + 2*K;

    funn::NN<> nn({IN,200,1},
    funn::diffSquared<>,funn::diffSquared_dx<>,
    funn::Sigmoid2<>,   funn::Sigmoid2_dx<>);

    nn.layers.back().activationFunction = Linear;
    nn.layers.back().activationFunction_dx = Linear_dx;

    srand(getpid());
    std::vector<funn::Data<>>dataset;

    const double gap = 0.02, xmax = 1.0;   // stay clear of the x=0 singularity of sin(1/x)
    auto target = [](double x){ return sin(1.0/x); };

    for(int i{};i<1024;i++){
        double m = gap + (xmax-gap)*(rand()/double(RAND_MAX));
        double x = (rand()&1) ? m : -m;        // both signs -> odd, mirrored output

        funn::Data<>dp;
        dp.input = feat(x);
        dp.expected.push_back(target(x));
        dataset.push_back(dp);
    }

    const int EPOCHS = 10000;
    for(int i{};i<EPOCHS;i++){
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

    // sweep both sign ranges and plot NN output vs real sin(1/x) as an SVG
    const int W = 900, H = 400, N = 3000;
    const double plo = -1.0, phi = 1.0, ylo = -1.5, yhi = 1.5;
    auto px = [&](double x){ return (x-plo)/(phi-plo)*W; };
    auto py = [&](double y){ return H - (y-ylo)/(yhi-ylo)*H; };

    double error = 0; int nErr = 0;
    // one polyline per side so the pen doesn't draw across the x=0 gap
    auto sweep = [&](double a, double b, std::ostringstream& sp, std::ostringstream& np){
        for(int i{};i<N;i++){
            double x = a + (b-a)*i/(N-1);
            funn::Data<>dp; dp.input = feat(x);
            nn.calc(dp);
            double output = nn.getOutput().output.front();
            error += fabs(output-target(x)); nErr++;
            sp<<px(x)<<','<<py(target(x))<<' ';
            np<<px(x)<<','<<py(output)<<' ';
        }
    };
    std::ostringstream sinL, nnL, sinR, nnR;
    sweep(-xmax, -gap, sinL, nnL);
    sweep( gap,  xmax, sinR, nnR);
    error /= nErr;
    std::cout<<"Average error: "<<error<<'\n';

    std::ofstream svg("network.svg");
    svg<<"<svg xmlns='http://www.w3.org/2000/svg' width='"<<W<<"' height='"<<H<<"' font-family='sans-serif'>\n"
       <<"<rect width='100%' height='100%' fill='white'/>\n"
       <<"<line x1='0' y1='"<<py(0)<<"' x2='"<<W<<"' y2='"<<py(0)<<"' stroke='#ccc'/>\n"
       <<"<line x1='"<<px(0)<<"' y1='0' x2='"<<px(0)<<"' y2='"<<H<<"' stroke='#ccc'/>\n"
       <<"<polyline fill='none' stroke='#888' stroke-width='2' points='"<<sinL.str()<<"'/>\n"
       <<"<polyline fill='none' stroke='#888' stroke-width='2' points='"<<sinR.str()<<"'/>\n"
       <<"<polyline fill='none' stroke='#e4572e' stroke-width='2' points='"<<nnL.str()<<"'/>\n"
       <<"<polyline fill='none' stroke='#e4572e' stroke-width='2' points='"<<nnR.str()<<"'/>\n"
       <<"<text x='12' y='20' fill='#888'>sin(1/x)</text>\n"
       <<"<text x='12' y='38' fill='#e4572e'>network output</text>\n"
       <<"<text x='"<<W-12<<"' y='20' text-anchor='end' fill='#555'>trained "<<EPOCHS<<" epochs, avg error "<<std::fixed<<std::setprecision(4)<<error<<"</text>\n"
       <<"</svg>\n";
    svg.close();
    std::cout<<"wrote network.svg\n";
}