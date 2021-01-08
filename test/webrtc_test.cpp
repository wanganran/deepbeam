#include <iostream>
#include <vector>
#define WEBRTC_AUDIO_PROCESSING_ONLY_BUILD
#include <webrtc/modules/audio_processing/beamformer/nonlinear_beamformer.h>
using namespace std;
#include <chrono>

int main(){

    std::vector<webrtc::Point> mics(6);
    for(int i=0;i<mics.size();i++)
        mics[i]=webrtc::Point(i*0.05,0,0);

    webrtc::NonlinearBeamformer beamformer(mics);
    const int FRAME=256;
    const int CHANNEL=4;
    const int SAMP_RATE=48000;
    const float delay=(float)FRAME/SAMP_RATE*1000;
    printf("delay %f\n", delay);
    beamformer.Initialize(delay, SAMP_RATE);

    float* randsig=new float[FRAME];
    webrtc::SphericalPointf dir(0.1, 0, 10);
    beamformer.AimAt(dir);
    for(int it=0;it<2;it++){


        for(int i=0;i<FRAME;i++){
            randsig[i]=(rand()-RAND_MAX/2)/(float)RAND_MAX;
        }
        webrtc::ChannelBuffer<float> buffer(FRAME, CHANNEL);
        cout<<"==="<<endl;
        for(int i=0;i<CHANNEL;i++){
            float* dest=buffer.bands(i)[0];
            memcpy(dest, randsig, sizeof(float)*FRAME);
        }

        webrtc::ChannelBuffer<float> output(FRAME, 1);

        beamformer.ProcessChunk(buffer,&output);

        cout<<"==="<<endl;


        for(int i=0;i<FRAME;i++){
            cout<<i<<"\t"<<randsig[i]<<"\t"<<output.bands(0)[0][i]<<endl;
        }
    }
}