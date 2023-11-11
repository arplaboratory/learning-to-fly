# Learning to Fly in Seconds

<div align="center">
<img src="https://github.com/arplaboratory/learning_to_fly_media/blob/master/training_simulation.gif" alt="animated" height='300'/>
</div>
<div align="center">
    Trained for 18s on a 2020 MacBook Pro (M1) using <span style="color:#7DB9B6">BackpropTools</span> then transfered to a real Crazyflie:
</div>
<div align="center">
<img src="https://github.com/arplaboratory/learning_to_fly_media/blob/master/trajectory_tracking_long_exposure.gif" alt="animated" height='200'/>
</div>

## Introduction
This repository contains the code for the paper "Learning to Fly in Seconds". The main dependency is the [BackpropTools](https://github.com/BackpropTools/BackpropTools) deep reinforcement learning library.

## Instructions to run the code
### Docker installation (isolated)
The easiest way to run the code is probably using Docker. If you want to run the code on bare metal jump [Native installation](#Native-installation).

First, install Docker on your machine. Then move to the original directory `learning_to_fly` and build the Docker image:
```
docker build -t learning_to_fly .
```
After that you can run it using e.g.:
```
docker run -it --rm -p 8000:8000 learning_to_fly
```
This will open the port `8000` for the UI of the training program and run it inside the container.

Navigate to "http://localhost:8000" with your browser, and you should see something like (after starting the training):

<div align="center">
<img src="https://github.com/arplaboratory/learning_to_fly_media/blob/master/simulator_screenshot.png" />
</div>

The training UI configuration does not log data by default. If you want to inspect the training data run:
```
docker run -it --rm -p 6006:6006 learning_to_fly training_headless
```
Navigate to "http://localhost:6006" with your browser to investigate the Tensorboard logs.

If you would like to benchmark the training speed you can use:
```
docker run -it --rm learning_to_fly training_benchmark
```
This is the fastest configuration, without logging, UI, checkpointing etc.
### Native installation
Clone this repository:
```
git clone https://github.com/arplaboratory/learning_to_fly
cd learning_to_fly
```
Then instantiate the `BackpropTools` submodule:
```
git submodule update --init -- external/backprop_tools
cd external/backprop_tools
```

Then instantiate some dependencies of `BackpropTools` (for conveniences like checkpointing, Tensorboard logging, testing, etc.):
```
git submodule update --init -- external/cli11 external/highfive external/json/ external/tensorboard tests/lib/googletest/
```

#### Install dependencies on Ubuntu
```
sudo apt update && sudo apt install libhdf5-dev libopenblas-dev protobuf-compiler libprotobuf-dev libboost-all-dev
```
As an alternative to openblas you can also install [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) which in our experience is faster than OpenBLAS.
#### Install dependencies on macOS
```
brew install hdf5 protobuf boost
```




Going back to the main directory (`learning_to_fly`), we can now configure the build of the code:
```
cd ../../
mkdir build
cd build
```
- Ubuntu + OpenBLAS: `cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS:BOOL=ON`
- Ubuntu + MKL: `cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON`
- macOS (tested on Sonoma): `cmake .. -DCMAKE_BUILD_TYPE=Release`

Finally, we can build the targets:
```
cmake --build . -j8
```

After successfully building the targets, we can run the code (in the original directory `learning_to_fly`):
```
cd ..
./build/src/rl_environments_multirotor_training_headless 
```
While this is running, you should be able to see training metrics using Tensorboard

If not already installed:
```
python3 -m pip install tensorboard
```
Then from the original directory `learning_to_fly`:
```
tensorboard --logdir=logs
```



