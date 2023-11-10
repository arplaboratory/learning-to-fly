# Learning to Fly in Seconds

<div align="center">
<img src="https://github.com/arplaboratory/learning_to_fly_in_seconds_media/blob/master/training_simulation.gif" alt="animated" height='300'/>
</div>
<div align="center">
<img src="https://github.com/arplaboratory/learning_to_fly_in_seconds_media/blob/master/trajectory_tracking_long_exposure.gif" alt="animated" height='200'/>
</div>

This repository contains the code for the paper "Learning to Fly in Seconds". The main dependency is the [BackpropTools](https://github.com/BackpropTools/BackpropTools) deep reinforcement learning library.

### Instructions to run the code
Clone this repository:
```
git clone https://github.com/arplaboratory/learning_to_fly_in_seconds
cd learning_to_fly_in_seconds
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
brew install hdf5 protobuf-compiler protobuf boost
```




Going back to the main directory (`learning_to_fly_in_seconds`), we can now configure the build of the code:
```
cd ../../
mkdir build
cd build
```
- Ubuntu + OpenBLAS: `cmake .. -DBACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS:BOOL=ON`
- Ubuntu + MKL: `cmake .. -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON`
- macOS (tested on Sonoma): `cmake ..`

Finally we can build the targets:
```
cmake --build . -j8
```

After successfully building the targets, we can run the code:
```

```
