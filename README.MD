# Learning to Fly in Seconds

<div align="center">
<img src="https://github.com/arplaboratory/learning_to_fly_media/blob/master/training_simulation.gif" alt="animated" height='300'/>
</div>
<div align="center">
    Trained for 18s on a 2020 MacBook Pro (M1) using <span style="color:#7DB9B6">RLtools</span>.
</div>
<br>
<div align="center" >
<img src="https://github.com/arplaboratory/learning_to_fly_media/blob/master/trajectory_tracking_long_exposure.gif" alt="animated" height='200'/>
</div>
<div align="center">
Trajectory tracking using a trained policy on a real Crazyflie (nano quadrotor).
</div>
<br>



## Introduction
This repository contains the code for the paper [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081). The main dependency is the [RLtools](https://github.com/rl-tools/rl-tools) deep reinforcement learning library.



## Instructions to run the code
### Docker (isolated)
We provide a pre-built Docker image with a simple web interface that can be executed using a single command (given that Docker is already installed on your machine):
```
docker run -it --rm -p 8000:8000 arpllab/learning_to_fly
```
After the container is running, navigate to [https://0.0.0.0:8000](https://0.0.0.0:8000) and you should see something like (after starting the training):

<div align="center">
<img src="https://github.com/arplaboratory/learning_to_fly_media/blob/master/simulator_screenshot.png" />
</div>

Note that to make this Docker image compatible with a broad range of CPUs, some optimizations have been turned off. For full speed we recommend a [Native installation](#Native-installation). 
### Docker installation (isolated)
With the following instructions you can also easily build the Docker image yourself. If you want to run the code on bare metal jump [Native installation](#Native-installation).

First, install Docker on your machine. Then move to the original directory `learning_to_fly` and build the Docker image:
```
docker build -t arpllab/learning_to_fly .
```
If desired you can also build the container for building the firmware:
```
docker build -t arpllab/learning_to_fly_build_firmware -f Dockerfile_build_firmware .
```
After that you can run it using e.g.:
```
docker run -it --rm -p 8000:8000 arpllab/learning_to_fly
```
This will open the port `8000` for the UI of the training program and run it inside the container.

Navigate to [https://0.0.0.0:8000](https://0.0.0.0:8000) with your browser, and you should see something like in the screenshot above (after starting the training).

The training UI configuration does not log data by default. If you want to inspect the training data run:
```
docker run -it --rm -p 6006:6006 arpllab/learning_to_fly training_headless
```
Navigate to [https://0.0.0.0:6006](https://0.0.0.0:6006) with your browser to investigate the Tensorboard logs.

If you would like to benchmark the training speed you can use:
```
docker run -it --rm arpllab/learning_to_fly training_benchmark
```
This is the fastest configuration, without logging, UI, checkpointing etc.
### Native installation
Clone this repository:
```
git clone https://github.com/arplaboratory/learning-to-fly learning_to_fly
cd learning_to_fly
```
Then instantiate the `RLtools` submodule:
```
git submodule update --init -- external/rl_tools
cd external/rl_tools
```

Then instantiate some dependencies of `RLtools` (for conveniences like checkpointing, Tensorboard logging, testing, etc.):
```
git submodule update --init -- external/cli11 external/highfive external/json/ external/tensorboard tests/lib/googletest/
```

#### Install dependencies on Ubuntu
```
sudo apt update && sudo apt install libhdf5-dev libopenblas-dev protobuf-compiler libprotobuf-dev libboost-all-dev
```
As an alternative to openblas you can also install [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) which in our experience is significantly faster than OpenBLAS.
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
- Ubuntu + OpenBLAS: `cmake .. -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS:BOOL=ON`
- Ubuntu + MKL: `cmake .. -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON`
- macOS (tested on Sonoma): `cmake .. -DCMAKE_BUILD_TYPE=Release`

Finally, we can build the targets:
```
cmake --build . -j8
```

After successfully building the targets, we can run the code (in the original directory `learning_to_fly`):
```
cd ..
./build/src/training_headless 
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

To run the training with the UI, we download the JavaScript dependencies in the form of the two files `three.module.js` and `OrbitControls.js`:
```
cd src/ui
./get_dependencies.sh
```

After that we can execute the UI binary from the root folder:
```
cd ../../
./build/src/ui 0.0.0.0 8000
```
Now you should be able to navigate to [http://0.0.0.0:8000](http://0.0.0.0:8000) in your browser and start the training.

To run the benchmark (with UI, checkpointing and Tensorboard logging turned off):
```
sudo nice -n -20 ./build/src/training_benchmark
```

## Deploying trained policies on a Crazyflie
Train a policy, e.g. using the Docker image with the UI:
```
docker run -it --rm -p 8000:8000 -v $(pwd)/checkpoints:/learning_to_fly/checkpoints arpllab/learning_to_fly 
```
The checkpoints are placed in the current working directory's `checkpoints` folder. Inspect the logs of the container to find the path of the final log, e.g., `checkpoints/multirotor_td3/2023_11_16_14_46_38_d+o+a+r+h+c+f+w+e+_002/actor_000000000300000.h`. 
We can mount this file into the container `arpllab/learning_to_fly_build_firmware` for building the firmware, e.g.: 
```
docker run -it --rm -v $(pwd)/checkpoints/multirotor_td3/2023_11_16_14_46_38_d+o+a+r+h+c+f+w+e+_002/actor_000000000300000.h:/controller/data/actor.h:ro -v $(pwd)/build_firmware:/output arpllab/learning_to_fly_build_firmware
```
This should build the firmware using the newly trained policy and output the binary to `build_firmware/cf2.bin`. After that we can use the `cfclient` package to flash the firmware (find the installation instructions [here](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/installation/install/))
```
cfloader flash build_firmware/cf2.bin stm32-fw -w radio://0/80/2M
```

**Note: Running the learned controller is at your own risk!**

Once the firmware is flashed, use the [cfclient](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/userguides/userguide_client/) UI to connect to the Crazyflie. To fly you should have some system for position estimation. In our tests using the Crazyflie 2.1 and the [Optical Flow Deck v2](https://www.bitcraze.io/products/flow-deck-v2/) 10/10 seeds yielded policies that could successfully take-off and hover. We recommend using a game-controller to use the "hover button" as a [Dead man's switch](https://en.wikipedia.org/wiki/Dead_man%27s_switch). The way our controller is configured by default, it reacts to the "hover button" as a signal to be activated. By default, it will shortly idle the motors and then pass through the RPM outputs of the policy. The default setpoint is 0.3m above the position of the quadrotor where the hover-button was pressed. Also make sure that the yaw angle is low before taking off because we noticed some agressive maneuvres when the yaw angle is too high which in combination with the ground effect might lead to a crash. 

If you find a policy to take-off and hover well, you can change the parameter `rlt.wn` to `4` which puts the controller into figure-eight tracking mode. The scale and interval ([s]) can be adjusted by the parameters `rlt.fes` and `rlt.fei` respectively. The height at which the figure-eight is executed is set by `rlt.target_z_fe`.

## Citing
When using our code/setup in an academic work please cite our publication using the following Bibtex citation:
```
@misc{eschmann2023learning,
      title={Learning to Fly in Seconds}, 
      author={Jonas Eschmann and Dario Albani and Giuseppe Loianno},
      year={2023},
      eprint={2311.13081},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```


