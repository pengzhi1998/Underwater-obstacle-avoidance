# Underwater-obstacle-avoidance

Hello everyone, welcome to this repository. This project is mainly about underwater vehicles' obstacle avoidance with neural networks as well as the method based on the single beam distance detection. The project mainly refers to the following projects:https://github.com/xie9187/Monocular-Obstacle-Avoidance, https://github.com/XPFly1989/FCRN as well as https://github.com/iro-cp/FCRN-DepthPrediction.
## contents
   1. [Introduction](https://github.com/2590477658/Underwater-obstacle-avoidance#1-introduction)
   2. [ Guide](https://github.com/2590477658/Underwater-obstacle-avoidance/blob/master/README.md#2-guide)  
### 1. Introduction
Nowadays, the AUVs (autonomous underwater vehichles) are widely used in underwater projects (underwater environment detection, cultural relic salvage, underwater rescues and etc). And in order to improve their efficiency, a great sense of obstacle avoidance of the robots is indispensable. But because of the rather complex underwater light conditions including light attenuation, dimmer environment, reflection, refraction along with the more complicated kinematics situation including caparicious current and more resistance, it is much harder for the robots to work well underwater. So we developed an ad-hoc methods to deal with that.

In the first part, we implemented a FCRN (fully convolutional residual network) to predict RGBD from the front monocular camera. `To train the network, we used the NYU dataset, the images pairs from which have been preprocessed according to the underwater environment. In the second part, we applied the DDDQN to control the robot in "POSHOLD" mode with the topic of "/rc/override".` We trained this DDDQN in a well-designed Gazebo world. At last, we combined the two neural networks with the method based on the single beam echo sounder to make the robot, BlueROV2 to avoid obstacles. The senario is designed as follows:<br>
1. Set a goal point for the robot.<br>
2. The robot spins toward the goal, then moves forward.<br>
3. When the echo sounder detects obstacles right in front of it (the distance is less than .8 meters), the robot will be controlled by the neural networks based on the front monocular camera until it successfully avoid the object.<br>
4. Repeat 2 and 3 until it reaches the goal point.
### 2. Guide
1. Clone the repository into a directory.
2. Download the NYU Depth Dataset V2 Labelled Dataset as well as the pre-trained TensorFlow weights as a .npy file for a part of the model from Laina et al. into the folder of FCRN_train:
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat; http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy 
3. Open the `create_underwater.m` file, and change the three parameters (Red_attenuation, Green_attenuation along with Blue_attenuation) to fit the environment where you'd like to test the  performance. Then run it to process the NYU dataset. It will generate a "test.mat" in the same directory.
4. Run `train.py` in FCRN_train to train the FCRN network which is for the RGBD prediction. After 30 epochs, the performance is relatively good. The parameters of the model will be stored into the checkpoint.pth.tar.
5. Launch the designed world with the command in one terminal<br>
`roslaunch turtlebot3_gazebo turtlebot3_house.launch world_file:=/TO PATH/Underwater-obstacle-avoidance/DDDQN_train/turtlebot3_bighouse.world`<br>
Run the `DDDQN.py` at the same time in another terminal. The training begins. You could find the robot moves aimlessly at first, but starts to show the ability of avoiding the obstacles after around 200 episodes. The average reward for each 50 episodes could be seen from the graph drew by visdom.<br>
We set the max episode number to be 100000. Nevertheless, if the performance is good enough, it is fine to terminate the process. The networks will be saved into `online_with_noise.pth.tar` as well as `target_with_noise.pth.tar`.
6. Copy the `checkpoint.pth.tar` from FCRN_train and `online_with_noise.pth.tar` from DDDQN_train into the folder, Test_on_robots. Then test on the ground robots or underwater robots.<br>
`ping_echo_sounder.launch` and `pingmessage.py` are helping to open the single beam echo sounder to detect the distance between the robot and the object right in front of the echo sounder.
