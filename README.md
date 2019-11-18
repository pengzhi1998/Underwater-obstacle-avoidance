# Underwater-obstacle-avoidance

Hello everyone, welcome to this repository. This project is mainly about underwater vehicles' obstacle avoidance with neural networks as well as the method based on the single beam distance detection. The project mainly refers to the following projects:https://github.com/xie9187/Monocular-Obstacle-Avoidance, https://github.com/XPFly1989/FCRN as well as https://github.com/iro-cp/FCRN-DepthPrediction.
## contents
[1. Introduction](https://github.com/2590477658/Underwater-obstacle-avoidance#1-introduction)

[2. Guide](https://github.com/2590477658/Underwater-obstacle-avoidance/blob/master/README.md#2-guide) 

3. Results and future improvements 

4. Acknowledgements 

### 1. Introduction
<div align=center><img width="600" height="450" src="https://github.com/2590477658/Underwater-obstacle-avoidance/raw/master/images/Bluerov_in_another_project.JPG" alt="The working Bluerov robot from another project(block assembly)"/></div>

Nowadays, the AUVs (autonomous underwater vehichles) are widely used in underwater projects (underwater environment detection, cultural relic salvage, underwater rescues and etc). And in order to improve their efficiency, a great sense of obstacle avoidance of the robots is indispensable. But because of the rather complex underwater light conditions including light attenuation, dimmer environment, reflection, refraction along with the more complicated kinematics situation including caparicious current and more resistance, it is much harder for the underwater robots to work well in the water. So we developed an ad-hoc methods to deal with that.

We are using the deep reinforcement neural networks as well as the single beam echo sounder to contorl our robot. In the first part, we implemented a FCRN (fully convolutional residual network) to predict RGBD from the front monocular camera. `To train the network, we used the NYU dataset, the images pairs from which have been preprocessed according to the underwater environment. In the second part, we applied the DDDQN to control the robot in "POSHOLD" mode with the topic of "/rc/override".` We trained this DDDQN in a well-designed Gazebo world. 
### 2. Guide
Please follow the guidance to train the neural networks and implement the experiments.
1. Clone the repository into a directory.
2. Download the NYU Depth Dataset V2 Labelled Dataset as well as the pre-trained TensorFlow weights as a .npy file for a part of the model from Laina et al. in to the folder of FCRN_train:
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat; http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy 
3. Open the create_underwater.m file, and change three parameters to fit the environment where you would like to test the robots' performance. Then run the code to process the NYU dataset. 
4. Run train.py to train the FCRN network. After 30-50 epochs, the performance is relatively good.
