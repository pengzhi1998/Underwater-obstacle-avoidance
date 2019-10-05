# Underwater-obstacle-avoidance

Hello everyone, welcome to this repository. This project is mainly on underwater vehicles obstacle avoidance with the use of two different neural networks. This work mainly refers to the following projects:https://github.com/xie9187/Monocular-Obstacle-Avoidance, https://github.com/XPFly1989/FCRN as well as https://github.com/iro-cp/FCRN-DepthPrediction.
## contents
1. Introduction
2. Guide
3. Results and future improvements
4. Acknowledgements
### 1. Introduction
Nowadays, the AUVs (autonomous underwater vehichles) are widely used in underwater projects (underwater environment detection, cultural relic salvage, underwater rescues and etc). And in order to make them work smoothly, a great sense of obstacle avoidance is necessary. But because of the more complex underwater light conditions including light attenuation, dimmer environment, reflection, refraction along with the more complicated kinematics situation including caparicious current and bigger resistance, it is much harder for the underwater robots to work well in the water. So we developed an ad-hoc methods to deal that.

We are using the deep reinforcement neural networks as well as the single beam echo sounder to contorl our robot. In the first part, we implemented a FCRN (fully convolutional residual network) to predict RGBD from the front camera. To train the network, we used the NYU dataset, but firstly `processed all the images from the dataset to make them equip with the features of underwater environment`. In the second part, we applied the DDDQN to control the robot in `"POSHOLD" mode with the topic of "/rc/override"`. We trained this DDDQN in Gazebo world. 
### 2. Guide

