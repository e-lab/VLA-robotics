# VLA robotics

Design of a class or visual-language-action models for use in robotics applications.


## Alex VLA

A custom VLA for [IHMC Alex robot](https://www.ihmc.us/news20251119/).

Inputs:
- a front head-mounted stereo camera with two 640x480 cameras
- a front head-mounted depth camera of 320x240
- proprioception of 32 motor control outputs
- a 512 vector word for commands

Output:
- predict 16 values in future: an action vector of the same size as motor control: 32 motor control outputs, 32 joint velocities, 32 joint speeds
- a 512 vector word for responses

We use a modified [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) architecture.