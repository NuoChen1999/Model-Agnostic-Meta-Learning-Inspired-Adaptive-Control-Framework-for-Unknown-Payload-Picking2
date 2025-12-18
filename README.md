# Model-Agnostic-Meta-Learning-Inspired-Adaptive-Control-Framework-for-Unknown-Payload-Picking
This repo aims to provide a new meta-Learning method to Franka robot in Pybullet. The proposed meta-learning method is inspired by Model-Agnostic Meta-Learning. In this study, 10 objects of different weights are used for training, and the manipulator is tested with unknown and new payload. Simulations are conducted to demonstrate the effectiveness of the proposed training and control framework. 

"MetaLearning_Proposed_Train.ipynb" trains the mass-dependent parameter with the proposed method.

"Pybullet_CollectData.py" collects training data and the desired position data.

"Pybullet_Proposed.py" runs the trained network and the proposed controller in Pybullet.

Please pay attention to the file path.

## Framework
![demo image](images/Framework.png)

## t-SNE of Parameter
![demo image](images/t-SNE.png)

## Controller Result
![demo image](images/Controller_Result.png)

If you use this code, please cite our paper:
```bibtex
@INPROCEEDINGS{11221857,
  author={Chen, Nuo and Pan, Ya-Jun},
  booktitle={IECON 2025 – 51st Annual Conference of the IEEE Industrial Electronics Society}, 
  title={Model-Agnostic Meta-Learning Inspired Adaptive Control Framework for Unknown Payload Picking}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Metalearning;Adaptation models;Training data;Machine learning;Manipulators;Real-time systems;Trajectory;Adaptive control;Payloads;Meta-Learning;Adaptive Control},
  doi={10.1109/IECON58223.2025.11221857}}
