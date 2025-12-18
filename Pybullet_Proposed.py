import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

def setup_simulation():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    table_position = [0, 0, 0]
    table = p.loadURDF("table/table.urdf", table_position, globalScaling=1.0)
    panda_base_position = [-0.3, 0, 0.625]
    panda = p.loadURDF("franka_panda/panda.urdf", basePosition=panda_base_position, useFixedBase=True)
    block_position = [0.49, 0.08, 0.65]
    block = p.loadURDF("cube_small.urdf", block_position, globalScaling=1)
    p.changeDynamics(block, -1, mass= 2.8, lateralFriction=10, rollingFriction=0.2, spinningFriction=0.2)
    return panda, block

def get_joint_indices(robot):
    num_joints = p.getNumJoints(robot)
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED]
    return joint_indices

def move_arm_to_target(robot, target_position, target_orientation):
    joint_indices = get_joint_indices(robot)
    joint_positions = p.calculateInverseKinematics(robot, 11, target_position, target_orientation)
    for j, joint_index in enumerate(joint_indices[:7]):
        p.setJointMotorControl2(robot, joint_index, p.POSITION_CONTROL, joint_positions[j])

def grasp_object(robot, close):
    finger_joints = [9, 10]
    grip_position = 0.02 if close else 0.1
    for joint in finger_joints:
        p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL, grip_position, force=100)

def estimate_C_matrix(robot_id, joint_indices, q, q_dot, delta=1e-6):
    dof = len(joint_indices)
    C_matrix = np.zeros((dof, dof))

    # Step 1: baseline torque: τ = C(q, q̇)q̇ + G(q)
    tau_cg = np.array(p.calculateInverseDynamics(robot_id, q, q_dot, [0.0]*dof))
    tau_g = np.array(p.calculateInverseDynamics(robot_id, q, [0.0]*dof, [0.0]*dof))
    baseline = tau_cg - tau_g  # This is C(q, q̇) q̇

    # Step 2: finite difference for each joint velocity dimension
    for i in range(dof):
        dq_dot = np.array(q_dot)
        dq_dot[i] += delta

        tau_cg_perturbed = np.array(p.calculateInverseDynamics(robot_id, q, dq_dot.tolist(), [0.0]*dof))
        perturbed = tau_cg_perturbed - tau_g

        # Estimate i-th column of C matrix
        C_matrix[:, i] = (perturbed - baseline) / delta

    return C_matrix

class SimpleNN(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=40):
        super(SimpleNN, self).__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, 7)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 7)
        self.relu = nn.ReLU()

    def forward(self, x, params=None):
        x = x.reshape(x.shape[0], -1)
        if params is None:
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
        else:
            x = torch.tanh(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = torch.tanh(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x


def main():
    panda, block = setup_simulation()
    # ==================================================================== #
    #                           Pick up the block                          #
    # ==================================================================== #
    pickup_position = [0.6, 0.1, 0.75]
    pickup_orientation = p.getQuaternionFromEuler([0, np.pi, 0])
    grasp_object(panda, close=False) # Open the gripper
    for _ in range(500):
        p.stepSimulation()
        time.sleep(1 / 240)
    move_arm_to_target(panda, pickup_position, pickup_orientation) # Move to the top of the block
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1 / 240)
    pickup_position[2] = 0.63
    move_arm_to_target(panda, pickup_position, pickup_orientation) # Move to the center of the block
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1 / 240)
    grasp_object(panda, close=True) # Close the gripper
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1 / 240)
    pickup_position = [0.4, 0, 1.3]
    interpolated_positions = np.linspace([0.6, 0.1, 0.63], pickup_position, 300)
    for i in range(300): # Go to the initial position of the trajectory
        move_arm_to_target(panda, interpolated_positions[i], pickup_orientation)
        p.stepSimulation()
        time.sleep(1 / 240)
    for i in range(300): # Stay
        move_arm_to_target(panda, pickup_position, pickup_orientation)
        p.stepSimulation()
        time.sleep(1 / 240)
    # ==================================================================== #
    #                         Define the trajectory                        #
    # ==================================================================== #
    desired_data = pd.read_csv('Test_x_2.csv')
    desired_q = desired_data.iloc[:, :7].to_numpy()
    desired_qdot = desired_data.iloc[:, 7:14].to_numpy()
    desired_u = desired_data.iloc[:, 14:21].to_numpy()
    desired_q_ddot = np.zeros_like(desired_q)
    num_points = len(desired_data) - 1
    dt = 1 / 240.0
    Kp = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
    Kd = np.eye(7) * np.array([15, 50, 5, 30, 2, 2, 1])
    # Kd = 0.3 * np.eye(7) * np.array([15, 50, 5, 30, 2, 2, 1]) # Kd for model 1
    Kd_pd = np.array([15, 50, 5, 30, 2, 2, 1])
    num_joints = p.getNumJoints(panda)
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(panda, i)[2] != p.JOINT_FIXED]
    # for j in joint_indices:
    #     p.setJointMotorControl2(
    #         panda, j,
    #         controlMode=p.VELOCITY_CONTROL,
    #         force=0
    #     )
    p.setJointMotorControlArray(panda, joint_indices, controlMode=p.VELOCITY_CONTROL, forces=[0] * len(joint_indices))
    joint_indices = np.array([0, 1, 2, 3, 4, 5, 6])
    error = []
    error_dot = []
    a_history = []
    y_history = []
    predict_history = []
    error_predict = []
    model1 = SimpleNN()
    model1.load_state_dict(torch.load("model_2.pth", weights_only=True))
    a = np.array([-11.10268627,  13.25685114,   3.43531352, -10.41580058, -2.20127973,  -4.28272647,   0.07695898])
    alpha = 1
    beta = 30
    for i in range(num_points):
        joint_states = p.getJointStates(panda, joint_indices)
        q = np.array([state[0] for state in joint_states])
        q_dot = np.array([state[1] for state in joint_states])
        real_torque = np.array([state[3] for state in joint_states])
        zeros = np.zeros((1, 2))
        q_1 = np.hstack((q, zeros[0]))
        q_dot_1 = np.hstack((q_dot, zeros[0]))

        M_full = p.calculateMassMatrix(panda, q_1.tolist())
        M_full = np.array(M_full)
        M = M_full[:7, :7]

        MCG_full = p.calculateInverseDynamics(panda, q_1.tolist(), q_dot_1.tolist(), [0] * len(q_1))
        MCG = MCG_full[:7]
        G_full = p.calculateInverseDynamics(panda, q_1.tolist(), [0] * len(q_1), [0] * len(q_1))
        G = G_full[:7]
        G = np.array(G)
        C_estimated_full = estimate_C_matrix(panda, np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]), q_1.tolist(), q_dot_1.tolist())
        C_estimated_full = np.array(C_estimated_full)
        C_estimated = C_estimated_full[:7, :7]

        input = q
        input = np.hstack((input, q_dot))
        input = np.hstack((input, real_torque))
        input = torch.tensor(input.reshape(1,21), dtype=torch.float32)
        y_pred = model1(input)

        # y_pred = torch.tensor(np.array([[0.1001175, 0.27657417, -0.06205704, -0.92647467, 0.11273581, 0.18845111, -0.05525976]]), dtype=torch.float32)
        phi = y_pred.detach().numpy().T * np.eye(7)
        neural_torque = y_pred.detach().numpy().T * a.reshape(7, 1)

        e = q - desired_q[i]
        e_dot = q_dot - desired_qdot[i]
        e = e.reshape(7, 1)
        e_dot = e_dot.reshape(7, 1)

        y = MCG - real_torque
        y = y.reshape(7, 1)

        s = e_dot + beta * e # 7x1

        P = 10 * np.eye(7)
        R = 0.1 * np.eye(7)
        a_hat_update = P @ phi @ s - P @ phi @ np.linalg.inv(R) @ (phi @ a.reshape(7, 1) - y) #- 10 * a.reshape(7, 1)
        a = a.reshape(7, 1) + a_hat_update * dt

        pd_torques = Kp[0] * e + Kd_pd[0] * e_dot
        sm_torques =         s'm_torques = (G.r
                         + M @ (desired_q_ddot[i].reshape(7, 1) - beta * e_dot)
                         + C_estimated @ (desired_qdot[i].reshape(7, 1) - beta * e))
        final_torques = (G.reshape(7, 1) - np.array(neural_torque).reshape(7, 1) - Kd @ s
                         + M @ (desired_q_ddot[i].reshape(7, 1) - beta * e_dot)
                         + C_estimated @ (desired_qdot[i].reshape(7, 1) - beta * e))
        final_torques = final_torques.reshape(7)
        error.append(e)
        error_dot.append(e_dot)
        a_history.append(a)
        predict_history.append(phi @ a.reshape(7, 1))
        y_history.append(y)
        error_predict.append(phi @ a.reshape(7, 1) - y)
        grasp_object(panda, close=True)

        p.setJointMotorControlArray(
            bodyUniqueId=panda,
            jointIndices=joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=final_torques.tolist()
        )
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
    error = np.array(error)
    error_dot = np.array(error_dot)
    a_history = np.array(a_history)
    predict_history = np.array(predict_history)
    y_history = np.array(y_history)
    error_predict = np.array(error_predict)

    error_sum1 = np.sum(np.abs(error[:, 0])) / num_points
    error_sum2 = np.sum(np.abs(error[:, 1])) / num_points
    error_sum3 = np.sum(np.abs(error[:, 2])) / num_points
    error_sum4 = np.sum(np.abs(error[:, 3])) / num_points
    error_sum5 = np.sum(np.abs(error[:, 4])) / num_points
    error_sum6 = np.sum(np.abs(error[:, 5])) / num_points
    error_sum7 = np.sum(np.abs(error[:, 6])) / num_points
    print(error_sum1,error_sum2,error_sum3,error_sum4,error_sum5,error_sum6,error_sum7)

    # plt.figure(1)
    # plt.plot(a_history[:, 0], label='joint 1')
    # plt.plot(a_history[:, 1], label='joint 2')
    # plt.plot(a_history[:, 2], label='joint 3')
    # plt.plot(a_history[:, 3], label='joint 4')
    # plt.plot(a_history[:, 4], label='joint 5')
    # plt.plot(a_history[:, 5], label='joint 6')
    # plt.plot(a_history[:, 6], label='joint 7')
    # plt.title("a")
    # plt.legend()
    # plt.show()
    # plt.figure(1)
    # plt.plot(error[:, 0], error_dot[:, 0], label='joint 1')
    # plt.plot(error[:, 1], error_dot[:, 1], label='joint 2')
    # plt.plot(error[:, 2], error_dot[:, 2], label='joint 3')
    # plt.plot(error[:, 3], error_dot[:, 3], label='joint 4')
    # plt.plot(error[:, 4], error_dot[:, 4], label='joint 5')
    # plt.plot(error[:, 5], error_dot[:, 5], label='joint 6')
    # plt.plot(error[:, 6], error_dot[:, 6], label='joint 7')
    # plt.plot(np.linspace(-0.05,0.05,100), -beta*np.linspace(-0.05,0.05,100), linestyle='--', label='sliding surface')
    # plt.title("(e, e_dot)")
    # plt.legend()
    # plt.show()
    plt.figure(figsize=(13, 9))
    plt.plot(error[:, 0], linewidth=4)
    plt.plot(error[:, 1], linewidth=4)
    plt.plot(error[:, 2], linewidth=4)
    plt.plot(error[:, 3], linewidth=4)
    plt.plot(error[:, 4], linewidth=4)
    plt.plot(error[:, 5], linewidth=4)
    plt.plot(error[:, 6], linewidth=4)
    plt.xlabel('Time Step', fontsize='25', fontweight='bold')
    plt.ylabel('Angle Error (rad)', fontsize='25', fontweight='bold')
    plt.xticks([0, 125, 250, 375, 500], fontsize='25')
    plt.yticks([-0.1, 0, 0.1], fontsize='25')
    plt.title('Proposed Controller, Trajectory 2', fontsize='30', fontweight='bold')
    plt.xlim(0, 500)
    plt.grid()
    plt.show()
    # plt.figure(3)
    # plt.plot(predict_history[:, 0], label='joint 1')
    # plt.plot(predict_history[:, 1], label='joint 2')
    # plt.plot(predict_history[:, 2], label='joint 3')
    # plt.plot(predict_history[:, 3], label='joint 4')
    # plt.plot(predict_history[:, 4], label='joint 5')
    # plt.plot(predict_history[:, 5], label='joint 6')
    # plt.plot(predict_history[:, 6], label='joint 7')
    # plt.plot(y_history[:, 0], label='y_history 1')
    # plt.plot(y_history[:, 1], label='y_history 2')
    # plt.plot(y_history[:, 2], label='y_history 3')
    # plt.plot(y_history[:, 3], label='y_history 4')
    # plt.plot(y_history[:, 4], label='y_history 5')
    # plt.plot(y_history[:, 5], label='y_history 6')
    # plt.plot(y_history[:, 6], label='y_history 7')
    # plt.title("phi a and y")
    # plt.legend()
    # plt.show()
    # plt.figure(figsize=(13, 9))
    # plt.plot(error_predict[:, 0], label = 'Joint 1', linewidth=4)
    # plt.plot(error_predict[:, 1], label = 'Joint 2', linewidth=4)
    # plt.plot(error_predict[:, 2], label = 'Joint 3', linewidth=4)
    # plt.plot(error_predict[:, 3], label = 'Joint 4', linewidth=4)
    # plt.plot(error_predict[:, 4], label = 'Joint 5', linewidth=4)
    # plt.plot(error_predict[:, 5], label = 'Joint 6', linewidth=4)
    # plt.plot(error_predict[:, 6], label = 'Joint 7', linewidth=4)
    # plt.xlabel('Time Step', fontsize='25', fontweight='bold')
    # plt.ylabel('Prediction Error (Nm)', fontsize='25', fontweight='bold')
    # plt.xticks([0, 250, 500, 750, 1000], fontsize='25')
    # plt.yticks([-5, 0, 1], fontsize='25')
    # plt.title('Proposed Controller, Trajectory 1', fontsize='30', fontweight='bold')
    # # plt.legend(loc=9, bbox_to_anchor=(0.5, 1.15), ncol=7, frameon=False, fontsize='20')
    # plt.xlim(0, 1000)
    # # plt.ylim(-0.06, 0.07)
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    main()
