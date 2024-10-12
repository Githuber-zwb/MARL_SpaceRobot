from gym.envs.robotics import rotations
import numpy as np

def eulerpotion2T(euler, point):
    mat = rotations.euler2mat(euler)
    T = np.zeros([4,4])
    T[0][:3] = mat[0]
    T[1][:3] = mat[1]
    T[2][:3] = mat[2]
    T[0][3] = point[0]
    T[1][3] = point[1]
    T[2][3] = point[2]
    T[3][3] = 1

    return T

if __name__ == "__main__":
    goal_pos3 = np.array([0.27814835,  0.29937622,  5.246062])
    goal_rot3 = np.array([-1.5688201,   0.489479 + 1.5707963,    0.06324279])
    # goal_rot3 = np.array([-1.5688201 + 1.5707963,   0.489479 - 1.5707963,    0.06324279])
    pos = np.array([0.361 + 0.2, 0, 0])
    rot = np.array([0, 0, 0])
    T_PO = eulerpotion2T(goal_rot3, goal_pos3)
    T_PR = eulerpotion2T(rot, pos)
    T_RO = T_PO @ np.linalg.inv(T_PR)  
    print(T_RO)
    print("Euler: ", rotations.mat2euler(T_RO[:3, :3]))
    print("Pos: ", T_RO[:3, -1])
