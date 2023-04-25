import gymnasium as gym
from gymnasium import spaces
import pybullet
import pybullet_data as pd
from pybullet_utils import bullet_client as bc
import time
import numpy as np
import math
import random

class FoosballEnv(gym.Env):

    def __init__(self, render_mode=None, just_goal=False):
        self.metadata = {"render_modes": ["human", "ascii"], "render_fps": 60}
        self._just_goal = just_goal
        self._table_length = 46 * 25.4 / 1000     # 46 inches in m
        self._table_width  = 26.75 * 25.4 / 1000  # 26.75 inches in m

        # Define observation space for gymnasium environment
        self.observation_space = spaces.Dict(
            {
                "ball_pos": spaces.Box(low=np.array([-self._table_length/2, -self._table_width/2]), high=np.array([self._table_length/2, self._table_width/2]), dtype=float),
                "ball_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=float),
                "t1_pos": spaces.Box(0 , 1, shape=(8,), dtype=float),
                "t1_vel": spaces.Box(-1, 1, shape=(8,), dtype=float),
                "t2_pos": spaces.Box(0 , 1, shape=(8,), dtype=float),
                "t2_vel": spaces.Box(-1, 1, shape=(8,), dtype=float),
            }
        )
        
        # Define action space for gymnasium environment
        self.action_space = spaces.Box(0, 1, shape=(16,), dtype=float)

        # Check for valid rendering modes
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Deal with pybullet client
        if (self.render_mode == "human"):
            self._p = bc.BulletClient(connection_mode=pybullet.GUI)
            self._p.resetDebugVisualizerCamera(1.5, 0, -80, (0,0,0))
        else:
            self._p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self._p.setAdditionalSearchPath(pd.getDataPath())
        
        # Load objects in scene
        table, ball, goal1, goal2 = self._load_urdf()
        
        # Store id's of objects in scene
        self._table = table
        self._ball  = ball
        self._goal1 = goal1
        self._goal2 = goal2
        
        # Enable gravity
        self._p.setGravity(0, 0, -9.81)

        # List of all prismatic and rotational joint ids
        prismatic_list = [0, 5, 10, 14, 18, 25, 32, 37]
        rotation_list = [1, 6, 11, 15, 19, 26, 33, 38]
        
        # Store team 1 joint ids
        self._t1_prismatic = prismatic_list[::2]
        self._t1_rotation = rotation_list[::2]

        # Store team 2 joint ids
        self._t2_prismatic = prismatic_list[1::2]
        self._t2_rotation = rotation_list[1::2]
        
        # Possible value ranges (for setpoints and value normalization)
        self._prismatic_maxes = [.18161, .35306, .11176, .22606]
        self._rotation_maxes = [math.pi, math.pi, math.pi, math.pi]
        self._prismatic_mins = [0, 0, 0, 0]
        self._rotation_mins = [-math.pi, -math.pi, -math.pi, -math.pi]

        self._prismatic_list = prismatic_list
        self._rotation_list = rotation_list

    def reset(self, seed=None, randomStart=True):
        
        super().reset(seed=seed)

        # With random start, ball is in random location and players are in random positions
        if randomStart:
            self._random_start()
        
        # Normal start is like in standard foosball where ball comes from the side
        else:
            self._normal_start() 

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        if self.render_mode == "ascii":
            self._render_ascii(observation, info, "N/A")

        return observation, info

    ## Handles state propagation and applying actions
    def step(self, action):
        # Convert action array to controller setpoints
        self._handle_action(action)
        
        # Step forward 1/60th of a second (4 frames) and see if a score occurs
        score = 0
        terminated = False
        for i in range(4):
            self._p.stepSimulation()
            new_score = self._check_score()
            if score == 0 and new_score > 0:
                score = new_score
                terminated = True
            if self.render_mode == "human":
                self._render_frame()

        
        # Get state and reward
        observation = self._get_obs()
        info = self._get_info()
        reward = self._get_reward(info, score)

        # Handle rendering
        if self.render_mode == "ascii":
            self._render_ascii(observation, info, reward)

        if (info["ball_z"] < -1):
            terminated = True

        return observation, reward, terminated, False, info

    ## TODO: Figure out what to do here (if anything)
    def render(self):
        return -1
    
    def close(self):
        self._p.disconnect()

    # Private method to handle loading in URDF objects into scene
    def _load_urdf(self):
        # Goal positions
        goal1Position = (-25.4/1000-1219.2/2000, 0, 25.4/1000)
        goal2Position = ((1219.2)/2000, 0, 25.4/1000)
        
        # Load objects and make sure no errors
        table = self._p.loadURDF("TableTest.urdf")
        assert table >= 0, "URDF error with table"

        ball  = self._p.loadURDF("ball.urdf", basePosition=(0, .3, .01))
        assert ball >= 0, "URDF error with ball"

        goal1 = self._p.loadURDF("Goal.urdf", basePosition=goal1Position)
        assert goal1 >= 0, "URDF error with goal1"
        self._p.changeVisualShape(goal1, -1, rgbaColor=(1,1,1,1))

        goal2 = self._p.loadURDF("Goal.urdf", basePosition=goal2Position)
        assert goal2 >= 0, "URDF error with goal2"
        self._p.changeVisualShape(goal2, -1, rgbaColor=(.1,.1,.1,1))

        return table, ball, goal1, goal2
    
    ## Private method to get ball state (pos and vel)
    def _get_ball_obs(self, z=False):
        ball_pos = self._p.getBasePositionAndOrientation(self._ball)
        ball_vel = self._p.getBaseVelocity(self._ball)
        
        ball_pos = ball_pos[0]
       
        if z:
            ball_z = ball_pos[2]
        
        ball_pos = ball_pos[:2]
        ball_vel = ball_vel[0]
        ball_vel = ball_vel[:2]
        
        if z:
            return ball_pos, ball_vel, ball_z

        return ball_pos, ball_vel

    ## Private method to get system state (ball, team1, team2)
    def _get_obs(self):
        # Get ball state
        ball_pos, ball_vel = self._get_ball_obs()
        
        # Get raw joint states for team 1 and 2
        t1_pris = self._p.getJointStates(self._table, self._t1_prismatic)
        t1_rot  = self._p.getJointStates(self._table, self._t1_rotation)

        t2_pris = self._p.getJointStates(self._table, self._t2_prismatic)
        t2_rot  = self._p.getJointStates(self._table, self._t2_rotation)

        # Process states and put into position and velocity vectors for each team (prisPos + rotPos)
        t1_pos = [t1_pris[x][0] for x in range(4)] + [t1_rot[x][0] for x in range(4)]
        t1_vel = [t1_pris[x][1] for x in range(4)] + [t1_rot[x][1] for x in range(4)]
        t2_pos = [t2_pris[x][0] for x in range(4)] + [t2_rot[x][0] for x in range(4)]
        t2_vel = [t2_pris[x][1] for x in range(4)] + [t2_rot[x][1] for x in range(4)]
        
        # Map values from normal units to [0,1] for joint positions and [-1, 1] for joint velocities
        t1_pos_mapped = self._map(t1_pos, self._prismatic_mins + self._rotation_mins, self._prismatic_maxes + self._rotation_maxes, np.zeros(8), np.ones(8))
        t1_vel_mapped = self._map(t1_vel, [-x for x in self._prismatic_maxes + self._rotation_maxes], self._prismatic_maxes + self._rotation_maxes, np.full(8, -1), np.ones(8))

        t2_pos_mapped = self._map(t2_pos, self._prismatic_mins + self._rotation_mins, self._prismatic_maxes + self._rotation_maxes, np.zeros(8), np.ones(8))
        t2_vel_mapped = self._map(t2_vel, [-x for x in self._prismatic_maxes + self._rotation_maxes], self._prismatic_maxes + self._rotation_maxes, np.full(8, -1), np.ones(8))

        return {
                "ball_pos": np.asarray(ball_pos, dtype=float),
                "ball_vel": np.asarray(ball_vel, dtype=float),
                "t1_pos": np.asarray(t1_pos_mapped, dtype=float),
                "t1_vel": np.asarray(t1_vel_mapped, dtype=float),
                "t2_pos": np.asarray(t2_pos_mapped, dtype=float),
                "t2_vel": np.asarray(t2_vel_mapped, dtype=float)
        }

    ## Private method to map a value from one range to a new range
    def _map(self, values, old_mins, old_maxes, new_mins, new_maxes):
        
        new_values = []
        for i in range(len(values)):
            new_values.append((values[i]-old_mins[i])*(new_maxes[i]-new_mins[i])/(old_maxes[i]-old_mins[i]) + new_mins[i])
        
        return new_values
    
    ## Private method to get info other than state about system
    def _get_info(self, goalx=1219.2/2000):
        # Get ball info
        ball_pos, ball_vel, ball_z = self._get_ball_obs(True)
        
        # Convert to np arrays
        ball_pos_np = np.asarray(ball_pos)
        ball_vel_np = np.asarray(ball_vel)
        
        # Create np arrays with goal positions
        goal1 = np.array([-goalx, 0])
        goal2 = np.array([goalx, 0])
        
        # Find position of goal relative to ball
        goal1_vec = goal1 - ball_pos_np
        goal2_vec = goal2 - ball_pos_np
        
        # Find distance between goal and ball
        goal1_dist = np.linalg.norm(goal1_vec)
        goal2_dist = np.linalg.norm(goal2_vec)
        
        # Find how much velocity is pointing towards goals (faster is better)
        goal1_vel = np.dot(ball_vel, goal1_vec)/goal1_dist
        goal2_vel = np.dot(ball_vel, goal2_vec)/goal2_dist

        return {
            "d1": goal1_dist,
            "d2": goal2_dist,
            "v1": goal1_vel, # Not actually velocity, but (p_goal/ball . v_ball)/|p_goal/ball|
            "v2": goal2_vel, # Not actually velocity, but (p_goal/ball . v_ball)/|p_goal/ball|
            "ball_z": ball_z
        }

    ## Private method to reset simulation to random starting location
    def _random_start(self):
        
        # Get random positions for table
        t1_pos = self.observation_space["t1_pos"].sample()
        t2_pos = self.observation_space["t2_pos"].sample()

        t1_pos_mapped = self._map(t1_pos, np.zeros(8), np.ones(8), self._prismatic_mins + self._rotation_mins, self._prismatic_maxes + self._rotation_maxes)
        t2_pos_mapped = self._map(t2_pos, np.zeros(8), np.ones(8), self._prismatic_mins + self._rotation_mins, self._prismatic_maxes + self._rotation_maxes)
        
        # Put table in random position
        for i in range(4):
            self._p.resetJointState(self._table, self._t1_prismatic[i], t1_pos_mapped[i], 0)
            self._p.resetJointState(self._table, self._t1_rotation[i], t1_pos_mapped[i+4], 0)

            self._p.resetJointState(self._table, self._t2_prismatic[i], t2_pos_mapped[i], 0)
            self._p.resetJointState(self._table, self._t2_rotation[i], t2_pos_mapped[i+4], 0)

        # Get random state for ball
        ball_pos = self.observation_space["ball_pos"].sample()
        ball_vel = self.observation_space["ball_vel"].sample()/2
        
        # Put ball in random position
        self._p.resetBasePositionAndOrientation(self._ball, ball_pos.tolist() + [.025], (0,0,0,1))
        self._p.resetBaseVelocity(self._ball, ball_vel.tolist() + [0], (0,0,0,1))
        
        # Check if ball is colliding with something at start, and if it is, then generate a new random position
        self._p.performCollisionDetection()
        contact = self._p.getContactPoints(self._ball)
        while contact:
            ball_pos = self.observation_space["ball_pos"].sample()
            self._p.resetBasePositionAndOrientation(self._ball, ball_pos.tolist() + [.1], (0,0,0,1))

            self._p.performCollisionDetection()
            contact = self._p.getContactPoints(self._ball)
        
    def set_state(self, state):
        
        # Get random positions for table
        t1_pos = state["t1_pos"]
        t2_pos = state["t2_pos"]

        t1_vel = state["t1_vel"]
        t2_vel = state["t2_vel"]

        ball_pos = state["ball_pos"]
        ball_vel = state["ball_vel"]

        t1_pos_mapped = self._map(t1_pos, np.zeros(8), np.ones(8), self._prismatic_mins + self._rotation_mins, self._prismatic_maxes + self._rotation_maxes)
        t2_pos_mapped = self._map(t2_pos, np.zeros(8), np.ones(8), self._prismatic_mins + self._rotation_mins, self._prismatic_maxes + self._rotation_maxes)
        
        t1_vel_mapped = self._map(t1_vel, np.full(8,-1), np.ones(8), self._prismatic_mins + self._rotation_mins, self._prismatic_maxes + self._rotation_maxes)
        t2_vel_mapped = self._map(t2_vel, np.full(8,-1), np.ones(8), self._prismatic_mins + self._rotation_mins, self._prismatic_maxes + self._rotation_maxes)

        for i in range(4):
            self._p.resetJointState(self._table, self._t1_prismatic[i], t1_pos_mapped[i], t1_vel_mapped[i])
            self._p.resetJointState(self._table, self._t1_rotation[i], t1_pos_mapped[i+4], t1_vel_mapped[i+4])

            self._p.resetJointState(self._table, self._t2_prismatic[i], t2_pos_mapped[i], t2_vel_mapped[i])
            self._p.resetJointState(self._table, self._t2_rotation[i], t2_pos_mapped[i+4], t2_vel_mapped[i+4])

        # Put ball in random position
        self._p.resetBasePositionAndOrientation(self._ball, ball_pos.tolist() + [.025], (0,0,0,1))
        self._p.resetBaseVelocity(self._ball, ball_vel.tolist() + [0], (0,0,0,1))
        

    def _normal_start(self):
        angle = random.gauss(0, 10) * math.pi / 180

        position = [0, .3, .1]
        velocity = [.5*math.sin(angle), .5*math.cos(angle), 0]
        

        self._p.resetBasePositionAndOrientation(self._ball, position, (0,0,0,1))
        self._p.resetBaseVelocity(self._ball, velocity, (0,0,0,1))
        
        for joint in self._prismatic_list + self._rotation_list:
            self._p.resetJointState(self._table, joint, 0, 0)


    ## Private method to calculate current reward given info
    def _get_reward(self, info, scored, gains=(10, 10, 1, 1, 1000)):

        # Combine info and gains to calculate initial reward
        t1_reward = -gains[0] * info["d1"] + gains[1] * info["d2"] + gains[2] * info["v1"] - gains[3] * info["v2"]
        
        if self._just_goal:
            t1_reward = 0
        # Scoring means that we want a big reward!
        if scored == 1:
            t1_reward += gains[4]
        if scored == 2:
            t1_reward -= gains[4]
        
        # reward_t1 = -reward_t2
        return {"t1_reward": t1_reward, "t2_reward": -t1_reward}

    def _handle_action(self, action):
        t1_pris = action[:4]
        t1_rot  = action[4:8]
        t2_pris = action[8:12]
        t2_rot  = action[12:]

        t1_pris_setpoint = self._map(t1_pris, np.zeros(4), np.ones(4), self._prismatic_mins, self._prismatic_maxes)
        t1_rot_setpoint = self._map(t1_rot, np.zeros(4), np.ones(4), self._rotation_mins, self._rotation_maxes)

        t2_pris_setpoint = self._map(t2_pris, np.zeros(4), np.ones(4), self._prismatic_mins, self._prismatic_maxes)
        t2_rot_setpoint = self._map(t2_rot, np.zeros(4), np.ones(4), self._rotation_mins, self._rotation_maxes)

        self._p.setJointMotorControlArray(bodyUniqueId=self._table, jointIndices=self._t1_prismatic, controlMode=pybullet.POSITION_CONTROL, targetPositions=t1_pris_setpoint)
        self._p.setJointMotorControlArray(bodyUniqueId=self._table, jointIndices=self._t1_rotation, controlMode=pybullet.POSITION_CONTROL, targetPositions=t1_rot_setpoint)
        
        self._p.setJointMotorControlArray(bodyUniqueId=self._table, jointIndices=self._t2_prismatic, controlMode=pybullet.POSITION_CONTROL, targetPositions=t2_pris_setpoint)
        self._p.setJointMotorControlArray(bodyUniqueId=self._table, jointIndices=self._t2_rotation, controlMode=pybullet.POSITION_CONTROL, targetPositions=t2_rot_setpoint)

    def _check_score(self):
        score1 = self._p.getContactPoints(bodyA=self._goal1)
        score2 = self._p.getContactPoints(bodyA=self._goal2)
        
        if score1:
            return 1

        if score2:
            return 2

        return 0


    ## Only occurs if render mode is human, and p.GUI is set
    def _render_frame(self):
        time.sleep(1/240)
    
    ## Only occurs if render mode is ascii, meaning p.DIRECT is set
    def _render_ascii(self, observation, info, reward):
        print("Observation: ")
        print(observation)
        print("Info: ")
        print(info)
        print("Reward: ")
        print(reward)
