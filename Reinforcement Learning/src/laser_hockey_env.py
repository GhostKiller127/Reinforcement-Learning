import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys, math
import numpy as np
import matplotlib.pyplot as plt

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding, EzPickle

import pyglet
from pyglet import gl
from pyglet.window import key

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

RECORDING = True
# RECORDING = False

FPS = 60
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well (Don't touch)

VIEWPORT_W = 600
VIEWPORT_H = 400
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CENTER_X = W/2
CENTER_Y = H/2

RACKETPOLY = [(-5,20),(+5,20),(+5,-20),(-5,-20),(-13,-10),(-15,0),(-13,10)]

FORCEMULIPLAYER = 5000
TORQUEMULTIPLAYER = 200

scaling = [ 1.0,  1.0 , 3.14, 4.0, 4.0, 2.0,  
            1.0,  1.0,  3.14, 4.0, 4.0, 2.0,  
            2.0, 2.0, 10.0, 10.0]


def dist_positions(p1,p2):
    return np.sqrt(np.sum(np.asarray(p1-p2)**2))

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.goal_player_2 == contact.fixtureA.body or self.env.goal_player_2 == contact.fixtureB.body:
            if self.env.puck == contact.fixtureA.body or self.env.puck == contact.fixtureB.body:
                # print('Player 1 scored')
                self.env.done = True
                self.env.winner = 1
        if self.env.goal_player_1 == contact.fixtureA.body or self.env.goal_player_1 == contact.fixtureB.body:
            if self.env.puck == contact.fixtureA.body or self.env.puck == contact.fixtureB.body:
                # print('Player 2 scored')
                self.env.done = True
                self.env.winner = -1
        if (contact.fixtureA.body == self.env.player1 or contact.fixtureB.body == self.env.player1) \
           and (contact.fixtureA.body == self.env.puck or contact.fixtureB.body == self.env.puck):
            # print("player 1 contacted the puck")
            self.env.player1_contact_puck = True

    def EndContact(self, contact):
        pass

class LaserHockeyEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    continuous = False
    NORMAL = 0
    TRAIN_SHOOTING = 1
    TRAIN_DEFENSE = 2

    def __init__(self, mode = NORMAL):
        """ mode is the game mode: NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE,
        it can be changed later using the reset function
        """
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.mode = mode

        self.world = Box2D.b2World([0,0])
        self.player1 = None
        self.player2 = None
        self.puck = None
        self.goal_player_1 = None
        self.goal_player_2 = None
        self.world_objects = []
        self.drawlist = []
        self.done = False
        self.truncated = False
        self.winner = 0
        self.one_starts = True # player one starts the game (alternating)

        self.max_puck_speed = 20

        self.timeStep = 1.0 / FPS
        self.time = 0
        self.max_timesteps = 600

        self.closest_to_goal_dist = 1000

        # x pos player one
        # y pos player one
        # angle player one
        # x vel player one
        # y vel player one
        # angular vel player one
        # x player two
        # y player two
        # angle player two
        # y vel player two
        # y vel player two
        # angular vel player two
        # x pos puck
        # y pos puck
        # x vel puck
        # y vel puck
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)

        # linear force in (x,y)-direction and torque
        self.action_space = spaces.Box(-1, +1, (3*2,), dtype=np.float32)

        # see discrete_to_continous_action()
        self.discrete_action_space = spaces.Discrete(7)

        self.reset(self.one_starts)

    def key_press(self, symbol, mod):
        if symbol == key.ESCAPE: self.done = True    
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._seed = seed
        return [seed]

    def _destroy(self):
        if self.player1 is None: return
        self.world.contactListener = None
        self.world.DestroyBody(self.player1)
        self.player1 = None
        self.world.DestroyBody(self.player2)
        self.player2 = None
        self.world.DestroyBody(self.puck)
        self.puck = None
        self.world.DestroyBody(self.goal_player_1)
        self.goal_player_1 = None
        self.world.DestroyBody(self.goal_player_2)
        self.goal_player_2 = None
        for obj in self.world_objects:
            self.world.DestroyBody(obj)
        self.world_objects = []
        self.drawlist = []

    def r_uniform(self,mini,maxi):
        return self.np_random.uniform(mini,maxi,1)[0]


    def _create_player(self, position, color, is_player_two):
        player = self.world.CreateDynamicBody(
            position=position,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[ (-x/SCALE if is_player_two else x/SCALE , y/SCALE)
                                              for x,y in RACKETPOLY ]),
                density=200.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x011,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        player.color1 = color
        player.color2 = color
        player.linearDamping = 1.0
        player.anguarDamping = 1.0

        return player

    def _create_puck(self, position, color):
        puck = self.world.CreateDynamicBody(
            position=position,
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=10/SCALE, pos=(0,0)),
                density=10.0,
                friction=0.1,
                categoryBits=0x001,
                maskBits=0x0010,  # collide only with ground
                restitution=0.95)  # 0.99 bouncy
        )
        puck.color1 = color
        puck.color2 = color
        puck.linearDamping = 0.05

        return puck

    def _create_world(self):
        def _create_wall(position, poly):
            wall = self.world.CreateStaticBody(
                position=position,
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in poly ]),
                    density=0,
                    friction=0.1,
                    categoryBits=0x011,
                    maskBits=0x0011)
            )
            wall.color1 = (0,0,0)
            wall.color2 = (0,0,0)

            return wall

        def _create_decoration():
            objs = []
            objs.append(self.world.CreateStaticBody(
                position=(W/2, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=100/SCALE, pos=(0,0)),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (0.8,0.8,0.8)
            objs[-1].color2 = (0.8,0.8,0.8)

            objs.append(self.world.CreateStaticBody(
                position=(W/2, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=100/SCALE, pos=(0,0)),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (0.8,0.8,0.8)
            objs[-1].color2 = (0.8,0.8,0.8)

            objs.append(self.world.CreateStaticBody(
                position=(W/2-250/SCALE, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=70/SCALE, pos=(0,0)),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (255./255,204./255,191./255)
            objs[-1].color2 = (255./255,204./255,191./255)

            poly = [(0,100),(100,100),(100,-100),(0,-100)]
            objs.append(self.world.CreateStaticBody(
                position=(W/2-240/SCALE, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[ (x/SCALE, y/SCALE) for x, y in poly]),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (1,1,1)
            objs[-1].color2 = (1,1,1)

            objs.append(self.world.CreateStaticBody(
                position=(W/2+250/SCALE, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=70/SCALE, pos=(0,0)),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (255./255,204./255,191./255)
            objs[-1].color2 = (255./255,204./255,191./255)

            poly = [(100,100),(0,100),(0,-100),(100,-100)]
            objs.append(self.world.CreateStaticBody(
                position=(W/2+140/SCALE, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[ (x/SCALE, y/SCALE) for x, y in poly]),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (1,1,1)
            objs[-1].color2 = (1,1,1)

            return objs

        self.world_objects = []

        self.world_objects.extend(_create_decoration())

        poly = [(-250,5), (-250,-5), (250,-5), (250,5)]
        self.world_objects.append(_create_wall((W/2,H - 1), poly))
        self.world_objects.append(_create_wall((W/2,1), poly))

        poly = [(-5,50), (5,50), (5,-50), (-5,-50)]
        self.world_objects.append(_create_wall((W/2-245/SCALE,H-52.5/SCALE-1), poly))
        self.world_objects.append(_create_wall((W/2-245/SCALE,52.5/SCALE+1), poly))

        self.world_objects.append(_create_wall((W/2+245/SCALE,H-52.5/SCALE-1), poly))
        self.world_objects.append(_create_wall((W/2+245/SCALE,52.5/SCALE+1), poly))

        self.drawlist.extend(self.world_objects)

    def _create_goal(self, position, poly):
        goal = self.world.CreateStaticBody(
            position=position,
            angle=0.0,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in poly ]),
                    density=0,
                    friction=0.1,
                    categoryBits=0x0010,
                    maskBits=0x001,
                    isSensor=True),
                fixtureDef(
                    shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in poly ]),
                    density=0,
                    friction=0.1,
                    categoryBits=0x010,
                    maskBits=0x0010)]
        )
        goal.color1 = (1,1,1)
        goal.color2 = (1,1,1)

        return goal



    def reset(self, one_starting = None, mode = None):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.done = False
        self.truncated = False
        self.winner = 0
        self.prev_shaping = None
        self.time = 0
        if mode is not None and mode in [self.NORMAL, self.TRAIN_SHOOTING, self.TRAIN_DEFENSE]:
            self.mode = mode

        if self.mode == self.NORMAL:
            if one_starting is not None:
                self.one_starts = one_starting
            else:
                self.one_starts = not self.one_starts
        self.closest_to_goal_dist = 1000
        self.player1_contact_puck = False

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create world
        self._create_world()

        poly = [(-5,66), (5,66), (5,-66), (-5,-66)]
        self.goal_player_1 = self._create_goal((W/2-245/SCALE,H/2), poly)
        self.goal_player_2 = self._create_goal((W/2+245/SCALE,H/2), poly)

        # Create players
        self.player1 = self._create_player(
            (W / 5 , H / 2),
            (1,0,0),
            False
        )
        if self.mode != self.NORMAL:
            self.player2 = self._create_player(
                (4* W / 5 + self.r_uniform(-W / 3, W/6), H/2 + self.r_uniform(-H/4, H/4)),
                (0,0,1),
                True
            )
        else:
            self.player2 = self._create_player(
                (4 * W / 5, H / 2),
                (0,0,1),
                True
            )
        if self.mode == self.NORMAL or self.mode == self.TRAIN_SHOOTING:
            if self.one_starts or self.mode == self.TRAIN_SHOOTING:
                self.puck = self._create_puck( (W / 2 - self.r_uniform(H/8, H/4),
                                               H / 2 + self.r_uniform(-H/8, H/8)), (0,0,0) )
            else:
                self.puck = self._create_puck( (W / 2 + self.r_uniform(H/8, H/4),
                                               H / 2 + self.r_uniform(-H/8, H/8)), (0,0,0) )
        elif self.mode == self.TRAIN_DEFENSE:
            self.puck = self._create_puck((W / 2 + self.r_uniform(0, W/3),
                                           H / 2 + 0.9*self.r_uniform(-H/2, H/2)),  (0,0,0) )
            force = -(self.puck.position - (0, H/2 + self.r_uniform(-66/SCALE, 66/SCALE)))*self.puck.mass/self.timeStep
            self.puck.ApplyForceToCenter(force, True)


        self.drawlist.extend([self.player1, self.player2, self.puck])

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _apply_action_with_max_speed(self, player, action, max_speed, is_player_one):
        velocity = np.asarray(player.linearVelocity)
        speed = np.sqrt(np.sum((velocity)**2))
        if is_player_one:
            force = action * FORCEMULIPLAYER
        else:
            force = -action * FORCEMULIPLAYER

        if (is_player_one and player.position[0] > CENTER_X) \
           or (not is_player_one and player.position[0] < CENTER_X): # bounce at the center line
            force[0] = 0
            if is_player_one:
                if player.linearVelocity[0] > 0:
                    force[0] = -2*player.linearVelocity[0] * player.mass / self.timeStep
                force[0] += -1*(player.position[0] - CENTER_X) * player.linearVelocity[0] * player.mass / self.timeStep
            else:
                if player.linearVelocity[0] < 0:
                    force[0] = -2*player.linearVelocity[0] * player.mass / self.timeStep
                force[0] += 1*(player.position[0] - CENTER_X) * player.linearVelocity[0] * player.mass / self.timeStep

            player.linearDamping = 10.0
            player.ApplyForceToCenter(force.tolist(), True)
            return

        if (speed < max_speed):
            player.linearDamping = 1.0
            player.ApplyForceToCenter(force.tolist(), True)
        else:
            player.linearDamping = 10.0
            deltaVelocity = self.timeStep * force / player.mass
            if (np.sqrt(np.sum((velocity + deltaVelocity)**2)) < speed):
                player.ApplyForceToCenter(force.tolist(), True)
            else:
                pass
            
    
    def obs_norm(self, obs):
        obs_norm = np.copy(obs)
        obs_norm[0] /= 7.5
        obs_norm[1] /= 5
        obs_norm[2] = (obs_norm[2] % (2*np.pi))
        if obs_norm[2] > np.pi:
            obs_norm[2] = obs_norm[2] - 2*np.pi
        obs_norm[2] /= np.pi
        obs_norm[3] /= 10
        obs_norm[4] /= 10
        obs_norm[5] /= 15
        obs_norm[6] /= 7.5
        obs_norm[7] /= 5
        obs_norm[8] = (obs_norm[8] % (2*np.pi))
        if obs_norm[8] > np.pi:
            obs_norm[8] = obs_norm[8] - 2*np.pi
        obs_norm[8] /= np.pi
        obs_norm[9] /= 10
        obs_norm[10] /= 10
        obs_norm[11] /= 15
        obs_norm[12] /= 7.5
        obs_norm[13] /= 5
        obs_norm[14] /= 20
        obs_norm[15] /= 20
        return obs_norm         
            
            
    def _get_obs(self):
        obs = np.hstack([
            self.player1.position-[CENTER_X,CENTER_Y],
            [self.player1.angle],
            self.player1.linearVelocity,
            [self.player1.angularVelocity],
            self.player2.position-[CENTER_X,CENTER_Y],
            [self.player2.angle],
            self.player2.linearVelocity,
            [self.player2.angularVelocity],
            self.puck.position-[CENTER_X,CENTER_Y],
            self.puck.linearVelocity
            ])

        return self.obs_norm(obs)

    def obs_agent_two(self):
        ''' returns the observations for agent two (symmetric mirrored version of agent one)
        '''
        obs = np.hstack([
            -(self.player2.position-[CENTER_X,CENTER_Y]),
            [self.player2.angle],
            -self.player2.linearVelocity,
            [self.player2.angularVelocity],
            -(self.player1.position-[CENTER_X,CENTER_Y]),
            [self.player1.angle],
            -self.player1.linearVelocity,
            [self.player1.angularVelocity],
            -(self.puck.position-[CENTER_X,CENTER_Y]),
            -self.puck.linearVelocity
            ])

        return self.obs_norm(obs)


    def _compute_reward(self):
        r = 0

        if self.done:
            if self.winner == 0: # tie
                r += 0
            elif self.winner == 1: # you won
                r += 10
            else: # opponent won
                r -= 10

        return r

    def _get_info(self):
        # different proxy rewards:
        # Proxy reward for being close to puck in the own half
        reward_closeness_to_puck = 0
        if self.puck.position[0] < CENTER_X:
            dist_to_puck = dist_positions(self.player1.position, self.puck.position)
            max_dist = 10.
            max_reward = -5. # max (negative) reward through this proxy
            factor = max_reward / (max_dist*self.max_timesteps/2)
            # reward_closeness_to_puck += dist_to_puck*factor # Proxy reward for being close to puck in the own half
            reward_closeness_to_puck = -0.005 * dist_to_puck
        # Proxy reward: touch puck
        reward_touch_puck = 0.
        if self.player1_contact_puck:
            reward_touch_puck = 1.

        # puck is flying in the right direction
        reward_puck_direction = 0
        max_reward = 1.
        factor = max_reward / (self.max_timesteps * self.max_puck_speed)
        # reward_puck_direction = self.puck.linearVelocity[0]*factor # Puck flies right is good and left not
        reward_puck_direction = 0.001 * self.puck.linearVelocity[0]

        obs2 = self.obs_agent_two()

        return { "winner": self.winner,
                 "reward_closeness_to_puck" : reward_closeness_to_puck,
                 "reward_touch_puck" : reward_touch_puck,
                 "reward_puck_direction" : reward_puck_direction,
                 "obs_agent_two": obs2
               }


    def _limit_puck_speed(self):
        puck_speed = np.sqrt(self.puck.linearVelocity[0]**2 + self.puck.linearVelocity[1]**2)
        if puck_speed > self.max_puck_speed:
            self.puck.linearDamping = 10.0
        else:
            self.puck.linearDamping = 0.05


    def step(self, action):
        action = np.clip(action, -1, +1).astype(np.float32)

        self._apply_action_with_max_speed(self.player1, action[:2], 10, True)
        self.player1.ApplyTorque(action[2] * TORQUEMULTIPLAYER, True)
        self._apply_action_with_max_speed(self.player2, action[3:5], 10, False)
        self.player2.ApplyTorque(action[5] * TORQUEMULTIPLAYER, True)

        self._limit_puck_speed()
        self.player1_contact_puck = False

        self.world.Step(self.timeStep, 6 * 30, 2 * 30)

        obs = self._get_obs()
        if self.time >= self.max_timesteps:
            self.truncated = True

        reward = self._compute_reward()
        info = self._get_info()
        reward += info['reward_closeness_to_puck'] + info['reward_touch_puck'] + info['reward_puck_direction']

        self.closest_to_goal_dist = min(self.closest_to_goal_dist,
                                        dist_positions(self.puck.position, (W,H/2)))
        self.time += 1
        return obs, reward, self.done, self.truncated, info

    def render(self, mode='human'):
        from gymnasium.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)
            # self.score_label = pyglet.text.Label('0000', font_size=50,
            #                                      x=VIEWPORT_W/2, y=VIEWPORT_H/2, anchor_x='center', anchor_y='center',
            #                                      color=(0, 0, 0, 255))

        # arr = None
        # win = self.viewer.window
        # win.clear()
        # gl.glViewport(0, 0, VIEWPORT_W, VIEWPORT_H)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        # self.score_label.draw()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class BasicOpponent():
    def __init__(self):
        self.mode = 0
     
    def obs_unnorm(self, obs):
        obs_unnorm = np.copy(obs)
        obs_unnorm[0] *= 7.5
        obs_unnorm[1] *= 5
        obs_unnorm[2] *= np.pi
        obs_unnorm[3] *= 10
        obs_unnorm[4] *= 10
        obs_unnorm[5] *= 15
        obs_unnorm[6] *= 7.5
        obs_unnorm[7] *= 5
        obs_unnorm[8] *= np.pi
        obs_unnorm[9] *= 10
        obs_unnorm[10] *= 10
        obs_unnorm[11] *= 15
        obs_unnorm[12] *= 7.5
        obs_unnorm[13] *= 5
        obs_unnorm[14] *= 20
        obs_unnorm[15] *= 20
        return obs_unnorm         
                
    def act(self, obs, verbose=False, descrete=True, rnd=False):
        if len(obs) != 16:
            actions = []
            for o in obs:
                actions.append(self.act(o, verbose, descrete, rnd))
            return actions
        if rnd == True:
            action = np.random.uniform(-1, 1, 3)
            action = np.where(action >= 0.25, 1, action)
            action = np.where(action <= -0.25, -1, action)
            action = np.where(np.logical_and(action > -0.25, action < 0.25), 0, action)
            return action
        obs = self.obs_unnorm(obs)
        p1 = np.asarray(obs[0:3])
        v1 = np.asarray(obs[3:6])
        p2 = np.asarray(obs[6:9])
        v2 = np.asarray(obs[9:12])
        puck = np.asarray(obs[12:14])
        puckv = np.asarray(obs[14:16])
        # print(p1,v1,puck,puckv)
        target_pos = p1[0:2]
        target_angle = p1[2]

        time_to_break = 0.1
        kp = 10
        kd = 0.5

        # if ball flies towards our goal or very slowly away: try to catch it
        if puckv[0]<1.0:
            dist = np.sqrt(np.sum((p1[0:2] - puck)**2))
            # Am I behind the ball?
            if p1[0] < puck[0] and abs(p1[1] - puck[1]) < 1.0:
                # Go and kick
                target_pos = [puck[0]+0.2, puck[1] + puckv[1]*dist*0.1]
                target_angle = np.random.uniform(-0.5,0.5) # calc proper angle here
            else:
                # get behind the ball first
                target_pos = [-7, puck[1]]
                target_angle = 0
        else: # go in front of the goal
            target_pos = [-7,0]
            target_angle = 0


        target = np.asarray([target_pos[0],target_pos[1], target_angle])
        # use PD control to get to target
        error = target - p1
        need_break = abs((error / (v1+0.01))) < [time_to_break, time_to_break, time_to_break*10]
        if verbose:
            print(error, abs(error / (v1+0.01)), need_break)

        action = error*[kp,kp,kp/2] - v1*need_break*[kd,kd,kd]
        action = np.array(action)
        action = np.where(action >= 1, 1, action)
        action = np.where(action <= -1, -1, action)
        if descrete == True:
            action = np.where(action >= 0.5, 1, action)
            action = np.where(action <= -0.5, -1, action)
            action = np.where(np.logical_and(-0.5 < action, action < 0.5), 0, action)
        return action



class HumanOpponent():
    def __init__(self, env, player=1):
        self.env = env
        self.player = player
        self.a_left = 0
        self.a_right = 0
        self.a_up = 0
        self.a_down = 0
        self.a_clockwise = 0
        self.a_anticlockwise = 0
            
        
    def key_press(self, symbol, mod):
        if self.player == 1:
            if symbol == key.A: self.a_left = 1
            if symbol == key.D: self.a_right = 1
            if symbol == key.W: self.a_up = 1
            if symbol == key.S: self.a_down = 1
            if symbol == key.H: self.a_clockwise = 1
            if symbol == key.G: self.a_anticlockwise = 1
        
        if self.player == 2:
            if symbol == key.NUM_1: self.a_right = 1
            if symbol == key.NUM_3: self.a_left = 1
            if symbol == key.NUM_5: self.a_down = 1
            if symbol == key.NUM_2: self.a_up = 1
            if symbol == key.RIGHT: self.a_clockwise = 1
            if symbol == key.LEFT: self.a_anticlockwise = 1
    
    
    def key_release(self, symbol, mod):
        if self.player == 1:
            if symbol == key.A: self.a_left = 0
            if symbol == key.D: self.a_right = 0
            if symbol == key.W: self.a_up = 0
            if symbol == key.S: self.a_down = 0
            if symbol == key.H: self.a_clockwise = 0
            if symbol == key.G: self.a_anticlockwise = 0

        if self.player == 2:
            if symbol == key.NUM_1: self.a_right = 0
            if symbol == key.NUM_3: self.a_left = 0
            if symbol == key.NUM_5: self.a_down = 0
            if symbol == key.NUM_2: self.a_up = 0
            if symbol == key.RIGHT: self.a_clockwise = 0
            if symbol == key.LEFT: self.a_anticlockwise = 0

    
    def act(self, obs):
        return [self.a_left * -1 + self.a_right * 1, # player x
                self.a_down * -1 + self.a_up * 1, # player y
                self.a_clockwise * -1 + self.a_anticlockwise * 1] # player angle

    
from gymnasium.envs.registration import register

register(
    id='LaserHockey-v0',
    entry_point='laser_hockey_env.laser_hockey_env:LaserHockeyEnv',
)






class Data():
    def __init__(self):
        self.obs_temp = []
        self.a_temp = []
        self.obs_all = []
        self.a_all = []
        self.r_all = []
        self.a_all_index = []
        self.a_all_onehot = []

    
    def delete_temp(self):
        self.obs_temp = []
        self.a_temp = []
        
        
    def obs_all_np(self):
        return np.array(self.obs_all)
    
    
    def a_all_np(self):
        return np.array(self.a_all)
    
    
    def a_all_np_descrete(self):
        a_all = []
        for action in self.a_all:
            action = np.where(action >= 0.5, 1, action)
            action = np.where(action <= -0.5, -1, action)
            action = np.where(np.logical_and(action > -0.5, action < 0.5), 0, action)
            a_all.append(action)
        return np.array(a_all)
    
    
    def a_all_index_np(self):
        if type(self.a_all_index).__module__ == np.__name__:
            return self.a_all_index
        if self.a_all_index:
            return np.array(self.a_all_index)
        else:
            for action in self.a_all:
                action = np.where(action >= 0.5, 1, action)
                action = np.where(action <= -0.5, -1, action)
                action = np.where(np.logical_and(action > -0.5, action < 0.5), 0, action)
                action[0] += 1
                action[1] += 4
                action[2] += 7
                action = action.astype(int)
                self.a_all_index.append(action)
            return np.array(self.a_all_index)
    
    
    def a_all_onehot_np(self):
        if type(self.a_all_index).__module__ == np.__name__:
            return self.a_all_onehot
        if self.a_all_onehot:
            return np.array(self.a_all_onehot)
        else:
            for action in self.a_all:
                action = np.where(action >= 0.5, 1, action)
                action = np.where(action <= -0.5, -1, action)
                action = np.where(np.logical_and(action > -0.5, action < 0.5), 0, action)
                action += 1
                action = action.astype(int)
                onehot = np.array([0,0,0,0,0,0,0,0,0])
                onehot[action[0]] = 1
                onehot[action[1]+3] = 1
                onehot[action[2]+6] = 1
                self.a_all_onehot.append(onehot)
            return np.array(self.a_all_onehot)
    
    
    def r_all_np(self):
        return np.array(self.r_all)
    
    
    def r_calculate(self, N, win=None):
        rewards = []
        if win == None:
            rewards = [[0.5]] * N
        elif win:
            for _ in range(N):
                rewards.append([0.5+0.5*0.99**(N-(_+1))])
        elif not win:
            for _ in range(N):
                rewards.append([0.5-0.5*0.99**(N-(_+1))])
        return rewards

    
    
       
class QNetwork(nn.Module):
    def __init__(self, device):
        super(QNetwork, self).__init__()
        self.h_N = 256
        self.sigmoid = nn.Sigmoid()
        self.linear_in = nn.Linear(16, self.h_N).to(device)
        self.linear_h = nn.Linear(self.h_N, self.h_N).to(device)
        self.linear_out_x = nn.Linear(self.h_N, 3).to(device)
        self.linear_out_y = nn.Linear(self.h_N, 3).to(device)
        self.linear_out_a = nn.Linear(self.h_N, 3).to(device)

    def forward(self, s, device=device, stochastic=False, imitation=False, rl=False, action=None, inside=False, rnd=0):
        if type(s).__module__ != 'torch':
            s = torch.tensor(np.array(s, dtype=np.float64), dtype=torch.float, device=device)
        if type(action).__module__ != 'torch':
            action = torch.tensor(np.array(action, dtype=np.float64), dtype=torch.float, device=device)
        dim = s.dim()-1
        q = F.relu(self.linear_in(s))
        q = F.relu(self.linear_h(q))
        q_x = self.linear_out_x(q)
        q_y = self.linear_out_y(q)
        q_a = self.linear_out_a(q)
        if imitation == True:
            return torch.hstack((q_x, q_y, q_a))
        q_x = self.sigmoid(q_x)
        q_y = self.sigmoid(q_y)
        q_a = self.sigmoid(q_a)
        if inside == True:
            return torch.hstack((q_x, q_y, q_a))
        if rl == True:
            q = torch.hstack((q_x, q_y, q_a))
            q = torch.gather(q, 1, action)
            return q
        if np.random.rand() < rnd:
            stochastic = True
        if stochastic == False:
            q_x = torch.sub(torch.argmax(q_x, dim=dim), 1)
            q_y = torch.sub(torch.argmax(q_y, dim=dim), 1)
            q_a = torch.sub(torch.argmax(q_a, dim=dim), 1)
            action = torch.transpose(torch.vstack((q_x, q_y, q_a)), 0, 1)
            if dim == 0: return action[0]
            else: return action
        if stochastic == True:
            q_x = torch.sub(torch.multinomial(q_x, 1), 1)
            q_y = torch.sub(torch.multinomial(q_y, 1), 1)
            q_a = torch.sub(torch.multinomial(q_a, 1), 1)
            action = torch.hstack((q_x, q_y, q_a))
            return action
    
    
  
    
class PolicyNetwork(nn.Module):
    def __init__(self, device):
        super(PolicyNetwork, self).__init__()
        self.h_N = 200
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.linear_in = nn.Linear(16, self.h_N).to(device)
        self.linear_h = nn.Linear(self.h_N, self.h_N).to(device)
        self.linear_out_x = nn.Linear(self.h_N, 3).to(device)
        self.linear_out_y = nn.Linear(self.h_N, 3).to(device)
        self.linear_out_a = nn.Linear(self.h_N, 3).to(device)

    def forward(self, s, device, stochastic=False, imitation=False, rl=False, action=None):
        if type(s).__module__ == np.__name__:
            s = torch.tensor(s, dtype=torch.float, device=device)
        dim = s.dim()-1
        a = F.relu(self.linear_in(s))
        a = F.relu(self.linear_h(a))
        a_x = self.linear_out_x(a)
        a_y = self.linear_out_y(a)
        a_a = self.linear_out_a(a)
        if imitation == True:
            return torch.hstack((a_x, a_y, a_a))
        a_x = self.softmax(a_x)
        a_y = self.softmax(a_y)
        a_a = self.softmax(a_a)
        if rl == True:
            p = torch.hstack((a_x, a_y, a_a))
            return torch.unsqueeze(torch.diagonal(torch.tensordot(p, action, dims=([1],[1])), 0), 1)
        if stochastic == False:
            a_x = torch.sub(torch.argmax(a_x, dim=dim), 1)
            a_y = torch.sub(torch.argmax(a_y, dim=dim), 1)
            a_a = torch.sub(torch.argmax(a_a, dim=dim), 1)
            action = torch.transpose(torch.vstack((a_x, a_y, a_a)), 0, 1)
            if dim == 0: return action[0]
            else: return action
        if stochastic == True:
            a_x = torch.sub(torch.multinomial(a_x, 1), 1)
            a_y = torch.sub(torch.multinomial(a_y, 1), 1)
            a_a = torch.sub(torch.multinomial(a_a, 1), 1)
            action = torch.hstack((a_x, a_y, a_a))
            return action
        



class CriticNetwork(nn.Module):
    def __init__(self, device):
        super(CriticNetwork, self).__init__()
        self.h_N = 200
        self.linear_in = nn.Linear(16, self.h_N).to(device)
        self.linear_h = nn.Linear(self.h_N, self.h_N).to(device)
        self.linear_out = nn.Linear(self.h_N, 1).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s, device):
        if type(s).__module__ == np.__name__:
            s = torch.tensor(s, dtype=torch.float, device=device)
        v = F.relu(self.linear_in(s))
        v = F.relu(self.linear_h(v))
        v = self.sigmoid(self.linear_out(v))
        return v

