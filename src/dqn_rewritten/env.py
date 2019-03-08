import numpy as np
import math
from gym.envs.classic_control import CartPoleEnv
env = CartPoleEnv()

def cartpole_transition_function(state:list, action):
    env.reset()
    env.state = np.array(state)
    
    # copied from gym src code
    assert env.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    state = env.state
    x, x_dot, theta, theta_dot = state
    force = env.force_mag if action==1 else -env.force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + env.polemass_length * theta_dot * theta_dot * sintheta) / env.total_mass
    thetaacc = (env.gravity * sintheta - costheta* temp) / (env.length * (4.0/3.0 - env.masspole * costheta * costheta / env.total_mass))
    xacc  = temp - env.polemass_length * thetaacc * costheta / env.total_mass
    if env.kinematics_integrator == 'euler':
        x  = x + env.tau * x_dot
        x_dot = x_dot + env.tau * xacc
        theta = theta + env.tau * theta_dot
        theta_dot = theta_dot + env.tau * thetaacc
    else: # semi-implicit euler
        x_dot = x_dot + env.tau * xacc
        x  = x + env.tau * x_dot
        theta_dot = theta_dot + env.tau * thetaacc
        theta = theta + env.tau * theta_dot
    env.state = (x,x_dot,theta,theta_dot)
    
    return list(env.state)

def cartpole_reward_function(state, action, next_state):
    env.reset()
    done = cartpole_done_function(next_state)
    
    if not done:
        reward = 1.0
    else:
        # reward = -1 when done
        reward = -1.0
    
    return reward

def cartpole_get_initial_state():
    return list(env.reset())

def cartpole_done_function(state):
    env.reset()
    
    # copied from gym src code
    x, x_dot, theta, theta_dot = state
    done =  x < -env.x_threshold \
            or x > env.x_threshold \
            or theta < -env.theta_threshold_radians \
            or theta > env.theta_threshold_radians
    done = bool(done)
    
    return done

def cartpole_render(state):
    env.reset()
    env.state = np.array(state)
    
    # copied from gym src code
    mode = 'human'
    screen_width = 600
    screen_height = 400

    world_width = env.x_threshold*2
    scale = screen_width/world_width
    carty = 100 # TOP OF CART
    polewidth = 10.0
    polelen = scale * (2 * env.length)
    cartwidth = 50.0
    cartheight = 30.0

    if env.viewer is None:
            from gym.envs.classic_control import rendering
            env.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            env.carttrans = rendering.Transform()
            cart.add_attr(env.carttrans)
            env.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            env.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(env.poletrans)
            pole.add_attr(env.carttrans)
            env.viewer.add_geom(pole)
            env.axle = rendering.make_circle(polewidth/2)
            env.axle.add_attr(env.poletrans)
            env.axle.add_attr(env.carttrans)
            env.axle.set_color(.5,.5,.8)
            env.viewer.add_geom(env.axle)
            env.track = rendering.Line((0,carty), (screen_width,carty))
            env.track.set_color(0,0,0)
            env.viewer.add_geom(env.track)

            env._pole_geom = pole

    if env.state is None: return None

    # Edit the pole polygon vertex
    pole = env._pole_geom
    l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
    pole.v = [(l,b), (l,t), (r,t), (r,b)]

    x = env.state
    cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
    env.carttrans.set_translation(cartx, carty)
    env.poletrans.set_rotation(-x[2])

    return env.viewer.render(return_rgb_array = mode=='rgb_array')