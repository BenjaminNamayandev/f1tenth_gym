import time
import keyboard  # pip install keyboard
import gym
import numpy as np

def main():
    env = gym.make('f110_gym:f110-v0',
                   map='Silverstone_map',
                   num_agents=1)

    initial_poses = np.array([[0.0, 0.0, 0.0]])
    obs = env.reset(poses=initial_poses)

    print("Use W, A, S, D to drive. Press Esc to exit.")

    try:
        while True:
            # Exit if Esc is pressed
            if keyboard.is_pressed('esc'):
                print("Exiting...")
                break

            # Default controls
            steer = 0.0
            throttle = 0.0

            # Keyboard-based controls
            if keyboard.is_pressed('w'):
                throttle = 50
            if keyboard.is_pressed('s'):
                throttle = -50
            if keyboard.is_pressed('a'):
                steer = 10
            if keyboard.is_pressed('d'):
                steer = -10

            action = np.array([[steer, throttle]], dtype=np.float32)
            obs, reward, done, info = env.step(action)

            # Get position of the agent for camera
            car_x = obs['poses_x'][0]
            car_y = obs['poses_y'][0]
            car_theta = obs['poses_theta'][0]

            # First update the camera (if the renderer exists and has update_cam):
            if hasattr(env, 'renderer') and hasattr(env.renderer, 'update_cam'):
                env.renderer.update_cam(car_x, car_y, angle=car_theta, zoom=50.0)

            # Then render
            env.render(mode='human')

            # If the episode ends, reset
            if done:
                obs = env.reset(poses=initial_poses)

            time.sleep(0.01)

    finally:
        env.close()
        print("Simulator closed.")

if __name__ == "__main__":
    main()
