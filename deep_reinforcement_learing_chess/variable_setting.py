from tensorflow import keras


iterations = 10
len_episodes = 20
frame_count = 10
max_steps_per_episode = 100

epsilon_interval = 2
epsilon_random_frames = 500
epsilon_greedy_frames = 1000
epsilon_min = 0.1

loss_function = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

update_target_network = 1000