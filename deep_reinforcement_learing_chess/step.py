def step(self,action):
    if self.board.turn:
        turn = 'white'
    else:
        turn = 'black'

    state = self.translate_board()
    rewards = evaluate_reward(self.board,action)
    self.rewards_history['white'].append(rewards[0])
    self.rewards_history['black'].append(rewards[0])
    self.update_pgn(action)
    self.board.push(action)

    state_next = self.board
    state_next = translate_board(state_next)

    self.done = self.board.is_game_over()

    self.action_history.append(move2num[action])
    self.state_history.append(state)
    self.state_next_history.append(state_next)
    self.done_history.append(self.done)
    self.episode_reward_history.append(rewards)