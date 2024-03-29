import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input

class Q_model():
    def __init__(self,model = None):
        if model:
            print('CUSTOM MODEL SET')
            self.model = model
        else:
            self.model = self.create_q_model()

    def create_q_model(self):
    # Network defined by the Deepmind paper
        input_layer = Input(shape=(8, 8, 12))

        # Convolutions on the frames on the screen
        x = Conv2D(filters=64,kernel_size = 2,strides = (2,2),activation = 'relu')(input_layer)
        x = Conv2D(filters=128,kernel_size=2,strides = (2,2),activation = 'relu')(x)
        x = Conv2D(filters=256,kernel_size=2,strides = (2,2),activation = 'relu')(x)
        x = Flatten()(x)

        action = Dense(4096,activation = 'softmax')(x)
        return Model(inputs=input_layer, outputs=action)
    
    def predict(self,env):
        state_tensor = tf.convert_to_tensor(env.translate_board())
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        action_space = filter_legal_moves(env.board,action_probs[0])
        action = np.argmax(action_space, axis=None)
        move= num2move[action]
        return move,action
    
    def explore(self,env):
        action_space = np.random.randn(4096)
        action_space = filter_legal_moves(env.board,action_space)
        action = np.argmax(action_space, axis=None)
        move= num2move[action]
        return move,action