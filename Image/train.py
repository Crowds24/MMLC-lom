import os

from random import random

from keras import regularizers
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from keras import backend as K
import pandas as pd
from keras.layers import Input, Dense, Attention
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, Loss
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution

def preparation_test_data():
    ground_truth = np.load('../Data/new/train/labels_train.npy')
    ground_truth = pd.DataFrame(ground_truth)
    ground_truth.reset_index(level=0, inplace=True)
    ground_truth.rename(columns={'index': 'taskId'}, inplace=True)
    truth_dict = {}
    for index, data in ground_truth.iterrows():
        truth_dict[data['taskId']] = data[0]

    return truth_dict
def preparation_data():
    # 1.列名
    train_data = pd.read_csv('../Data/redundancy_1/train_data_3.csv', index_col=None)
    data = train_data
    train_gate_data = train_data[['worker']]
    train_gate_data = pd.get_dummies(train_gate_data, columns=['worker'])
    train_label = train_data[['label']]
    train_label = pd.get_dummies(train_label, columns=['label'])
    train_data = train_data.drop(columns=['taskId', 'worker', 'label'])

    data = data.drop_duplicates('taskId')
    taskId = data['taskId'].reset_index(drop=True)
    predict_data = data.drop(columns=['taskId', 'worker', 'label'])
    print(predict_data)
    return train_gate_data, train_data, train_label, predict_data, taskId

SEED = 42
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)



def random_oracle():

    return np.random.uniform(0, 1, (1, 1, 59))

class CommonWCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, gate_train_data):
        super(CommonWCallback, self).__init__()
        self.model = model
        self.gate_train_data = gate_train_data

    def on_epoch_end(self, epoch, logs=None):
        common_w = self.model.get_layer('common_w').predict(self.gate_train_data)
        print(f'Epoch {epoch + 1}: common_w = {common_w}')

# 这里是你原本定义的模型函数
def expert_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(500, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(32, activation='relu')(x)
    return Model(inputs=inputs, outputs=outputs)

def gate_model_common(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(59, activation='relu', kernel_regularizer=regularizers.l1(0.1))(inputs)
    x = K.expand_dims(x, axis=1)
    x = Attention()([x, x])
    model = Model(inputs=inputs, outputs=x)
    return model

def gate_model_w(common_w, oracle_w):
    return common_w + oracle_w

def gate_model_softmax(input_shape, num_experts):
    inputs = Input(shape=input_shape)
    x = Dense(num_experts, activation='softmax')(inputs)
    x = tf.repeat(x, 32, axis=1)
    model = Model(inputs=inputs, outputs=x)
    return model

def ShareMoE(input_shape, num_experts):
    inputs = Input(shape=input_shape)
    experts_outputs = []
    for _ in range(num_experts):
        expert = expert_model(input_shape)(inputs)
        expert = tf.expand_dims(expert, axis=0)
        experts_outputs.append(expert)
    output = tf.concat(experts_outputs, axis=0)
    output = tf.transpose(output, [1, 2, 0])
    model = Model(inputs=inputs, outputs=output)
    return model

def tower_output(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(input_shape, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    output = Dense(8, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def multiply(output1, output2):
    output = tf.multiply(output1, output2)
    output = tf.reduce_sum(output, axis=2)
    return output

class CommonWLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CommonWLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # 计算 common_w
        common_w = inputs
        return common_w

class CustomLoss(Loss):
    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        base_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        common_w = model.get_layer('model_79').output
        regularization_loss = tf.reduce_sum(tf.square(common_w))
        return base_loss + 0.001 * regularization_loss



if __name__ == '__main__':

    ground_truth = preparation_test_data()
    gate_train_data, expert_train_data, type_label_data, predict_data, taskId = preparation_data()
    gate_train_data, expert_train_data, type_label_data = shuffle(gate_train_data, expert_train_data, type_label_data)

    in_features_expert = expert_train_data.shape[1]
    in_features_gate = gate_train_data.shape[1]
    num_experts = 10

    # 定义模型
    gate_input_layer = Input(shape=(in_features_gate,))
    expert_input_layer = Input(shape=(in_features_expert,))

    gate_common_model = gate_model_common(in_features_gate)
    common_w = gate_common_model(gate_input_layer)

    oracle_w = random_oracle()
    print(oracle_w)
    gate_output = gate_model_w(common_w, oracle_w)
    gate_output_model = gate_model_softmax(gate_output.shape[1:], num_experts)
    gate_output = gate_output_model(gate_output)

    share_model = ShareMoE(in_features_expert, num_experts)
    share_output = share_model(expert_input_layer)

    final_out = multiply(share_output, gate_output)
    final_out_model = tower_output(final_out.shape[1])
    final_out = final_out_model(final_out)

    # 定义最终模型
    model = Model(inputs=[gate_input_layer, expert_input_layer], outputs=final_out)
    model.summary()


    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(loss=CategoricalCrossentropy(), optimizer=adam_optimizer, metrics=['accuracy'])

    best_train_acc = 0
    best_test_acc = 0

    # 初始化列表用于保存每个epoch的损失值
    losses = []

    # 实例化自定义回调
    common_w_callback = CommonWCallback(model, gate_train_data)
    for epoch in range(150):
        print("epoch:", epoch)
        history = model.fit(
            x=[gate_train_data, expert_train_data],
            y=type_label_data,
            epochs=1,
            batch_size=32,
            shuffle=False,
            workers=1,
        )

        expert = model.layers[4].predict(predict_data, verbose=0)
        oracle = model.layers[5].predict(oracle_w, verbose=0)
        output = tf.multiply(expert, oracle)
        output = K.sum(output, axis=2)
        pred = model.layers[8].predict(output, verbose=0)
        predict_result = np.argmax(pred, axis=1)
        predict_result = pd.DataFrame({'label': predict_result})
        predict_result = pd.concat([taskId, predict_result], axis=1)
        predict_dict = {}
        for index, data in predict_result.iterrows():
            predict_dict[data['taskId']] = data['label']
        count = 0
        for predict in predict_dict:
            if (predict_dict[predict] == ground_truth[predict]):
                count += 1
        train_acc = count / len(predict_dict)
    model.save('model')
