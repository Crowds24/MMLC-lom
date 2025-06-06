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
    train_true = pd.read_csv('../Data/train/music_true.csv')
    truth_dict = {}
    for index, t_data in train_true.iterrows():
        truth_dict[t_data['Input.song']] = t_data['Input.true_label']

    data = pd.read_csv('./data/data_test.csv', index_col=0)
    print(data)
    data_test = data.drop(columns=['id', 'class'])
    label_test = data['class'].values

    train_data = pd.read_csv('../Data/train/music_feature.csv', index_col=None)
    print("train_data", train_data)
    train_data = train_data.drop(columns=['class', 'annotator']).drop_duplicates('id')
    predict_data = train_data.drop(columns=['id'])
    id = train_data['id'].reset_index(drop=True)
    print(data_test)

    return data_test, label_test, truth_dict, predict_data, id

def preparation_data():
    # 1.列名
    data = pd.read_csv('../Data/train/music_feature.csv', index_col=None)
    train_data = data.drop(columns=["id", "annotator", "class"])
    worker_data = data['annotator']
    worker_data = pd.get_dummies(worker_data, columns=['annotator'])
    label_data = data['class']
    label_data = pd.get_dummies(label_data, columns=['class'])

    return worker_data, train_data, label_data

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
    # return [[[1.17502369 ,1.40270413 ,1.39737694 ,1.26755933 ,1.33476267 ,1.25122619,
    #    0.40209898 ,1.06539328 ,1.78881589 ,0.2805919  ,0.60101125 ,1.20097792,
    #    1.29707369 ,1.24063833 ,0.01429405 ,1.68108914 ,0.37398184 ,1.69763634,
    #    0.89016573 ,1.49937697 ,0.69075856 ,1.58832762 ,0.64293775 ,1.04280179,
    #    1.7694413  ,0.70331353 ,1.14180363 ,0.19613815 ,0.92263842 ,1.55233381,
    #    0.45592367 ,0.91888821 ,0.51588845 ,1.17517886 ,1.21608451 ,1.43943915,
    #    0.24476592 ,0.97965912 ,0.61739795 ,0.93856209 ,0.47410257 ,1.1033144,
    #    0.57808307 ,1.02242292]]]
    return np.random.uniform(0, 2, (1, 1, 44))

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
    outputs = Dense(16, activation='relu')(x)
    return Model(inputs=inputs, outputs=outputs)

def gate_model_common(input_shape):
    inputs = Input(shape=input_shape)
    # , kernel_regularizer=regularizers.l1(10)
    x = Dense(44, activation='relu', kernel_regularizer=regularizers.l1(0.005))(inputs)
    x = K.expand_dims(x, axis=1)
    x = Attention()([x, x])
    model = Model(inputs=inputs, outputs=x)
    return model

def gate_model_w(common_w, oracle_w):
    return common_w + oracle_w

def gate_model_softmax(input_shape, num_experts):
    inputs = Input(shape=input_shape)
    x = Dense(num_experts, activation='softmax')(inputs)
    x = tf.repeat(x, 16, axis=1)
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
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
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
        common_w = model.get_layer('model').output
        regularization_loss = tf.reduce_sum(tf.square(common_w))
        return base_loss + 0.001 * regularization_loss



if __name__ == '__main__':

    data_test, label_test, truth_dict, predict_data, id = preparation_test_data()
    gate_train_data, expert_train_data, type_label_data = preparation_data()
    gate_train_data, expert_train_data, type_label_data = shuffle(gate_train_data, expert_train_data, type_label_data)

    in_features_expert = expert_train_data.shape[1]
    in_features_gate = gate_train_data.shape[1]
    num_experts = 8

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


    adam_optimizer = Adam(learning_rate=0.005)
    model.compile(loss=CategoricalCrossentropy(), optimizer=adam_optimizer, metrics=['accuracy'])

    best_train_acc = 0
    best_test_acc = 0

    # 初始化列表用于保存每个epoch的损失值
    losses = []

    # 实例化自定义回调
    common_w_callback = CommonWCallback(model, gate_train_data)
    for epoch in range(1500):
        print("epoch:", epoch)
        history = model.fit(
            x=[gate_train_data, expert_train_data],
            y=type_label_data,
            epochs=1,
            batch_size=2042,
            shuffle=False,
            workers=1,
        )

        # 记录当前epoch的损失值
        # loss = history.history['loss'][0]
        # losses.append(loss)
        #
        # expert = model.layers[4].predict(data_test, verbose=0)
        # oracle = model.layers[5].predict(oracle_w, verbose=0)
        # common_w = model.layers[1].predict(gate_train_data, verbose=0)
        # output = tf.multiply(expert, oracle)
        # output = tf.reduce_sum(output, axis=2)
        # pred = model.layers[8].predict(output, verbose=0)
        # pred = np.argmax(pred, axis=1)
        # count = 0
        # for i in range(len(pred)):
        #     if pred[i] == label_test[i]:
        #         count += 1
        # test_acc = count/300
        # if test_acc > best_test_acc:
        #     best_test_acc = test_acc
        # print("test_acc:", test_acc)
        # print("best_test_acc:", best_test_acc)
        expert = model.layers[4].predict(predict_data, verbose=0)
        oracle = model.layers[5].predict(oracle_w, verbose=0)
        output = tf.multiply(expert, oracle)
        output = K.sum(output, axis=2)
        pred = model.layers[8].predict(output, verbose=0)
        predict_result = np.argmax(pred, axis=1)
        predict_result = pd.DataFrame({'label': predict_result})
        predict_result = pd.concat([id, predict_result], axis=1)
        predict_dict = dict(zip(predict_result['id'], predict_result['label']))
        # predict_dict = {}
        # for index, data in predict_result.iterrows():
        #     predict_dict[data['id']] = data['label']
        count = 0
        for predict in predict_dict:
            if (predict_dict[predict] == truth_dict[predict]):
                count += 1
        train_acc = count / len(predict_dict)
    bestModel.save('model')
