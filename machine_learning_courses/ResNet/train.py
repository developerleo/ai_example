import tensorflow as tf
from keras.src.datasets import cifar100
from keras.src.optimizers import Adam
import os
from resnet import resnet18

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


def preprocess(x, y):
    # [-1, 1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(10).map(preprocess).batch(8)

test_db = tf.data.Dataset.from_tensor_slices((x, y))
test_db = test_db.map(preprocess).batch(8)

sample = next(iter(train_db))
print('sample', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():
    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    optimizer = Adam(learning_rate=1e-3)

    for epoch in range(5):
        for step, (x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                #[b , 32 ,32, 3] => [b, 100]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf._losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step%100 == 0:
                 print('epcoh:', epoch, 'step:', step, 'loss:', float(loss))
        #Test
        total_num = 0
        total_correct = 0
        for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equel(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)

if __name__ == '__main__':
    main()