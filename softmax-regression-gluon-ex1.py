import mxnet as mx
from mxnet import nd,autograd
from mxnet import gluon
import numpy as np

ctx = mx.cpu()
batch_size = 64
num_inputs = 784
num_outputs = 10


#dataset

def transform(data,label):
    return data.astype(np.float32)/255,label.astype(np.float32)

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True,transform=transform),batch_size=batch_size,shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False,transform=transform),batch_size=batch_size,shuffle=False)

#model
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_outputs))

#intialize
net.collect_params().initialize(mx.init.Normal(sigma=1.),ctx=ctx)

#loss
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#optimize
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})

#accuracy
def evaluate_accuracy(data_iterator,net):
    acc = mx.metric.Accuracy()
    for i,(data,label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)

        output = net(data)
        predictions = nd.argmax(output,axis=1)
        acc.update(predictions,label)

    return acc.get()[1]

# evaluate_accuracy(train_data,net)

#training
epochs = 4
moving_loss = 0.
smoothing_constant = .01
niter = 0

for e in range(epochs):
    for i,(data,label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)

        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        trainer.step(batch_size)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter +=1
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * nd.mean(loss).asscalar()
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)

    test_accuracy = evaluate_accuracy(test_data,net)
    train_accuracy = evaluate_accuracy(train_data,net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, est_loss, train_accuracy, test_accuracy))

    #prediction