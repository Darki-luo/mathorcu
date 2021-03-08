import paddle
#paddle.callbacks.Callback
#custom Callback

# visualdl
class VDL(paddle.callbacks.Callback):
    def __init__(self, write, iters=0, epochs=0):
        super(VDL, self).__init__()
        self.write = write
        self.iters = iters
        self.epochs = epochs



    def on_train_batch_end(self, step, logs):

        self.iters += 1

        #记录loss
        self.write.add_scalar(tag="train/loss",step=self.iters,value=logs['loss'][0])
        #记录 accuracy
        self.write.add_scalar(tag="train/miou",step=self.iters,value=logs['miou'])

    #def on_eval_batch_begin(self, step, logs):



    def on_eval_end(self, logs):

        self.epochs += 1

        # 记录loss
        #self.write.add_scalar(tag="eval/loss", step=self.epochs, value=logs['loss'][0])
        # 记录 accuracy
        self.write.add_scalar(tag="eval/miou", step=self.epochs, value=logs['miou'])


        




