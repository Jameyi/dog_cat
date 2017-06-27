from __future__ import print_function
import numpy as np
import os
import PIL.Image as Image
from matplotlib import pyplot as plt

"""
next:plot the batch image,verify the matrix and label whether match
"""
class libn():

    def __init__(self):
        self.image_list = []
        self.label_list = []
        self.num_examples = 0
        self.idx_in_epoch = 0
        self.epoch_completed = 0
        self.result = np.array([])

    def next_batch(self,batch_size):
        img = np.array([])
        start = self.idx_in_epoch
        self.idx_in_epoch += batch_size
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.image_list = self.image_list[perm]
        self.label_list = self.label_list[perm]
        if self.idx_in_epoch > self.num_examples:
            self.epochs_completed += 1
            perm = np.arrange(self.num_examples)
            np.random.shuffle(perm)
            self.image_list = self.image_list[perm]
            self.label_list = self.label_list[perm]
            start = 0
            self.idx_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.idx_in_epoch

        for i in range(len(self.image_list[start:end])):
            img_arr = self.img2vec(self.image_list[i])
            img = np.concatenate((img,img_arr))
        img = img.reshape(len(self.image_list[start:end]),3072)
        lbl = self.label_list[start:end]
        return img,lbl
        
    
    def print_list(self):
        print(self.num_examples)
        print(self.image_list.shape)
        print(self.label_list.shape)

    def print_shape(self):
        pass

    def get_file(self,file_dir):
        image_list = []
        label = []
        for fil in os.listdir(file_dir):
            name = fil.split('.')
            image_list.append(file_dir + fil)
            
            if name[0] == 'cat':
                label.append(0)
            else:
                label.append(1)
        self.image_list = np.array(image_list)
        self.label_list = np.array(label)
        self.num_examples = len(os.listdir(file_dir))
    
    def img2vec(self,image_name):
        image = Image.open(image_name)
        image = image.resize((32,32),Image.ANTIALIAS)
        r,g,b = image.split()
        r_arr = np.array(r).reshape(1024)
        g_arr = np.array(g).reshape(1024)
        b_arr = np.array(b).reshape(1024)
        image_arr = np.concatenate((r_arr,g_arr,b_arr))
        return image_arr

    def vec2img(self,vec):
        arr = vec
 #       print(arr.shape)
        #rows = vec.shape[0]
        arr = arr.reshape(3,32,32)
        #for idx in range(rows):
        #    a = arr[idx]
        #    r = Image.fromarray(a[0]).convert('L')
        #    g = Image.fromarray(a[1]).convert('L')
        #    b = Image.fromarray(a[2]).convert('L')
        #    image = Image.merge("RGB",(r,g,b))
        r = Image.fromarray(arr[0]).convert('L')
        g = Image.fromarray(arr[1]).convert('L')
        b = Image.fromarray(arr[2]).convert('L')
        image = Image.merge("RGB",(r,g,b))
        return image
    
if __name__ == '__main__':
    train_dir = '/home/shiyanlou/cat_dog/train/'
    mni = libn()
    mni.get_file(train_dir)
    X,Y = mni.next_batch(5)
    print(Y.shape)
   # print(X.shape)
   # print(X)
   # print(X[0])
   # print(Y)
 
#==========================================
# plot to verify X whether match Y
#==========================================
#    for i in range(5):
#        print('label: %d' %Y[i])
#        tmp = X[i]
#        x = mni.vec2img(tmp)
#        plt.imshow(x)
#        plt.show()
    
