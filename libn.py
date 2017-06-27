from __future__ import print_function
import numpy as np
import os
import PIL.Image as Image

class libn():

    def __init__(self):
        self.image_list = []
        self.label_list = []
        self.num_examples = 100
       # self.img = img
       # self.lbl = lbl
        self.idx_in_epoch = 0
        self.epoch_completed = 0
        self.result = np.array([])

    def next_batch(self,batch_size):
        start = self.idx_in_epoch
        self.idx_in_epoch += batch_size
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
        return self.image_list[start:end],self.label_list[start:end]
    
    def print_list(self):
        print(self.result)
      #  print(self.image_list)
      #  print(self.label_list)

    def print_shape(self):
        #print(self.image_list.shape)
        #print(self.label_list.shape)
        print(self.result.shape)

    def get_file(self,file_dir):
        cats = []
        label_cats = []
        dogs = []
        label_dogs = []
        #self.num_examples = len(os.listdir(file_dir))
        for fil in os.listdir(file_dir):
            name = fil.split('.')
            if name[0] == 'cat':
                cats.append(file_dir + fil)
                label_cats.append(0)
            else:
                dogs.append(file_dir + fil)
                label_dogs.append(1)
        img_list = np.hstack((cats,dogs))
        lbl_list = np.hstack((label_cats,label_dogs))

        temp = np.array([img_list,lbl_list])
        temp = temp.transpose()
        np.random.shuffle(temp)
        
        #self.image_list = list(temp[:,0])
        #lbl_list = list(temp[:,1])
        #self.label_list = [int(i) for i in lbl_list]
        self.image_list = temp[:,0]
        lbl_list = temp[:,1]
        self.label_list = np.array([int(i) for i in lbl_list])

        return self.image_list,self.label_list
    
    def img2vec(self):
        n = self.num_examples
        for i in range(n):
            image = Image.open(self.image_list[i])
            image = image.resize((32,32),Image.ANTIALIAS)
            r,g,b = image.split()
            r_arr = np.array(r).reshape(1024)
            g_arr = np.array(g).reshape(1024)
            b_arr = np.array(b).reshape(1024)
            image_arr = np.concatenate((r_arr,g_arr,b_arr))
            self.result = np.concatenate((self.result,image_arr))
        self.result = self.result.reshape((n,3072))
        return self.result
    
if __name__ == '__main__':
    train_dir = '/home/shiyanlou/cat_dog/train/'
    mni = libn()
    mni.get_file(train_dir)
    mni.img2vec()
    #mni.print_list()
    #mni.next_batch(100)
    mni.print_shape()
    """
    for i in range(1):
        mni.next_batch(100)
    """
