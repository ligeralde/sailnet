import numpy as np
import array
import os
import math
import random
from os.path import expanduser
# from sklearn.preprocessing import StandardScaler as scale

#### TO ADD: finish full data then split
              # add randomizer for filepath order
class VHDataset:
  def __init__(self, img_dir, file_ext, filenames=None, dims=(1024,1532), logscale=True, with_mean=True, with_std=False, train_prop=.8):
    assert dims[0] % 2 == 0 and dims[1] % 2 == 0
    self.folderpath = os.path.expanduser(img_dir)
    self.file_ext = file_ext
    if filenames:
      self.filenames = filenames
    else:
      self.filenames = os.listdir(folderpath)
    self.train_prop = train_prop
    self.dims = dims
    self.logscale = logscale
    self.with_mean = with_mean
    self.with_std = with_std
    self.filepaths = random.shuffle([os.path.join(self.folderpath, name) for name in self.filenames if not '.DS_Store' in name])

  def extract_dataset(self):
    train_length = int(self.train_prop*len(self.filepaths))
    test_length = len(self.filepaths)-int(self.train_prop*len(self.filepaths))
    print('Preallocating array...')
    full_data = np.empty((len(self.filepaths) self.dims[0], self.dims[1]),dtype='float32')
    print('Beginning loop...')
    for i, path in enumerate(self.filepaths):
      if (i+1)%10 == 0:
        if i < train_length-1:
          print('Processing image #{} out of {}...'.format(i+1, train_length))
        else:
          print('Processing testing image #{} out of {}...'.format(i-train_length+1, test_length))
      img = self.extract_image(path, self.dims, self.logscale, self.with_mean, self.with_std)
      full_data[i, :, :] = img.reshape(self.dims)
    print('Reshaping and saving raw training images...')
    np.savez('raw_train_{}'.format(self.file_ext), self.move_axis_to_batch_minor(full_data[:train_length,:,:],0))
    if test_length != 0:
      np.savez('raw_test{}'.format(self.file_ext), self.move_axis_to_batch_minor(full_data[train_length:,:,:],0))





    #   else:
    #     if i == train_length-1:

    #       np.savez('raw_train_{}'.format(self.file_ext), self.move_axis_to_batch_minor(train_data,0))
    #       train_data = None
    #       test_data = np.empty((test_length,self.dims[0],self.dims[1]), dtype='float32')
    #     if (i-train_length)%10 == 0 and i-train_length>-1:
    #       print('Processing testing image #{} out of {}...'.format(i-train_length, test_length))
    #     test_data[i-train_length+1, :, :] = img.reshape(self.dims)
    # if test_data != []:
    #   print('Reshaping and saving raw testing images...')
    #   np.savez('raw_test_{}'.format(self.file_ext), self.move_axis_to_batch_minor(test_data,0))

  def extract_image(self, filepath, dims, logscale, with_mean, with_std):
    return self.mean_center(self.center_crop(self.bytes_to_arrays(filepath,logscale),dims),with_mean,with_std)

  def bytes_to_arrays(self, filepath, logscale):
    with open(filepath, 'rb') as handle:
      s = handle.read()
    arr = array.array('H', s)
    arr.byteswap()
    if logscale == True:
      return np.log(1+np.array(arr,dtype='uint16'))
    else:
      return np.array(arr,dtype='uint16')

  def center_crop(self, img, dims):
    if dims[0] != 1024:
      row_crop = (1024-dims[0])//2
      img = img.reshape((1024,1536))[row_crop:-row_crop,col_crop:-col_crop]
    if dims[1] != 1536:
      col_crop = (1536-dims[1])//2
      img = img.reshape((dims[0],1536))[:,col_crop:-col_crop]
    return img.ravel()

  def mean_center(self, img, with_mean, with_std):
    if with_std == False:
      std = 1
    else:
      std = img.std()
    if with_mean == False:
      mean = 1
    else:
      mean = img.mean()
    # return scale(with_mean=with_mean, with_std=with_std).fit(img)
    return (img-mean)/std

  def move_axis_to_batch_minor(self, batch_major_data, batch_axis):
    #moves batch axis to the last axis
    return np.moveaxis(batch_major_data, batch_axis, -1)


  # folderpath = os.path.expanduser("vh_raw")
# filenames = ['imk00264.iml', 'imk00315.iml', 'imk00665.iml', 'imk00695.iml', 'imk00735.iml', 'imk00765.iml', 'imk00777.iml',
# 'imk00944.iml', 'imk00968.iml', 'imk01026.iml', 'imk01042.iml', 'imk01098.iml', 'imk01251.iml', 'imk01306.iml',
# 'imk01342.iml', 'imk01726.iml', 'imk01781.iml', 'imk02226.iml', 'imk02260.iml', 'imk02262.iml', 'imk02982.iml',
# 'imk02996.iml', 'imk03332.iml', 'imk03362.iml', 'imk03401.iml', 'imk03451.iml', 'imk03590.iml', 'imk03686.iml',
# 'imk03751.iml', 'imk03836.iml', 'imk03848.iml', 'imk04099.iml', 'imk04103.iml', 'imk04172.iml', 'imk04207.iml']

# filepaths  = [os.path.join(folderpath, name) for name in filenames] #os.listdir(folderpath)]
# filepaths = [x for x in filepaths if not '.DS_Store' in x]
# # random.shuffle(filepaths)

# train_length = int(1*len(filepaths))
# test_length = len(filepaths)-int(1*len(filepaths))
# print('Preallocating array...')
# train = np.empty((train_length, 1024,1024),dtype='float32')#1536-4),dtype='float32')
# print('Beginning loop...')
# for i, path in enumerate(filepaths):
#         with open(path, 'rb') as handle:
#                 s = handle.read()
#         arr = array.array('H', s)
#         arr.byteswap()
#         if i < train_length:
#                 if i%10 == 0 and i>0:
#                         print('Processing training image #{} out of {}...'.format(i, train_length))

#                 train[i, :, :] = scale(np.array(arr, dtype='uint16').reshape((1024,1536))[:,2:][:,:-
# 2].ravel()).reshape((1024,1536-4))
#         else:
#                 if i == train_length:
#                         print('Reshaping and saving raw training images...')
#                         np.savez('raw_train_no_test', move_axis_to_batch_minor(train,0))
#                         train = None
#                         test = np.empty((test_length,1024,1536-4), dtype='float32')
#                 if (i-train_length)%10 == 0 and i-train_length>0:
#                         print('Processing testing image #{} out of {}...'.format(i-train_length, tes
# t_length))
#                 test[i-train_length, :, :] = scale(np.array(arr, dtype='uint16').reshape((1024,1536)
# )[:,2:][:,:-2].ravel()).reshape((1024,1536-4))

# print('Reshaping and saving raw testing images...')
# np.savez('raw_test_no_test', move_axis_to_batch_minor(test,0))



