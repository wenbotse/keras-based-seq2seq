# keras-based-seq2seq


seq2seq_large_scale_dataset.py

遇到的问题：利用fit_generator 把train_data 按batch_size送入train，但是为啥Losss一直上升？？
解决：
encoder_input_data[:]=0
decoder_input_data[:]=0
decoder_target_data[:]=0
yield了之后，没有将data array reset ,导致最后，所有的位都被设置为1
