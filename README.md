## ViTCatsVDogs
Cats &amp; Dogs Image Classification using Vision Transformers


## To run
> % run main.py

# Modify params.json for the following

* training image folder
* epochs
* LR
* mean & std
* batch size



## Final 5 epochs

EPOCH: 6
100%|██████████| 157/157 [01:14<00:00,  2.12it/s]
  0%|          | 0/157 [00:00<?, ?it/s]Epoch : 7 - loss : 0.2399 - acc: 0.9014 - val_loss : 0.9127 - val_acc: 0.6396

EPOCH: 7
100%|██████████| 157/157 [01:14<00:00,  2.11it/s]
  0%|          | 0/157 [00:00<?, ?it/s]Epoch : 8 - loss : 0.1896 - acc: 0.9251 - val_loss : 1.0679 - val_acc: 0.6367

EPOCH: 8
100%|██████████| 157/157 [01:13<00:00,  2.14it/s]
  0%|          | 0/157 [00:00<?, ?it/s]Epoch : 9 - loss : 0.1519 - acc: 0.9416 - val_loss : 1.1603 - val_acc: 0.6488

EPOCH: 9
100%|██████████| 157/157 [01:13<00:00,  2.13it/s]
Epoch : 10 - loss : 0.1336 - acc: 0.9479 - val_loss : 1.1993 - val_acc: 0.6436


## Logs

[![image.png](https://i.postimg.cc/qvkVVsWy/image.png)](https://postimg.cc/zVPPTgzG)


## Notes
We notice poor model performance, this is because the model was trained from scratch
and Vision Transformers require pretraining with a large dataset to work properly
due to fewer inductive bias as compared to CNNs
