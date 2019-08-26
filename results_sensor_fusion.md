# AMDIM
## Baseline
### Single Modality
#### RGB
* Results AMDIM - RGB - Crops - No color augmentation
  * Epoch 29, 43 updates -- 0.5194 sec/update
      * loss: 0.372, 
    * train_acc_glb_mlp: 0.955, 
    * train_acc_glb_lin: 0.936, 
    * test_acc_glb_mlp: 0.914, 
    * test_acc_glb_lin: 0.903

#### (jet-)Depth
* Results AMDIM - Depth - Crops - No color augmentation
  * Epoch 29, 43 updates -- 0.5599 sec/update
    * loss: 1.860, 
    * train_acc_glb_mlp: 0.741, 
    * train_acc_glb_lin: 0.656, 
    * test_acc_glb_mlp: 0.715, 
    * test_acc_glb_lin: 0.647

## Sensor Fusion
### RGB/Depth Self-Supervised embedding training - Random selection of RGB / (jet-)Depth
* Results AMDIM - Both modalities - No color augmentation
  * Epoch 29, 43 updates -- 0.8864 sec/update
    * loss: 0.473, 
    * train_acc_glb_mlp: 0.934, 
    * train_acc_glb_lin: 0.921, 
    * test_acc_glb_mlp: 0.810, 
    * test_acc_glb_lin: 0.807

### Both modalities for self supervised - only RGB modality in testing
  * Epoch 29, 43 updates -- 0.9963 sec/update
    * loss: 0.474, 
    * train_acc_glb_mlp: 0.934, 
    * train_acc_glb_lin: 0.921, 
    * test_acc_glb_mlp: 0.920, 
    * test_acc_glb_lin: 0.916

### Both modalities for self supervised - only Depth modality in testing
  * Epoch 29 
    * loss: 0.475, 
    * train_acc_glb_mlp: 0.935, 
    * train_acc_glb_lin: 0.922, 
    * test_acc_glb_mlp: 0.663, 
    * test_acc_glb_lin: 0.673
    
    
### Results on 26th Aug 2019 (1)
* Epoch 29 - RGB-D in representation Learning and jet-Depth in Learning of linear classifier
  * loss: 0.691
  * train_acc_glb_mlp: 0.888 
  * train_acc_glb_lin: 0.874 
  * test_acc_glb_mlp: 0.716 
  * test_acc_glb_lin: 0.706
* Epoch 29 - RGB-D in representation Learning and random selection of RGB or jet-Depth in Learning of linear classifier
  * loss: 0.636 
  * train_acc_glb_mlp: 0.904 
  * train_acc_glb_lin: 0.892
  * test_acc_glb_mlp: 0.774
  * test_acc_glb_lin: 0.767




## Timeline
* [Johannes, 26. Aug 2019]: Check the performance drop from RGB to only Depth. Are only images from one modality used for the training of the linear classifier?
  * see Results on 26th Aug 2019 (1)
* [Johannes, 26. Aug 2019]: Train AMDIM model end-to-end with supervision