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
### RGB / (jet-)Depth
#### RGB/Depth Self-Supervised embedding training - Random selection of RGB / (jet-)Depth
* Results AMDIM - Both modalities - No color augmentation
  * Epoch 29, 43 updates -- 0.8864 sec/update
    * loss: 0.473, 
    * train_acc_glb_mlp: 0.934, 
    * train_acc_glb_lin: 0.921, 
    * test_acc_glb_mlp: 0.810, 
    * test_acc_glb_lin: 0.807

#### Both modalities for self supervised - only RGB modality in testing
  * Epoch 29, 43 updates -- 0.9963 sec/update
    * loss: 0.474, 
    * train_acc_glb_mlp: 0.934, 
    * train_acc_glb_lin: 0.921, 
    * test_acc_glb_mlp: 0.920, 
    * test_acc_glb_lin: 0.916
    

#### Both modalities for self supervised - only Depth modality in testing
  * Epoch 29 
    * loss: 0.475, 
    * train_acc_glb_mlp: 0.935, 
    * train_acc_glb_lin: 0.922, 
    * test_acc_glb_mlp: 0.663, 
    * test_acc_glb_lin: 0.673
    
    
#### Results on 26th Aug 2019 (1)
* Epoch 29 - RGB-D in representation Learning and jet-Depth in Learning of linear classifier
  * loss: 0.691
  * train_acc_glb_mlp: 0.888 
  * train_acc_glb_lin: 0.874 
  * test_acc_glb_mlp: 0.716 
  * test_acc_glb_lin: 0.706
* Epoch 29 - RGB-D in representation Learning and random selection of RGB or jet-Depth in Learning of linear classifier
  * loss: 0.636
  * train_acc 
    * train_acc_glb_mlp: 0.904 
    * train_acc_glb_lin: 0.892
  * test_acc
    * test_acc_glb_mlp: 0.774
    * test_acc_glb_lin: 0.767
  
 #### Results on 26th Aug 2019 with top 5 (2)
* Epoch 29 - RGB-D in representation Learning and random selection of RGB or jet-Depth in Learning of linear classifier
  * Self-supervision: RGB/ jet-Depth
  * Linear classifier: random selection of RGB / jet-Depth
  * loss: 0.649
  * train_acc
    * train_acc_glb_mlp: 0.900
    * train_acc_glb_lin: 0.888
  * test_acc 
    * test_acc_glb_mlp: 0.780
    * test_acc_glb_mlp_top_5: 0.961
    * test_acc_glb_lin: 0.774
    * test_acc_glb_lin_top_5: 0.957
* Epoch 29 - RGB-D in representation learning and using RGB information while learning the linear classifier
  * Self-supervision: RGB/ jet-Depth
  * Linear classifier: RGB
  * loss: 0.474, 
  * train_acc
    * train_acc_glb_mlp: 0.934, 
    * train_acc_glb_lin: 0.921,
  * test_acc 
    * test_acc_glb_mlp: 0.919
    * test_acc_glb_mlp_top_5: 0.985
    * test_acc_glb_lin: 0.913
    * test_acc_glb_lin_top_5: 0.985
* Epoch 29 - RGB-D in representation Learning and jet-Depth in Learning of linear classifier
  * Self-supervision: RGB/ jet-Depth
  * Linear classifier: jet-Depth
  * loss: 0.691, 
  * train_acc
    * train_acc_glb_mlp: 0.887, 
    * train_acc_glb_lin: 0.874, 
  * test_acc 
    * test_acc_glb_mlp: 0.716, 
    * test_acc_glb_mlp_top_5: 0.952, 
    * test_acc_glb_lin: 0.705, 
    * test_acc_glb_lin_top_5: 0.946


### RGB / 3x-Depth Padded

#### Self-supervised feature extraction (no training of classifier)
 * Self-supervision: RGB/ jet-Depth
 * epoch 99
 * loss_inf: 6.857
 * loss_cls: 0.761 
 * loss_g2l: 6.088 
 * lgt_reg: 0.769
 * train_acc
   * train_acc_glb_mlp: 0.890 
   * train_acc_glb_lin: 0.869 
 * test_acc_
   * test_acc_glb_mlp: 0.743
   * test_acc_glb_mlp_top_5: 0.941
   * test_acc_glb_lin: 0.726
   * test_acc_glb_lin_top_5: 0.933

#### Random selection
 * Self-supervision: RGB/ 3xDepth
 * Linear classifier: random selection of RGB / 3xDepth
 * epoch 29
 * loss: 0.697
 * train_acc
   * train_acc_glb_mlp: 0.895
   * train_acc_glb_lin: 0.881
 * test_acc
   * test_acc_glb_mlp: 0.794
   * test_acc_glb_mlp_top_5: 0.960 
   * test_acc_glb_lin: 0.786
   * test_acc_glb_lin_top_5: 0.956

#### RGB 
 * Self-supervision: RGB/ jet-Depth
 * Linear classifier: RGB

#### 3x Depth
 * Self-supervision: RGB/ jet-Depth
 * Linear classifier: 3xDepth




## Timeline
* [Johannes, 26. Aug 2019]: Check the performance drop from RGB to only Depth. Are only images from one modality used for the training of the linear classifier?
  * [x] see Results on 26th Aug 2019 (1)
* [Johannes, 26. Aug 2019]: Train AMDIM model end-to-end with supervision
* [Johannes, 27. Aug 2019]: Adopt the pre-processing for amdim training