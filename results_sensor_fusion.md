# AMDIM
## Baseline - Supervision
### Single Modality
#### RGB - Fully Supervised
* epoch 149
* loss: 0.032
* train_acc
  * train_acc_glb_mlp: 0.990
* test_acc
  * test_acc_glb_mlp: 0.932
  * test_acc_glb_mlp_top_5: 0.992

#### jet-Depth (padded) - Fully Supervised
* epoch 149
* loss: 0.581
* train_acc
  * train_acc_glb_mlp: 0.793
* test_acc
  * test_acc_glb_mlp: 0.835
  * test_acc_glb_mlp_top_5: 0.984

#### 3xDepth (padded) - Fully Supervised
* epoch 149
* loss: 0.851
* train_acc
  * train_acc_glb_mlp: 0.705
* test_acc
  * test_acc_glb_mlp: 0.792
  * test_acc_glb_mlp_top_5: 0.977

## Baseline - AMDIM
### Single Modality
#### RGB (not padded)
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
    

#### RGB - Padded images
##### Self-supervised training
* epoch 99
* loss_inf: 3.164
* loss_cls: 0.362
* loss_g2l: 2.572
* lgt_reg: 0.592
* train_acc
  * train_acc_glb_mlp: 0.964
  * train_acc_glb_lin: 0.936
* test_acc
  * test_acc_glb_mlp: 0.914
  * test_acc_glb_lin: 0.902
  * test_acc_glb_mlp_top_5: 0.986
  * test_acc_glb_lin_top_5: 0.985

#### Linear classifier
* epoch 29
* loss: 0.342
* train_acc
  * train_acc_glb_mlp: 0.962
  * train_acc_glb_lin: 0.942
* test_acc
  * test_acc_glb_mlp: 0.914
  * test_acc_glb_lin: 0.909
  * test_acc_glb_mlp_top_5: 0.988
  * test_acc_glb_lin_top_5: 0.989
    
#### 3xDepth
##### Self-supervised training
* epoch 99
* loss_inf: 3.976
* loss_cls: 1.763
* loss_g2l: 3.334
* lgt_reg: 0.641
* train_acc
  * train_acc_glb_mlp: 0.786
  * train_acc_glb_lin: 0.665
* test_acc
  * test_acc_glb_mlp: 0.741
  * test_acc_glb_lin: 0.647
  * test_acc_glb_mlp_top_5: 0.976
  * test_acc_glb_lin_top_5: 0.939

#### Linear classifier
* epoch 29 
* loss: 1.727
* train_acc
  * train_acc_glb_mlp: 0.766
  * train_acc_glb_lin: 0.690
* test_acc
  * test_acc_glb_mlp: 0.731
  * test_acc_glb_lin: 0.668
  * test_acc_glb_mlp_top_5: 0.970
  * test_acc_glb_lin_top_5: 0.945

##### run 2
* epoch 29
* loss: 1.724
* train_acc
  * train_acc_glb_mlp: 0.769
  * train_acc_glb_lin: 0.691
* test_acc
  * test_acc_glb_mlp: 0.731
  * test_acc_glb_lin: 0.665
  * test_acc_glb_mlp_top_5: 0.972
  * test_acc_glb_lin_top_5: 0.948
  
##### run 3 - more epochs
* epoch 59
* loss: 1.690
* train_acc
  * train_acc_glb_mlp: 0.777
  * train_acc_glb_lin: 0.693
* test_acc
  * test_acc_glb_mlp: 0.734
  * test_acc_glb_lin: 0.670
  * test_acc_glb_mlp_top_5: 0.973
  * test_acc_glb_lin_top_5: 0.946
  
##### run 4 - 120 Epochs
* epoch 119
* loss: 1.631
* train_acc
  * train_acc_glb_mlp: 0.788
  * train_acc_glb_lin: 0.697
* test_acc
  * test_acc_glb_mlp: 0.741
  * test_acc_glb_lin: 0.672
  * test_acc_glb_mlp_top_5: 0.973
  * test_acc_glb_lin_top_5: 0.946
  
### 3x-Depth - 300 epochs
#### Self-Supervised
* epoch 299
* loss_inf: 3.292
* loss_cls: 1.610
* loss_g2l: 2.688
* lgt_reg: 0.604
* train_acc
  * train_acc_glb_mlp: 0.808
  * train_acc_glb_lin: 0.689
* test_acc
  * test_acc_glb_mlp: 0.758
  * test_acc_glb_lin: 0.667
  * test_acc_glb_mlp_top_5: 0.974
  * test_acc_glb_lin_top_5: 0.945

#### Linear Classifier
* epoch 29
* loss: 1.492
* train_acc
  * train_acc_glb_mlp: 0.800
  * train_acc_glb_lin: 0.731
* test_acc
  * test_acc_glb_mlp: 0.757
  * test_acc_glb_lin: 0.699
  * test_acc_glb_mlp_top_5: 0.977
  * test_acc_glb_lin_top_5: 0.956


---

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
    * train_acc_glb_lin: 0.921
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
 * Self-supervision: RGB / 3xDepth
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
 * Self-supervision: RGB/ 3xDepth
 * Linear classifier: RGB
 * epoch 29
 * loss: 0.503,
 * train_acc 
   * train_acc_glb_mlp: 0.934, 
   * train_acc_glb_lin: 0.918, 
 * test_acc
   * test_acc_glb_mlp: 0.917
   * test_acc_glb_mlp_top_5: 0.987
   * test_acc_glb_lin: 0.913 
   * test_acc_glb_lin_top_5: 0.988


#### 3xDepth
 * Self-supervision: RGB / 3xDepth
 * Linear classifier: 3xDepth
 * epoch 29
 * loss: 0.771
 * train_acc
   * train_acc_glb_mlp: 0.875
   * train_acc_glb_lin: 0.858
 * test_acc
   * test_acc_glb_mlp: 0.648
   * test_acc_glb_mlp_top_5: 0.934
   * test_acc_glb_lin: 0.632
   * test_acc_glb_lin_top_5: 0.927
   
   
   
### RGB / 3x-Depth Padded - Verification 
#### Self-Supervision training 
* 99
* loss_inf: 5.632
* loss_cls: 0.708
* loss_g2l: 4.823
* lgt_reg: 0.809
* train_acc
  * train_acc_glb_mlp: 0.897
  * train_acc_glb_lin: 0.878
* test_acc
  * test_acc_glb_mlp: 0.878
  * test_acc_glb_lin: 0.866
  * test_acc_glb_mlp_top_5: 0.988
  * test_acc_glb_lin_top_5: 0.985

#### Random
* epoch 29
* loss: 0.659
* train_acc
    * train_acc_glb_mlp: 0.898
    * train_acc_glb_lin: 0.890
* test_acc
    * test_acc_glb_mlp: 0.883
    * test_acc_glb_lin: 0.877
    * test_acc_glb_mlp_top_5: 0.988
    * test_acc_glb_lin_top_5: 0.986

#### RGB
* epoch 29
* loss: 0.507
* train_acc
  * train_acc_glb_mlp: 0.931
  * train_acc_glb_lin: 0.918
* test_acc
  * test_acc_glb_mlp: 0.922
  * test_acc_glb_lin: 0.918
  * test_acc_glb_mlp_top_5: 0.989
  * test_acc_glb_lin_top_5: 0.989

#### 3xDepth
* epoch 29
* loss: 0.712
* train_acc
    * train_acc_glb_mlp: 0.889
    * train_acc_glb_lin: 0.874
* test_acc
    * test_acc_glb_mlp: 0.829
    * test_acc_glb_lin: 0.824
    * test_acc_glb_mlp_top_5: 0.984
    * test_acc_glb_lin_top_5: 0.981
    

### RGB / jet-Depth 
#### Self-Supervised training 
* epoch 99
* loss_inf: 6.054
* loss_cls: 0.739
* loss_g2l: 5.248
* lgt_reg: 0.805
* train_acc
  * train_acc_glb_mlp: 0.890
  * train_acc_glb_lin: 0.872
* test_acc
  * test_acc_glb_mlp: 0.870
  * test_acc_glb_lin: 0.853
  * test_acc_glb_mlp_top_5: 0.984
  * test_acc_glb_lin_top_5: 0.981

#### RGB
* epoch 29
* loss: 0.590
* train_acc
  * train_acc_glb_mlp: 0.914
  * train_acc_glb_lin: 0.902
* test_acc
  * test_acc_glb_mlp: 0.918
  * test_acc_glb_lin: 0.910
  * test_acc_glb_mlp_top_5: 0.988
  * test_acc_glb_lin_top_5: 0.987


#### jetd
* epoch 29
* loss: 1.135
* train_acc
  * train_acc_glb_mlp: 0.821
  * train_acc_glb_lin: 0.801
* test_acc
  * test_acc_glb_mlp: 0.852
  * test_acc_glb_lin: 0.837
  * test_acc_glb_mlp_top_5: 0.988
  * test_acc_glb_lin_top_5: 0.986


## Timeline
* [Johannes, 26. Aug 2019]: Check the performance drop from RGB to only Depth. Are only images from one modality used for the training of the linear classifier?
  * [x] see Results on 26th Aug 2019 (1)
* [Johannes, 26. Aug 2019]: Train AMDIM model end-to-end with supervision
* [Johannes, 27. Aug 2019]: Adopt the pre-processing for amdim training
  * [x] see "RGB / 3x-Depth Padded"
* Verify the results above
  * [x] see "RGB / 3x-Depth Padded - Verification"