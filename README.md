## Classification of Social Network Node based on Graph Convolutional Network




### GCN/GraphSAGE_Project - comp7404-group4 

Team Member:  Wang Luozhou, Zhang He, Ji Yeon Fung, Ren Xinzhu

#### Language Environment: 
Python3.6.9 
 
#### Setup 
```
pip install -r requirements.txt 
``` 

PS:
If you report an error in the dgl_cu101 ==0.5.2 command, please delete it and run the following command:
```
pip install --verbose --no-cache-dir torch-sparse torch-scatter
```
Because you probably don't have a graphics card or version of the driver, you cannot use CUDA to train your model. Therefore, you'll need to install the environment packages(Torch-Sparse and Scatter) to make sure that the code works (This model also can be trained directly with the CPU).

#### Run Demo 
Run three datasets by type command in terminal 
  
1. Cora Dataset
```
python run.py --dataset Cora 
```  
2. CiteSeer Dataset 
```
python run.py --dataset CiteSeer 
```  
3. PubMed Dataset 
```
python run.py --dataset PubMed 
```
The output will plot a curve of accuracy and loss firstly. When user switches off this window, the output will be two figures showing nodes position through t-SNE visualization before input into model and after respectively. Here are samples.

<img src="https://github.com/ZHANGHE24/Classification-of-Social-Network-Node-based-on-Graph-Convolutional-Network/blob/main/image/Cora.png" width="50%" height="50%">
<img src="https://github.com/ZHANGHE24/Classification-of-Social-Network-Node-based-on-Graph-Convolutional-Network/blob/main/image/untrained_Cora.png" width="50%" height="50%">
<img src="https://github.com/ZHANGHE24/Classification-of-Social-Network-Node-based-on-Graph-Convolutional-Network/blob/main/image/trained_Cora.png" width="50%" height="50%">


#### If user want to use different of aggregators of GraphSAGE, use command 
  
1. gcn aggregator 
```
python run_aggregator.py --aggregator gcn 
```  
2. MaxPooling aggregator 
```
python run_aggregator.py --aggregator pool 
```  
3. LSTM aggregator 
```
python run_aggregator.py --aggregator lstm 
```
#### User can also add --dataset to select different dataset among Cora, CiteSeer, PubMed. 


