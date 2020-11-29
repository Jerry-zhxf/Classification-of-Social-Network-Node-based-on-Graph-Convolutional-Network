## Classification of Social Network Node based on Graph Convolutional Network


### GCN/GraphSAGE_Project - comp7404-group4 

   
#### Language Environment: 
Python3.6.9 
 
#### Setup 
```
pip install -r requirements.txt 
``` 
   
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

![image](https://github.com/ZHANGHE24/Classification-of-Social-Network-Node-based-on-Graph-Convolutional-Network/tree/main/image/Cora.png)
![image](https://github.com/ZHANGHE24/Classification-of-Social-Network-Node-based-on-Graph-Convolutional-Network/tree/main/image/untrained_Cora.png)
![image](https://github.com/ZHANGHE24/Classification-of-Social-Network-Node-based-on-Graph-Convolutional-Network/tree/main/image/trained_Cora.png)


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
#### user can also add --dataset to select different dataset among Cora, CiteSeer, PubMed. 


