# GNN_Project - comp7404-group4

# Language Environment:
Python3.6.9

# Setup
pip install -r requirements.txt


# Run Demo
run three datasets by type command in terminal

1. Cora Dataset
python run.py --dataset Cora

2. CiteSeer Dataset
python run.py --dataset CiteSeer

3. PubMed Dataset
python run.py --dataset PubMed

IF user want to use different of aggregators of GraphSAGE, use command

1. gcn aggregator
python run_aggregator.py --aggregator gcn

2. MaxPooling aggregator
python run_aggregator.py --aggregator pool

3. LSTM aggregator
python run_aggregator.py --aggregator lstm

user can also add --dataset to select different dataset among Cora, CiteSeer, PubMed.
