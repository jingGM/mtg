#### [Video](https://youtu.be/3eJ2soAzXnU?si=PFYqICdi3hfB72Ky)

### install
#### python
```
conda install pytorch==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install -r requirements.txt
```

Download the data folder in the root:
https://drive.google.com/drive/folders/1qEn5OcDVy2jbySMp207AtSzMQcuBv3nP?usp=sharing

#### train
```
python main.py --batch_size=16 --lidar_mode=0 --device=0 --name="mtg" --w_others
```