### install
#### python
```
conda install pytorch==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install -r requirements.txt
```

#### prepare ground truth data
```
python data.data_preparation
```

#### train
python main.py --batch_size=16 --lidar_mode=0 --device=0 --name="mtg" --data_root="./local_map_files" --snap_shot="" --data_name="data.pkl" --no_eval --lr_decay_steps=10 --grad_step=5 --collision_type=1 --dlow_type=0 --last_ratio=1000 --collision_ratio=1000
```