# MAGNN
This repository is an implementation of our proposed Multiple-aspect Attentional Graph Neural Networks (MAGNN) model in the following paper:

``Multiple-aspect Attentional Graph Neural Networks for Online Social Network User Localization``

# Requirements
python 3.7

tensorflow 1.14.0

check requirements.txt for more details.

# File Structure

- geo_data/cmu/ : the processed data of CMU (i.e., GeoText).
- geo_data/model/ : the trained model.
- models/ : the base attention module.
- utils/ : some utils to process data.
- MAGNN_for_geo.py : the main file.
- MAGNN_for_geo_sp.py : the sparse format main file for lager memory require.
- MAGNN_for_geo_sp_na.py : the main file for NA (i.e., Twitter-US).
- geo_dataProcess.py : the data process code.
- requirements.txt : the main requirement environment list. 

