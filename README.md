This repo contains two scripts to trial/run classification of the astrometric
quality of gaia sources. It does this with the classifier from Rybizki et al.
https://ui.adsabs.harvard.edu/abs/2021arXiv210111641R/abstract.
The code is partially copied or adapted from https://colab.research.google.com/drive/1lPzhGSSIjx2nQ7XM2v8bQZtkf0Atrk0z?usp=sharing
with the permission of the authors.

`cluster_gaia_query.py` searches for sources around globular clusters.

`verify_gaia_models.py` reads source ids from the training data to check if 
the classifier works as intended. To run it  download 
```
nn_model_v2/model_highsnr
nn_model_v2/model_lowsnr
training_data_v2/good_ids.h5
```
from 

https://keeper.mpdl.mpg.de/d/21d3582c0df94e19921d/?p=%2F&mode=list
