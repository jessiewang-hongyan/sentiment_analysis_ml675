# sentiment_analysis_ml675
Final project for EN.601.675. 


## Run virutal environment in VSCode:
1. Open cmd in terminal
2. input '.venv/Scripts/activate'


## Packages to install
Use pip freeze command to install packages in *requirements.txt*:

*env2/bin/python -m pip install -r requirements.txt*


## Use TensorBoard in VSCode to track model training
1. Press "Ctrl+Shift+P"
2. Click "Python: Launch TensorBoard"
3. Install plug-ins if needed
4. Train the model
5. Select 'Scalar' tab at the right top corner of the panel
6. Refersh the board
7. You will see the trends of your scalars in dynamic charts


## Use Multithreading
In *dataloader.py*, line 28-30, add argument *num_workers* for more threads.


## Use GPUs
In *main.py*, line 18, uncomment the *gpus* argument and assign correct numbers of GPUs you'd like to use.

## Upload trained models and model
1. Find the version folder you wish to upload in folder *./lightninglogs*, rename the folder to avoid overwriting
2. Add *hparams.txt* in the version folder
3. In *hparams.txt*, add epochs, model structures, hyperparameters, final train/val loss and accuracy
4. Push all files in that folder

