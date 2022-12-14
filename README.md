# sentiment_analysis_ml675
Final project for EN.601.675. 

Runable colab notebook: https://colab.research.google.com/drive/1cEBcIBwoRyPbYammOuuY2SuTNC-TCmQ7?usp=sharing

Fine-tuning excel: https://docs.google.com/spreadsheets/d/1hs2sxvi-YK2UUQlPB3JRQbNBst0QiHt1D1fjKly2XVA/edit?usp=sharing


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

## Upload trained stats
1. Find the version folder you wish to upload in folder *./lightninglogs*, rename the folder to avoid overwriting
2. Add *hparams.txt* in the version folder
3. In *hparams.txt*, add epochs, model structures, hyperparameters, final train/val loss and accuracy
4. Push all files except the checkpoint in that folder

**Important**: Please do not upload checkpoint to github, as the file is too large to push

## Hyperparameter Tuning
1. Epoch: main.py
2. Optimizer & learning rate: trainer.py
3. Batch size: dataloader.py

## TODO: 
1. predict function - wyj
2. 80% accuracy model - 3
3. data visualization - zq
4. optimizer tuning - 
5. ppt