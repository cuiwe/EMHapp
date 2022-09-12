# About EMHapp
A pipeline for automatic detection, visualization and localization of epileptic magnetoencephalography high-frequency oscillations.

GUI was programed using App designer in MATALB.   
    
EMHapp pipeline was  programed using python, and accelerated by GPU with functions of pyTorch. 
    
# Install EMHapp

1. Download EMHapp code via github.

2. Install MATLAB toolbox:

    Brainstorm3: https://neuroimage.usc.edu/brainstorm/Introduction.
  
    SPM12: https://www.fil.ion.ucl.ac.uk/spm/.
  
    FieldTrip: https://www.fieldtriptoolbox.org/.

3. Install python packages (Nvidia drive version is 10.2): 

   ```shell
   conda env create -f requirements.yml
   ```

# USE EMHapp

A walkthrough video of EMHapp can be found at https://www.youtube.com/watch?v=Eq_fxFGbRiY or https://www.bilibili.com/video/BV1nY411c7H5 (The data process in this test was used Nvidia V100 GPU).

1. Open EMHapp GUI (run in MATLAB):

```matlab
EMHapp
```

2. Press "Load Button" to load project:


   1). Press "Add Button" and select anatomy file (MRI or FreeSurfer).

   3). Set NAS, LPA and RPA fiducial points for each subject.

   4). Press "Add Button" and select MEG file for each Subject.

   5). Press "OK Button" and check database.
   
3. Press "Process Button" to set parameters and run pipeline:

   1). Press "Set parameters of process Buttom" to set process parameters, and EMHapp will automaticly load data.
   
   2). Select subjects that need to be processed.

   3). Select process localy or remotely. If process remotely, Press "Add Server Buttom" to input info of remote server (ip, user name, passwd, matlab dir, project dir).

   4). Select devices to do process. If multipal devices were selected, the subject data will be processed parallelly. 
   
   5). Press "OK Buttom" to run process. If process remotely, data will be uploaded to remote servers.
   
4. Press "Process Status Button" to check status of process (output of log) in each device and download data if process remotely.

5. Select subject data for visualization.

6. Press "Spike Button" to view IED detection results.

7. Press "HFO Button" to view IED detection results.

8. Press "Show all HFO source Button" to view source map of all checked HFO events.
   
