def download_train_export_model(run_number,MODEL,DOWNLOAD_BASE,num_steps=5000):
  
  import shutil
  import re
  import subprocess
  import os
  import glob
  import urllib
  import tarfile

  if run_number==0:
    DEST_DIR = 'trained_models/run00/'
    os.makedirs(DEST_DIR)
    
    MODEL_FILE = MODEL + '.tar.gz'
    
    if not (os.path.exists(MODEL_FILE)):
      opener = urllib.URLopener()
      opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

    tar = tarfile.open(MODEL_FILE)
    tar.extractall()
    tar.close()

    os.remove(MODEL_FILE)
    if (os.path.exists(DEST_DIR)):
      shutil.rmtree(DEST_DIR)
    os.rename(MODEL, DEST_DIR)  
    print('Downloaded model')
  
  print('run{0:02d} started.'.format(run_number))
  
  ## Copy config file
  shutil.copyfile('models/research/object_detection/samples/configs/faster_rcnn_inception_v2_pets.config','trained_models/run{0:02d}/run{0:02d}.config'.format(run_number))

  ## Adjust config file
  filename = 'trained_models/run{0:02d}/run{0:02d}.config'.format(run_number,run_number)
  with open(filename) as f:
    s = f.read()
  with open(filename, 'w') as f:
    s = re.sub('PATH_TO_BE_CONFIGURED/model.ckpt', 'trained_models/run{0:02d}/model.ckpt'.format(run_number), s)
    s = re.sub('PATH_TO_BE_CONFIGURED/pet_faces_train.record-\?\?\?\?\?-of-00010', 'datalab/tf_train.record', s)
    s = re.sub('PATH_TO_BE_CONFIGURED/pet_faces_val.record-\?\?\?\?\?-of-00010', 'datalab/tf_val.record', s)
    s = re.sub('PATH_TO_BE_CONFIGURED/pet_label_map.pbtxt', 'datalab/label_map.pbtxt', s)
    s = re.sub("num_classes: 90", "num_classes: 1", s)
    s = re.sub("num_examples: 1101", "num_examples: 20", s)
    s = re.sub("num_steps: 200000", "num_steps: {}".format(num_steps), s) # 10.000 = 1 uur?
    f.write(s)  
  
  ## Train model
  process = 'python models/research/object_detection/model_main.py \
    --pipeline_config_path=trained_models/run{0:02d}/run{1:02d}.config \
    --model_dir=working/run{2:02d} \
    --alsologtostderr'.format(run_number,run_number,run_number)
  output = subprocess.check_output(process, shell=True)
  #print(output)
  print('run{0:02d} finished training.'.format(run_number))
  
  ## Export inference graph
  lst = os.listdir('working/run00')
  lf = filter(lambda k: 'model.ckpt-' in k, lst)
  last_model = sorted(lf)[-1].replace('.meta', '')

  process = 'python models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=trained_models/run{0:02d}/run{1:02d}.config \
    --output_directory=trained_models/run{2:02d} \
    --trained_checkpoint_prefix=working/run{3:02d}/{4}'.format(run_number,run_number,run_number+1,run_number,last_model)
  output = subprocess.check_output(process, shell=True)
  #print(output)
  print('run{0:02d} finished exporting inference graph.'.format(run_number))
