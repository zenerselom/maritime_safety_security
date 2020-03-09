# maritime_safety_security
The worl is fueled by maritime trade and we need to ensure safety and security in sea routes and ports. To do so one of the main means, is the  earliest detection of craft or vessel  approching our own vessel or entering in the harbord area. Security camera are there for this purpose but, can't be  efficiently manned by human. Artificial Intelligence is one of the solution to complement human skill.
For this project, we custom trained a YOLOV3 detection model with a six (6) classes vessel dataset available. Here are the project description:
1. find and process a suitable dataset for our purpose: ship detection
   https://www.researchgate.net/publication/327085456_SeaShips_A_Large-Scale_Precisely-Annotated_Dataset_for_Ship_Detection
   http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/datasets/SeaShips(7000).zip<br/>
   
    two utility python scripts are used to process the dataset in yolov3 format
       seaships.py to produce training  files 
       extract_xml.py to produce label file
  reference for training and dataset structure: https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9
                          
2. Custom trained YOLOV3  model on our dataset with pytorch
   For this purpose colab jupyter notebook was used: seaship_custom_train.ipynb
   you have to provide your own google drive storage with all the files
   reference:https://github.com/cfotache/pytorch_custom_yolo_training.git
3. Convert model to Tensorflow 
   for this purpose, colab jupyter notebook was used: convert_tensorflow.ipynb
   reference: https://github.com/mystic123/tensorflow-yolo-v3.git
4. Convert model to Intermediate representation (IR) with model optimizer
    python3 mo_tf.py
                    --input_model /path/to/yolo_v3.pb
                    --tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json
                     --batch 1
   yolo_v3.json 
   [
  {
    "id": "TFYOLOV3",
    "match_kind": "general",
    "custom_attributes": {
      "classes": 33,
      "anchors": [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
      "coords": 4,
      "num": 9,
      "masks":[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
      "entry_points": ["detector/yolo-v3/Reshape", "detector/yolo-v3/Reshape_4", "detector/yolo-v3/Reshape_8"]
    }
  }
]
   reference: https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html#yolov3-overview
5. Build an inference engine app for  detection
the inference was build on two files: inference.py, ship_detection.py
ship_detection.py is build on top of yolov3 openvino demo
usage: ship_detection.py [-h] -m MODEL -i INPUT
                                       [-l CPU_EXTENSION] [-d DEVICE]
                                       [--labels LABELS] [-t PROB_THRESHOLD]
                                       [-iout IOU_THRESHOLD] [-ni NUMBER_ITER]
                                       [-pc] [-r]
Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. Path to an image/video file. (Specify 'cam'
                        to work with camera)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU
  --labels LABELS       Optional. Labels mapping file
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for detections
                        filtering
  -iout IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD
                        Optional. Intersection over union threshold for
                        overlapping detections filtering
  -ni NUMBER_ITER, --number_iter NUMBER_ITER
                        Optional. Number of inference iterations
  -pc, --perf_counts    Optional. Report performance counters
  -r, --raw_output_message
                        Optional. Output inference results raw values showing
                        
 file out.mp4 is created and contains the detections
reference: https://docs.openvinotoolkit.org/latest/_demos_python_demos_object_detection_demo_yolov3_async_README.html
