# maritime_safety_security
The worl is fueled by maritime trade and we need to ensure safety and security in sea routes and ports. To do so one of the main means, is the  earliest detection of craft or vessel  approching our own vessel or entering in the harbord area. Security camera are there for this purpose but, can't be  efficiently manned by human. Artificial Intelligence is one of the solution to complement human skill.
For this project, we custom trained a YOLOV3 detection model with a six (6) classes vessel dataset available. Here are the project description:
1. find and process a suitable dataset for our purpose: ship detection<br/>
   https://www.researchgate.net/publication/327085456_SeaShips_A_Large-Scale_Precisely-Annotated_Dataset_for_Ship_Detection
   http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/datasets/SeaShips(7000).zip<br/>
   
    two utility python scripts are used to process the dataset in yolov3 format<br/>
       seaships.py to produce training  files<br/> 
       extract_xml.py to produce label file<br/>
  reference for training and dataset structure: https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9<br/>
                          
2. Custom trained YOLOV3  model on our dataset with pytorch
   For this purpose colab jupyter notebook was used: seaship_custom_train.ipynb
   you have to provide your own google drive storage with all the files<br/>
   reference:https://github.com/cfotache/pytorch_custom_yolo_training.git<br/>
3. Convert model to Tensorflow 
   for this purpose, colab jupyter notebook was used: convert_tensorflow.ipynb<br/>
   reference: https://github.com/mystic123/tensorflow-yolo-v3.git<br/>
4. Convert model to Intermediate representation (IR) with model optimizer<br/>
    python3 mo_tf.py<br/>
                    --input_model /path/to/yolo_v3.pb<br/>
                    --tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json<br/>
                     --batch 1<br/>
   yolo_v3.json<br/> 
   [<br/>
  &nbsp;&nbsp;&nbsp;{<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"id": "TFYOLOV3",<br/> 
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"match_kind": "general",<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"custom_attributes": {<br/>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"classes": 33,<br/>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"anchors": [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],<br/>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"coords": 4,<br/>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"num": 9,<br/>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"masks":[[6, 7, 8], [3, 4, 5], [0, 1, 2]],<br/>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "entry_points": ["detector/yolo-v3/Reshape", "detector/yolo-v3/Reshape_4", "detector/yolo-v3/Reshape_8"]<br/>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; }<br/>
  &nbsp;&nbsp;&nbsp;}<br/>
]<br/>
   reference: https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html#yolov3-overview<br/>
5. Build an inference engine app for  detection<br/>
the inference was build on two files: inference.py, ship_detection.py<br/>
ship_detection.py is build on top of yolov3 openvino demo<br/>
usage: ship_detection.py [-h] -m MODEL -i INPUT<br/>
&nbsp; &nbsp;  &nbsp;                  [-l CPU_EXTENSION] [-d DEVICE]<br/>
                                       [--labels LABELS] [-t PROB_THRESHOLD]<br/>
                                       [-iout IOU_THRESHOLD] [-ni NUMBER_ITER]<br/>
                                       [-pc] [-r]<br/>
Options:<br/>
     -h, --help            Show this help message and exit.<br/>
     -m MODEL, --model MODEL<br/>
                        Required. Path to an .xml file with a trained model.<br/>
  -i INPUT, --input INPUT<br/>
                        Required. Path to an image/video file. (Specify 'cam'<br/>
                        to work with camera)<br/>
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION<br/>
                        Optional. Required for CPU custom layers. Absolute<br/>
                        path to a shared library with the kernels<br/>
                        implementations.<br/>
  -d DEVICE, --device DEVICE<br/>
                        Optional. Specify the target device to infer on; CPU,<br/>
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample<br/>
                        will look for a suitable plugin for device specified.<br/>
                        Default value is CPU<br/>
  --labels LABELS       Optional. Labels mapping file<br/>
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD<br/>
                        Optional. Probability threshold for detections<br/>
                        filtering<br/>
  -iout IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD<br/>
                        Optional. Intersection over union threshold for<br/>
                        overlapping detections filtering<br/>
  -ni NUMBER_ITER, --number_iter NUMBER_ITER<br/>
                        Optional. Number of inference iterations<br/>
  -pc, --perf_counts    Optional. Report performance counters<br/>
  -r, --raw_output_message<br/>
                        Optional. Output inference results raw values showing<br/>
                        
 file out.mp4 is created and contains the detections<br/>
reference: https://docs.openvinotoolkit.org/latest/_demos_python_demos_object_detection_demo_yolov3_async_README.html<br/>
