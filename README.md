# maritime_safety_security
The worl is fueled by maritime trade and we need to ensure safety and security in sea routes and ports. To do so one of the main means, is the  earliest detection of craft or vessel  approching our own vessel or entering in the harbord area. Security camera are there for this purpose but, can't be  efficiently manned by human. Artificial Intelligence is one of the solution to complement human skill.
For this project, we custom trained a YOLOV3 detection model with a six (6) classes vessel dataset available. Here are the project description:
1. find and process a suitable dataset for our purpose: ship detection<br/>
   https://www.researchgate.net/publication/327085456_SeaShips_A_Large-Scale_Precisely-Annotated_Dataset_for_Ship_Detection
   http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/datasets/SeaShips(7000).zip<br/>
   
    two utility python scripts are used to process the dataset in yolov3 format<br/>
       [seaships.py](https://github.com/zenerselom/maritime_safety_security/blob/master/seaships.py) to produce training  files<br/> 
       [extract_xml.py](https://github.com/zenerselom/maritime_safety_security/blob/master/extract_xml.py) to produce label file<br/>
  reference for training and dataset structure: https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9<br/>
                          
2. Custom trained YOLOV3  model on our dataset with pytorch
   For this purpose colab jupyter notebook was used: [seaship_custom_train.ipynb](https://github.com/zenerselom/maritime_safety_security/blob/master/seaship_custom_train.ipynb)
   you have to provide your own google drive storage with all the files<br/>
   reference:https://github.com/cfotache/pytorch_custom_yolo_training.git<br/>
3. Convert model to Tensorflow 
   for this purpose, colab jupyter notebook was used: [convert_tensorflow.ipynb](https://github.com/zenerselom/maritime_safety_security/blob/master/convert_tensorflow.ipynb)<br/>
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
link for IR models : https://drive.google.com/open?id=1M1Zb3N6K_sh-lCmEJqj0Hp740xby0jet<br/>

5. Build an inference engine app for  detection<br/>
the inference was build on two files: [inference.py](https://github.com/zenerselom/maritime_safety_security/blob/master/inference.py), [ship_detection.py](https://github.com/zenerselom/maritime_safety_security/blob/master/ship_detection.py)<br/>
ship_detection.py is build on top of yolov3 openvino demo<br/>
usage: ship_detection.py [-h] -m MODEL -i INPUT<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-l CPU_EXTENSION] [-d DEVICE]<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-l CPU_EXTENSION] [-d DEVICE]<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[--labels LABELS] [-t PROB_THRESHOLD]<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-iout IOU_THRESHOLD] [-ni NUMBER_ITER]<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-pc] [-r]<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Options:<br/>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-h, --help            Show this help message and exit.<br/>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-m MODEL, --model MODEL<br/>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                   Required. Path to an .xml file with a trained model.<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-i INPUT, --input INPUT<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                    Required. Path to an image/video file. (Specify 'cam'<br/>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                   to work with camera)<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                       Optional. Required for CPU custom layers. Absolute<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                       path to a shared library with the kernels<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                       implementations.<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-d DEVICE, --device DEVICE<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                      Optional. Specify the target device to infer on; CPU,<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                      GPU, FPGA, HDDL or MYRIAD is acceptable. The sample<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                      will look for a suitable plugin for device specified.<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                      Default value is CPU<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--labels LABELS       Optional. Labels mapping file<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                      Optional. Probability threshold for detections<br/>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                     filtering<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-iout IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD<br/>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                     Optional. Intersection over union threshold for<br/>
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;overlapping detections filtering<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-ni NUMBER_ITER, --number_iter NUMBER_ITER<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                      Optional. Number of inference iterations<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-pc, --perf_counts    Optional. Report performance counters<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-r, --raw_output_message<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Optional. Output inference results raw values showing<br/>
                       
                        
 file out.mp4 is created and contains the detections<br/>
reference: https://docs.openvinotoolkit.org/latest/_demos_python_demos_object_detection_demo_yolov3_async_README.html<br/>
To do : <br/>
train a better model with larger dataset. Due to time constraint and computing ressource, this one were train on 20 epoch (2200 iterations). We are targeting 12000 iterations at least 
implement realtime detection<br/>


DEMO VIDEO:<br/>
[demo_video_0.8_0.4](https://github.com/zenerselom/maritime_safety_security/blob/master/out.mp4) Confidence_threshold = 0.8 and IOU = 0.4<br/>
[demo_video_0.9_0.4](https://github.com/zenerselom/maritime_safety_security/blob/master/out_90_01.mp4) Confidence_threshold = 0.9 and IOU = 0.1<br/>
