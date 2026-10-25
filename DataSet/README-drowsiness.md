# Driver Monitoring Dataset Guide
## Drowsiness-related Material
#### Last modification: 27/03/2025

Thanks for downloading the DMD! :)

This dataset is public and free for academic use. For easier access, the data been **divided** and organized by its primary applications, offered through **three different downloading services**. This repository specifically contains data related to **drowsiness estimation**. This is the [*Gaze*](https://opendatasets.vicomtech.org/di21-dmd-dataset-gaze/8433939e) downloading form. Distraction part have been splitted in two for better downloading experience. These are the forms for [*Distraction (RGB-IR)*](https://opendatasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir/a268f10c) and [*Distraction (Depth)*](https://opendatasets.vicomtech.org/di21-dmd-dataset-distraction-depth/779bd812).

Drowsiness estimation data contains videos of drivers showing signs of fatigue such as microsleeps and yawning. These states were acted, drivers were not actually tired or falling sleep. 

## Content
- [Introduction](#introduction)
- [DMD Structure](#dmd-structure)
- [Annotations](#annotations)
  - [Labels and criterion](#labels-and-criterion)
  - [Format and access](#format-and-access)
- [Dataset Specifications and Set-up](#dataset-specifications-and-set-up)
  - [s5](#S5)
  - [Recordings environments](#recording-environments)
  - [Cameras](#cameras)
  - [Participants](#participants)
- [Dataset Post-Processing and Annotation](#dataset-post-processing-and-annotation)
  - [Raw to videos](#raw-to-videos)
  - [Three-camera videos alignment](#three-camera-videos-alignment)
- [License](#license)


## Introduction

The [Driver Monitoring Dataset (DMD)](https://dmd.vicomtech.org/) was created to address the shortage of data for training and testing driver monitoring systems based on computer vision. This dataset includes data for tasks such as distraction detection, drowsiness detection, and gaze estimation. It enables both temporal and discrete multi-sensor analysis, as it is provided in video format with information from three cameras, each capturing RGB, IR, and depth data.

Processing large volumes of information, particularly annotating, is time-consuming. As we progressed with the data, we have continued to publish additional material. All the videos have been processed, but some work is still to be done around annotations. Right now, the focus is on temporal annotations. We are still working to offer more.

There is a [GitHub repository](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset) with tools and documentation you might find useful, please check the Wiki. There is a new Known Issues section in the [readme](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset?tab=readme-ov-file#known-issues).

Check our [website](https://dmd.vicomtech.org/) to read our scientific papers and don't forget to cite us! 

This document hopes to be a guide for the understanding and use of the DMD. 

## DMD Structure

The material is organized to be easily used, but it can be confusing. As shown in the Figure below, the material is first divided into groups of subjects (gA, gB, gC, etc.). Inside each group, there is the material of about 5 participants. Inside a subject folder, the material is organized by recording sessions (s1, s2, s3, etc.). For **drowsines**, there is only **s5**. Inside the recording session folder, there are the videos from three cameras, three streams and a mosaic video, and the annotation file (json). All of the files follow the same nomenclature as shown in the Figure. 

![DMD Structure](https://raw.githubusercontent.com/Vicomtech/DMD-Driver-Monitoring-Dataset/refs/heads/master/docs/readme-assets/dmdStructure.png)

## Annotations

### Labels and criterion

The defined levels describe temporal actions or events which occur when the driver is performing some drowsiness-related actions. To annotate temporal actions, we defined 4 levels of annotations. Basically, they are annotations that can simultaneously be present and describe one frame. Each level of annotation has its own set of labels. Within each level, the labels are mutually exclusive, meaning that, for each level a maximum of one label is allowed.

The drowsiness-related annotation levels are:
 - Occlusion
    - Face occlusion	
    - Body occlusion
    - Hands occlusion
 - Eyes state
    - Openend
    - Closed
    - Opening
    - Closing
 - Blinks
    - Blinking
 - Yawning
    - Yawning with hand
    - Yawning without hand


To perform the temporal action annotation of the DMD we have defined an **annotation criterion** that has to be followed by all annotators to guarantee consistent annotation production. Here, there is the document for [drowsiness annotations](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset/wiki/DMD-drowsiness-related-action-annotation-criteria).
### Format and access

A tool [(TaTo)](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset/tree/master/annotation-tool) was developed to annotate temporal actions in the DMD dataset.

The DMD annotations are in ASAM [OpenLABEL](https://www.asam.net/standards/detail/openlabel/) description format based on a JSON Schema. To create annotations, the [VCD (Video content description)](https://vcd.vicomtech.org/) library was used. It can include descriptions of actions, objects, relationships, and events, all in one file.

On the other hand, a tool to access the annotations easily was built. It is called [DEx Tool](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset/tree/master/exploreMaterial-tool). It offers to prepare DMD data for training. There are configuration variables that can be changed to extract specific material. 

If you wish, you can also get the annotations direclty from the .json file. It will be a little bit confusing, but inside the file, there is information about each frame (inside "frames" object). Each frame in the "frames" list shows the ID of the action that is present in that frame. To know the name of the action, you can go to the "actions" object list in the JSON, and you will find the name in the "type" field. Also, each "action" object has the frame intervals in which that action is present in the "frame_intervals" field. This way, annotations can be accessed by frame or by action. For a better understanding, check the documentation of the ASAM OpenLabel format.

## Dataset Specifications and Set-up

A protocol was created for the recordings, trying to capture different activities with different conditions in every recording. These were made continuously; this means that after performing one activity, the driver starts a new one, as realistically as possible, until the completion of all the activities within a protocol. Below is the description of recording session s6, including a list of the recorded activities. Note that the activities are not equal to the list of annotations.

### S5: 
In this session, all the **drowsiness** activities are recorded. By the time of the writing of this document, this was only performed in an unmoving real car, not the simulator. The activities are: 
 - Safe driving
 - Sleepy driving 
 - Yawn no hand
 - Yawn with hand
 - Microsleep

### Recording environments

For this project, two environments were prepared to accomplish the recordings for the dataset: a real car and a simulator. The performance of some of the activities that wanted to be included in the dataset, in a real scenario, could be somehow illegal or dangerous to perform. For example, is not legal to text while driving. Hence, to still have the real experience and performance of the activities, it was proposed to have both a properly equipped car and a simulator. 

![DMD Structure](https://raw.githubusercontent.com/Vicomtech/DMD-Driver-Monitoring-Dataset/refs/heads/master/docs/readme-assets/environments.png)

### Cameras

For the DMD, we used three devices installed in both environments, the car and simulator, and placed them correctly to capture images of the driver’s face, body and hands. All cameras recorded RGB, IR and depth information. 

![alt text](https://raw.githubusercontent.com/Vicomtech/DMD-Driver-Monitoring-Dataset/refs/heads/master/docs/readme-assets/cameras.png)

All the devices are configured to work with a considerable resolution and exposure level without risking the frame rate. This way, the images obtained fit the dataset requirements. The resolution was set to 1280x720 px in all channels. This resolution is offered in the videos, even if it implies heavier files.

### Participants


37 experienced drivers volunteered for the project, as represented in the figure below.

![alt text](https://raw.githubusercontent.com/Vicomtech/DMD-Driver-Monitoring-Dataset/refs/heads/master/docs/readme-assets/participants.png)

All participants were grouped into six groups of 5 and one of 7 people. Knowing the availability of the participants, first, they were organized by groups depending on the recording sessions they could attend (morning, afternoon or both). Then, a double participation schedule was assigned to 3 groups of volunteers whose availability was "both". All the protocols were recorded twice for these participants, one in the morning and another in the afternoon, not exactly on the same date. This is because the variability in lighting was important. Therefore, you will find groups that have a double size of data. 20% of the data is reserved for possible benchmarking purposes and **will not be public**. 

## Dataset Post-Processing and Annotation

### Raw to videos

First, each frame was exported into an image in png format. Images were converted into video files using H.264 codec (libx264 library). The videos are created at the rate they were originally recorded with. For the body camera, the fps was 29.98, and the face and hands camera was 29.76, so the videos are created with those fps. 
For the next step, the frame rate of all videos must be the same. Therefore, the actual body video is processed with FFMPEG to have 29.76fps, creating another video with this frame rate. If you want to know more of how to process depth data, access our [Wiki](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset/wiki/DMD-Depth-Material).

### Three-camera videos alignment

The next thing in the process is to align the three perspective videos. The cameras did not start recording simultaneously, which is why there is a shift between each video. With a script, it was possible to synchronize the cameras. As a result, we obtained the shifts between the three cameras. This way, knowing the starting order of the videos and their offsets between them, a mosaic (see Figure below) is created in which the three cameras are aligned. These mosaics are available in the dataset, and it was used to annotate.

![alt text](https://raw.githubusercontent.com/Vicomtech/DMD-Driver-Monitoring-Dataset/refs/heads/master/docs/readme-assets/mosaic.png)

## License

All datasets are copyright by Vicomtech and published under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License](https://creativecommons.org/licenses/by-nc-nd/4.0/). This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you remix, transform, or build upon the material, you may not distribute the modified material.

