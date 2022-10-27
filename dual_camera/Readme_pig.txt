CSI camera :
$ python3 trt_SingleDistanceEstimation.py --onboard 0 -c 80 -m yolov4-tiny-416   (coco)



for dual_camera DistanceEstimation:
$ python3 trt_SingleDistanceEstimation_dualcam.py --onboard 0 -c 80 -m yolov4-tiny-416 --width 640 --height 480

for dual_camera in bigmap DistanceEstimation:
$ python3 trt_SingleDistanceEstimation_dualcam_bigmap.py --onboard 0 -c 80 -m yolov4-tiny-416 --width 640 --height 480

for party hot_spot mqtt :
$ python3 trt_SingleDistanceEstimation_dualcam_bigmap_party.py --onboard 0 -c 80 -m yolov4-tiny-416 --width 640 --height 480

for informationIndustry mqtt :
$ python3 trt_SingleDistanceEstimation_dualcam_informationIndustry.py --onboard 0 -c 80 -m yolov4-tiny-416 --width 640 --height 480

