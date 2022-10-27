# Wheelchair_DistanceEstimation
在電動輪椅上加兩個 pi camera 鏡頭與平板即時的偵測前方物體距離，警示使用者(開發板為 Jeston Nano)
## 使用 Tenson RT :
> 1. 請自先訓練完 darknet 的 weights，以得到 **.weights** 和 **.cfg** 檔
> 2. 將 **.weights** 和 **.cfg** 檔放入```/dual_camera/yolo```裡，並修改檔名(檔名最後要是數字，為輸入的大小)
```shell
$ cd dual_camera/yolo
$ python3 yolo_to_onnx.py -c 80 -m yolov4-tiny-416
```
> (80 為 classes 的數量)
```shell
$ python3 onnx_to_tensorrt.py -c 4 -m yolov4-tiny-416
```
> (80 為 classes 的數量)
***
## Jetson Nano B01 測試連接兩塊 pi camera 開啟雙鏡頭拍照 :
```shell
$ python3 dual_camera_takepicture.py
```
***
## for dual_camera DistanceEstimation:
```shell
$ cd dual_camera
$ python3 trt_SingleDistanceEstimation_dualcam.py --onboard 0 -c 80 -m yolov4-tiny-416 --width 640 --height 480
```
***
## for dual_camera in bigmap DistanceEstimation:
```shell
$ cd dual_camera
$ python3 trt_SingleDistanceEstimation_dualcam_bigmap.py --onboard 0 -c 80 -m yolov4-tiny-416 --width 640 --height 480
```
***
## for informationIndustry mqtt :
```shell
$ cd dual_camera
$ python3 trt_SingleDistanceEstimation_dualcam_informationIndustry.py --onboard 0 -c 80 -m yolov4-tiny-416 --width 640 --height 480
```
