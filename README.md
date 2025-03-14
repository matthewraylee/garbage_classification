Initial setup:

```
poetry install
```

Import the data:

```
kaggle datasets download asdasdasasdas/garbage-classification -p garbage_classification/data/ --unzip
```

To run the app, use the following command:

```
streamlit run app.py
```

# run conda

conda activate yolov8_env
conda deactivate

# run model

yolo task=detect mode=train model=yolov8n.pt data=datasets/data.yaml epochs=10 imgsz=640
yolo task=detect mode=train model=yolov8n.pt data=datasets/data.yaml epochs=50 batch=16 imgsz=640 patience=5
yolo detect train model=yolov8n.pt data=datasets/data.yaml epochs=50 batch=16 imgsz=640 patience=5 cfg=hyp.yaml

# test model

yolo task=detect mode=val model=runs/detect/train10/weights/best.pt data=datasets/data.yaml
yolo detect val model=runs/detect/train10/weights/best.pt data=datasets/data.yaml batch=8 conf=0.25 device=0

# run web application

python garbage_classification/waste_classification_webcam.py
