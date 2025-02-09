from roboflow import Roboflow
rf = Roboflow(api_key="pfp0zw8bjZWp5AtULvmd")
project = rf.workspace("projectverba").project("yolo-waste-detection")
version = project.version(1)
dataset = version.download("yolov8")

