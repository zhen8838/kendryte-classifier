#!/bin/bash
python3 predict_one_pic.py --model "../pretrained/mobilenetv1_1.0.pb" --image "data/dog.jpg"
python3 predict_one_pic.py --model "../pretrained/mobilenetv1_1.0.pb" --image "data/eagle.jpg"
python3 predict_one_pic.py --model "../pretrained/mobilenetv1_1.0.pb" --image "data/giraffe.jpg"
python3 predict_one_pic.py --model "../pretrained/mobilenetv1_1.0.pb" --image "data/horses.jpg"
python3 predict_one_pic.py --model "../pretrained/mobilenetv1_1.0.pb" --image "data/person.jpg"
python3 predict_one_pic.py --model "../pretrained/mobilenetv1_1.0.pb" --image "data/scream.jpg"






