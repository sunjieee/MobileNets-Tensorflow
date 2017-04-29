#!/bin/bash

python convert_to_tfrecords.py --jpeg_file_path=../train/*/* --tfrecord_name=../records/train.tfrecords

python convert_to_tfrecords.py --jpeg_file_path=../valid/*/* --tfrecord_name=../records/valid.tfrecords
