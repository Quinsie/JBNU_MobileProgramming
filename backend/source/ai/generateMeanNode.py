# backend/source/ai/generateMeanNode.py

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from multiprocessing import Pool

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_pos")
from source.utils.getDayType import getDayType

