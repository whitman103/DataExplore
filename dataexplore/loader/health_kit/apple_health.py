import xml.etree.ElementTree as ET
import pandas as pd
from pprint import pprint as print
from dataexplore.loader.health_kit.schema import (
    RecordType,
    Distance,
)
import matplotlib.pyplot as plt
from datetime import datetime, timezone

tree = ET.parse("/Users/johnwhitman/Downloads/apple_health_export/export.xml")

measurement_types = {}

root = tree.getroot()

test = [
    Distance.from_record(x.attrib)
    for x in root.iter("Record")
    if x.attrib["type"] in RecordType
]


print(measurement_types)
