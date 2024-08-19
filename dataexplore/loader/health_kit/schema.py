from enum import Enum
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Dict


class RecordType(str, Enum):
    active_calories = "HKQuantityTypeIdentifierActiveEnergyBurned"
    body_mass = "HKQuantityTypeIdentifierBodyMass"
    distance = "HKQuantityTypeIdentifierDistanceWalkingRunning"
    heart_rate = "HKQuantityTypeIdentifierHeartRate"
    resting_heart_rate = "HKQuantityTypeIdentifierRestingHeartRate"
    step_count = "HKQuantityTypeIdentifierStepCount"
    vo2_max = "HKQuantityTypeIdentifierVO2Max"
    null = "null"


class Record(BaseModel):
    record_type: RecordType = RecordType.null


class PointRecord(Record):
    creation_date: datetime

    @classmethod
    def from_record(cls, input_data: Dict[str, str]) -> "PointRecord":
        return cls(creation_date=convert_datetime(input_data["creationDate"]))


class DurationRecord(Record):
    duration: timedelta
    creation_date: datetime

    @classmethod
    def from_record(cls, input_data: Dict[str, str]) -> "DurationRecord":
        duration = convert_datetime(input_data["endDate"]) - convert_datetime(
            input_data["startDate"]
        )
        return cls(
            creation_date=convert_datetime(input_data["creationDate"]),
            duration=duration,
        )


class BodyMass(PointRecord):
    record_type: RecordType = RecordType.body_mass
    unit: str
    weight: float

    @classmethod
    def from_record(cls, input_data: Dict[str, str]) -> "BodyMass":
        base = PointRecord.from_record(input_data)
        return cls(
            unit=input_data["unit"],
            weight=float(input_data["value"]),
            **base.model_dump()
        )


class ActiveCalories(DurationRecord):
    record_type: RecordType = RecordType.active_calories
    calories_burned: float

    @classmethod
    def from_record(cls, input_data: Dict[str, str]) -> "ActiveCalories":
        base = DurationRecord.from_record(input_data)
        return cls(calories_burned=float(input_data["value"]), **base.model_dump())


class Distance(DurationRecord):
    record_type: RecordType = RecordType.distance
    miles: float

    @classmethod
    def from_record(cls, input_data: Dict[str, str]) -> "Distance":
        base = DurationRecord.from_record(input_data)
        return cls(miles=float(input_data["value"]), **base.model_dump())


class HeartRate(PointRecord):
    record_type: Record = RecordType.heart_rate
    bpm: float

    @classmethod
    def from_record(cls, input_data: Dict[str, str]) -> "HeartRate":
        base = PointRecord.from_record(input_data)
        return cls(bpm=float(input_data["value"]), **base.model_dump())


class RestingHeartRate(DurationRecord):
    record_type: RecordType = RecordType.resting_heart_rate
    bpm: float

    @classmethod
    def from_record(cls, input_data: Dict[str, str]) -> "RestingHeartRate":
        base = DurationRecord.from_record(input_data)
        return cls(bpm=float(input_data["value"]), **base.model_dump())


class StepCount(DurationRecord):
    record_type: RecordType = RecordType.step_count
    steps: int

    @classmethod
    def from_record(cls, input_data: Dict[str, str]) -> "StepCount":
        base = DurationRecord.from_record(input_data)
        return cls(steps=int(input_data["value"]), **base.model_dump())


class VO2Max(PointRecord):
    record_type: RecordType = RecordType.vo2_max
    value: float

    @classmethod
    def from_record(cls, input_data: Dict[str, str]) -> "VO2Max":
        base = PointRecord.from_record(input_data)
        return cls(value=float(input_data["value"]), **base.model_dump())


def convert_datetime(hk_time_record: str) -> datetime:
    return datetime.strptime(hk_time_record, "%Y-%m-%d %H:%M:%S %z")
