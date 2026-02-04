import os
import warnings

warnings.filterwarnings("ignore")

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, EmailStr, field_validator, computed_field
from typing import Optional, Literal, Union, Dict, List
from datetime import date, datetime
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class EmploymentType(str, Enum):
    full_time = "full_time"
    contractor = "contractor"
    part_time = "part_time"

class EmployeeStatus(str, Enum):
    active = "active"
    probation = "probation"
    terminated = "terminated"

class OfficeAddress(BaseModel):
    building: str
    city: str
    country: str
    postal_code : str = Field(..., pattern=r"^\d{5}$")

class EmergencyContact(BaseModel):
    name : str
    relationship: str
    phone : str = Field(..., pattern=r"^\+?[0-9]{10,15}$")

class EmployeeProfile(BaseModel):

    employee_id: int
    full_name: str = Field(..., min_length=3, max_length=100)
    email: EmailStr = Field(..., alias="workEmail")
    age: int = Field(..., ge =18, le=50)
    department: Literal["engineering", "hr", "finance", "sales", "marketing"]
    employment_type :EmploymentType
    status : EmployeeStatus = EmployeeStatus.probation
    employee_code: Optional[str] = Field(default=None, description="code such as MAC-1234 EMP-5312")
    skills: List[str] = Field(default_factory=list, max_length=10)
    office_address: OfficeAddress
    emergency_contact: Optional[EmergencyContact] = None
    joining_date: date | str
    internal_notes: Dict[str, Union[str, int, bool]] = Field(default_factory=dict)

    @field_validator("skills")
    @classmethod
    def engineering_requires_core_skills(cls, v, info):
        if info.data.get("department") == "engineering":
            required = {"python", "git"}
            if not required.issubset({s.lower() for s in v}):
                raise ValueError("Engineering requires Python and Git")
        return v

    # Computed field
    @computed_field
    @property
    def is_senior_employee(self) -> bool:
        return self.age >= 40
    
    @field_validator("joining_date")
    @classmethod
    def parse_joining_date(cls, v):
        if isinstance(v, date):
            return v
        return datetime.strptime(v, "%B %d, %Y").date()
    
    @field_validator("internal_notes", mode="before")
    @classmethod
    def normalize_notes(cls, v):
        if isinstance(v, str):
            return {"raw_notes": v}
        return v


structured_details = model.with_structured_output(EmployeeProfile)

res = structured_details.invoke("""
    Employee Onboarding Summary

    We are onboarding a new team member named Jonathan Miller, who will be joining
    the engineering department as a full-time employee. Jonathan is 34 years old.

    His official company email address will be jonathan.miller@acmecorp.com.
    Internally, HR has assigned him employee ID 78231 and employee code ENG-4432.

    Jonathan will work from our Berlin office in Building A, Berlin, Germany,
    postal code 10115. His joining date is June 17, 2024.

    He has experience with Python, Git, Docker, and distributed systems.
    He will be responsible for backend development and mentoring junior engineers.

    During the initial phase, his employment status should be probation.

    Emergency contact details:
    Name: Sarah Miller
    Relationship: Wife
    Phone: +4915123456789

    Internal notes:
    Security training completed.
    Laptop asset number: 55421.
    Background verification complete.
""")

print(res.content)