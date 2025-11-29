from sqlalchemy import Column, Integer, String, Text
from .database import Base

class SkinRecord(Base):
    __tablename__ = "skin_records"
    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String, unique=True)
    question = Column(Text)
    ai_response = Column(Text)
    audio_path = Column(String)
