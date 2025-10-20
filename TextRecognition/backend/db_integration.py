import os

import fastapi
from pydantic import BaseModel
from fastapi import status, responses

from datetime import datetime

from sqlalchemy.orm import Session, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, DateTime, Text


DB_USER = os.getenv('DB_USER')
DB_USER_PASSWORD = os.getenv('DB_USER_PASSWORD') 

#creating necessary information for the database 
Base = declarative_base()
engine = create_engine('postgresql://'+DB_USER+':'+DB_USER_PASSWORD+'@localhost:5432/textrecognitiondatabase')

#creating models for the database
class UserInteraction(Base):
    __tablename__ = 'user_interaction'

    user_id: Mapped[int] = mapped_column(primary_key = True)
    datetime: Mapped[datetime] = mapped_column(DateTime)
    input: Mapped[str] = mapped_column(Text)
    prediction: Mapped[str] = mapped_column(Text)

#creating models in database
Base.metadata.create_all(engine)

#creating models for API
class ImageData(BaseModel):
    input: str
    prediction: str

app = fastapi.FastAPI()

@app.post('/interaction')
async def interaction_insertion(interaction: ImageData, status_code = status.HTTP_201_CREATED):
    current_datetime = datetime.now()
    with Session(engine) as session:
        #creating and pushing a row for the database
        user_interaction_row = UserInteraction(datetime = current_datetime, input = interaction.input, prediction = interaction.prediction)
        session.add(user_interaction_row)
        session.commit()
    #returning the response
    return responses.Response(status_code = status.HTTP_201_CREATED)
# session = sessionmaker(bind=engine)()
# session.add(UserInteraction(input="Test input", result="Test result"))
# session.commit()