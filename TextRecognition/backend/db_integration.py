from sqlalchemy import create_engine, Column, DateTime, Text, insert
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()
engine = create_engine('postgresql://luchian:luchianN007@localhost:5432/textrecognitiondatabase')

class UserInteraction(Base):
    __tablename__ = 'user_interaction'
    datetime = Column(DateTime, primary_key = True)
    input = Column(Text, nullable=False)
    prediction = Column(Text, nullable=False)

Base.metadata.create_all(engine)
with engine.connect() as conn:
    #we insert data here from the api that we receive
    pass
# session = sessionmaker(bind=engine)()
# session.add(UserInteraction(input="Test input", result="Test result"))
# session.commit()