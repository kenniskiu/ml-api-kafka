from random import randint
from typing import Set, Any
from fastapi import FastAPI
from kafka import TopicPartition

import uvicorn
import aiokafka
import asyncio
import json
import logging
import joblib

gender_vectorizer = open("models/gender_vectorizer.pkl","rb")
gender_cv = joblib.load(gender_vectorizer)
gender_nv_model = open("models/gender_nv_model.pkl","rb")
gender_clf = joblib.load(gender_nv_model)
app = FastAPI()

# instantiate the API
app = FastAPI()

# global variables
consumer_task = None
consumer = None
_name = ''

# env variables
KAFKA_TOPIC = 'topic'
KAFKA_CONSUMER_GROUP_PREFIX = 'consumer'
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'

# initialize logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
log = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    log.info('Initializing API ...')
    await initialize()
    await consume()


@app.on_event("shutdown")
async def shutdown_event():
    log.info('Shutting down API')
    consumer_task.cancel()
    await consumer.stop()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/name")
async def name():
    return _name


async def initialize():
    loop = asyncio.get_event_loop()
    global consumer
    group_id = f'{KAFKA_CONSUMER_GROUP_PREFIX}-{randint(0, 10000)}'
    log.debug(f'Initializing KafkaConsumer for topic {KAFKA_TOPIC}, group_id {group_id}'
              f' and using bootstrap servers {KAFKA_BOOTSTRAP_SERVERS}')
    consumer = aiokafka.AIOKafkaConsumer(KAFKA_TOPIC, loop=loop,
                                         bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                                         group_id=group_id)
    # get cluster layout and join group
    await consumer.start()

    partitions: Set[TopicPartition] = consumer.assignment()
    nr_partitions = len(partitions)
    if nr_partitions != 1:
        log.warning(f'Found {nr_partitions} partitions for topic {KAFKA_TOPIC}. Expecting '
                    f'only one, remaining partitions will be ignored!')
    for tp in partitions:

        # get the log_end_offset
        end_offset_dict = await consumer.end_offsets([tp])
        end_offset = end_offset_dict[tp]

        if end_offset == 0:
            log.warning(f'Topic ({KAFKA_TOPIC}) has no messages (log_end_offset: '
                        f'{end_offset}), skipping initialization ...')
            return

        log.debug(f'Found log_end_offset: {end_offset} seeking to {end_offset-1}')
        consumer.seek(tp, end_offset-1)
        msg = await consumer.getone()
        log.info(f'Initializing API with data from msg: {msg}')
        value = json.loads(msg.value)
        name = value['name']
        prediction = await predict(name)
        print(prediction) # run the function to predict
        # update the API state
        _update_name(prediction)
        return


async def consume():
    global consumer_task
    consumer_task = asyncio.create_task(send_consumer_message(consumer))


async def send_consumer_message(consumer):
    try:
        # consume messages
        async for msg in consumer:
            # x = json.loads(msg.value)
            log.info(f"Consumed msg: {msg}")

            # update the API state
            _update_name(msg)
    finally:
        # will leave consumer group; perform autocommit if enabled
        log.warning('Stopping consumer')
        await consumer.stop()


def _update_name(message) -> None:
    value = message
    global _name
    _name = value

async def predict(name):
    vectorized_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vectorized_name)
    if prediction[0] == 0:
        result = 'female'   
    else:
        result = 'male'
    return {"orig_name" : name,"prediction":result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
