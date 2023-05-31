from models import VideoEnhanceRequest
import json

from pika.adapters.blocking_connection import BlockingChannel
from models import EnhancedVideoResponse
from config import AMQPconnection
import dotenv

dotenv.load_dotenv()

url = "https://drive.google.com/uc?id=1yczEIQHc14jc0B412eB5JlouGLU4sobJ"

request = VideoEnhanceRequest(1000, 1000, url)

request = request.__dict__

connection = AMQPconnection()
producerCh = connection.create_channel()
# result = producerCh.queue_declare(queue="enhanced.video", durable=True)
queue = "360p_queue"

body = json.dumps(request)
producerCh.basic_publish(exchange='', routing_key=queue, body=body)

print("message sent", request)
