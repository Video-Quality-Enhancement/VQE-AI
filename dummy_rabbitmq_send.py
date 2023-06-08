from models import VideoEnhanceRequest
import json

from pika.adapters.blocking_connection import BlockingChannel
from models import EnhancedVideoResponse
from config import AMQPconnection
import dotenv

dotenv.load_dotenv()

url = "https://download.pexels.com/vimeo/822411462/pexels-tommy-t-16609645.mp4?fps=29.97&width=426"

request = VideoEnhanceRequest('user01', 'request01', url)

request = request.__dict__

connection = AMQPconnection()
producerCh = connection.create_channel()
# result = producerCh.queue_declare(queue="enhanced.video", durable=True)
queue = "240p_queue"

body = json.dumps(request)
producerCh.basic_publish(exchange='', routing_key=queue, body=body)

print("message sent", request)
