from pika.adapters.blocking_connection import BlockingChannel
from models import EnhancedVideoResponse
from config import AMQPconnection

def enhanced_video_producer(enhanced_video_response: EnhancedVideoResponse):
    connection = AMQPconnection()
    producerCh = connection.create_channel()
    result = producerCh.queue_declare(queue="enhanced.video", durable=True)
    queue = result.method.queue

    body = enhanced_video_response.dumps()
    producerCh.basic_publish(exchange='', routing_key=queue, body=body)
    
    print("message sent", enhanced_video_response)