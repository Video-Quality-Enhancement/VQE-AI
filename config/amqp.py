import pika, os

class AMQPconnection:

    def __init__(self):
        parameters = pika.URLParameters(os.getenv('AMQP_URL'))
        self.conn = pika.BlockingConnection(parameters)
        self.channels = []
    
    def create_channel(self):
        channel = self.conn.channel()
        # self.channels.append(channel)
        return channel

    def close(self):
        # for channel in self.channels:
        #     channel.close()
        self.conn.close()

    def send_heartbeat(self):
        self.conn.process_data_events()