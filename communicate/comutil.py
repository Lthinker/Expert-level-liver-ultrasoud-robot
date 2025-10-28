from pymodbus.client.sync import ModbusSerialClient as ModbusClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.constants import Endian
from pymodbus.transaction import ModbusRtuFramer
import time
import numpy as np
from nptdms import TdmsWriter, ChannelObject

def read_float_values(client):  
    for attempt in range(10):  
        try:  
            response = client.read_holding_registers(address=0x0400, count=12, unit=0x01)  
            if not response.isError():  
                decoder = BinaryPayloadDecoder.fromRegisters(response.registers, byteorder=Endian.Big, wordorder=Endian.Big)  
                values = [decoder.decode_32bit_float() for _ in range(6)]  
                return values, time.time()   
            else:  
                print("Read error:", response)  
        except Exception as e:  
            print("Communication error:", e)  
        time.sleep(0.001)    
    return None, None  


def high_speed_collect(duration=10, interval=0.027, tdms_file_name="data.tdms",client=None):
    with TdmsWriter(tdms_file_name) as tdms_writer:
        channel_names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        data_lists = {name: [] for name in channel_names}
        time_data = []

        start_time = time.time()
        while time.time() - start_time < duration:
            btime = time.time()
            values, timestamp = read_float_values(client)
            if values and timestamp is not None:
                for name, value in zip(channel_names, values):
                    data_lists[name].append(value)
                time_data.append(timestamp)
            etime = time.time()
            time.sleep(interval-(etime-btime))
        channels = [ChannelObject("Group", name, np.array(data_lists[name])) for name in channel_names]
        time_channel = ChannelObject("Group", "Time", np.array(time_data))
        tdms_writer.write_segment(channels + [time_channel])

def create_client(port='/dev/ttyUSB0'):
    client = ModbusClient(method='rtu', port=port, baudrate=115200, timeout=3, parity='N', stopbits=1, bytesize=8)
    # 尝试连接
    if client.connect():
        print("Modbus RTU Client connected")
    else:
        print("Connection failed")
        client.close()
        exit()
    return client