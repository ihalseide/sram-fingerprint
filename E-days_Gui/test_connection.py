import time
write_txt = "3\r"


def main(thread_object):
    thread_object.com_port.write(write_txt.encode())        
    response = thread_object.com_port.read_all().decode('utf-8')
    thread_object.com_port.close()
    print(response)
    if "f" in response:
        return False
    else:
        return True