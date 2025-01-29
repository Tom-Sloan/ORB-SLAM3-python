# reader.py
from multiprocessing import shared_memory, Semaphore

# Attach to existing shared memory and semaphore
shm = shared_memory.SharedMemory(name="shared_block")
sem = Semaphore(1)  # Use the same semaphore as the writer

try:
    while True:
        sem.acquire()  # Lock access
        data = bytes(shm.buf[:10]).decode()  # Read data
        print(f"Reader: Data read from shared memory: {data}")
        sem.release()  # Unlock access
finally:
    shm.close()
