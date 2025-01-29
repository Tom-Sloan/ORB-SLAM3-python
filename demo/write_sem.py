# writer.py
from multiprocessing import shared_memory, Semaphore
import time

# Create shared memory and semaphore
shm = shared_memory.SharedMemory(create=True, size=10, name="shared_block")
sem = Semaphore(1)  # Binary semaphore
counter = 0

try:
    while True:
        sem.acquire()  # Lock access
        # Convert string to bytes and copy to shared memory
        message = f"Hello {counter}".encode()
        shm.buf[0:len(message)] = message  # Write data byte by byte
        print(f"Writer: Data written to shared memory. Counter: {counter}")
        counter += 1
        sem.release()  # Unlock access
        time.sleep(2)  # Simulate periodic writes
finally:
    shm.close()
    shm.unlink()
