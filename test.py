import numpy as np
import struct

def load_idx3_ubyte(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))  # 헤더(16바이트) 읽기
        data = np.frombuffer(f.read(), dtype=np.uint8)  # 이미지 데이터 읽기
        data = data.reshape(num_images, rows, cols)  # (N, 28, 28) 형태로 변환
    return data

# WSL 내부 경로 사용
filename = "/home/inchanbaek/ai_blog/t10k-images.idx3-ubyte"
images = load_idx3_ubyte(filename)

print("이미지 데이터 크기:", images.shape)  # (10000, 28, 28)

