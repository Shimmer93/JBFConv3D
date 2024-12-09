import numpy as np
from PIL import Image
from io import BytesIO

def decode_jbf(jbf, num_maps=19):
    jbf_bytes = jbf.tobytes()
    jbf_img = Image.open(BytesIO(jbf_bytes))
    jbf_img = np.array(jbf_img)

    J = num_maps
    HJ, W = jbf_img.shape[:2]
    assert HJ % J == 0
    H = HJ // J
    
    jbf_out = jbf_img.reshape(J, H, W).astype(np.float32)
    return jbf_out

def read_jbf_seq(fn, num_maps=19):
    jbf_seq = np.load(fn, allow_pickle=True)
    jbf_seq = [decode_jbf(jbf, num_maps) for jbf in jbf_seq]
    return jbf_seq