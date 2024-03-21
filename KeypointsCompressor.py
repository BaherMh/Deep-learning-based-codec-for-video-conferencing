from golomb_coding import golomb_coding, golomb_decoding
import math
import torch

def golomb_code(num, M):
    if num > 0:
        return golomb_coding(int(2*num), M)
    else:
        return golomb_coding(int(-2*num+1), M)





def golomb_decode(code, M):
    res = []
    q = 0
    r = 0
    flag = True
    b = int(math.log2(M))
    i = -1
    while i+1 < len(code):
        i += 1
        if code[i]=='0' and flag:
            q += 1
        elif code[i]=='1' and flag:
            flag = False
        else:
            #print(i)
            r = 0
            r_ = int("0b" + code[i:i+b], 2)
            if r_ < 2**(b+1) - M:
                r = r_
                i += b
            else:
                r_ = int("0b" + code[i:i+b+1], 2)
                r = r_ - 2**(b+1) + M
                i += (b+1)
            ans = q*M+r
            if ans%2 == 0:
                return(ans/2)
            else:
                return(-1*((ans-1)/2))
            i -= 1
            flag = True
            q = 0
            r = 0


def compress_keypoints(kp_driving, kp_source, quantization):
    keys_difference = torch.round(quantization* (kp_driving['fg_kp'] - kp_source['fg_kp']))
    #print(keys_difference)
    keys_compressed = []
    key_bits = 0
    for key in keys_difference:
        for x in key:
            point_compressed = []
            for y in x:
                point_compressed.append(golomb_code(y.item(), 3))
                key_bits += len(golomb_code(y.item(), 3))
               
            keys_compressed.append(point_compressed)

    return keys_compressed, key_bits

def decompress_keypoints(keys_compressed, kp_source, quantization):
    kp_decompressed = {}
    kp_decompressed['fg_kp'] = torch.tensor(kp_source['fg_kp'])
    for i in range(len(kp_decompressed['fg_kp'][0])):
        kp_decompressed['fg_kp'][0][i].data[0] = torch.tensor(((golomb_decode(keys_compressed[i][0], 3)))/quantization) + kp_source['fg_kp'][0][i].data[0]
        kp_decompressed['fg_kp'][0][i].data[1] = torch.tensor(((golomb_decode(keys_compressed[i][1], 3)))/quantization) + kp_source['fg_kp'][0][i].data[1]
    return kp_decompressed
