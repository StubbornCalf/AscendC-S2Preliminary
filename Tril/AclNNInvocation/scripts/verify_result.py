import os
import sys
import numpy as np

loss = 1e-3 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
minimum = 10e-10

def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.float32) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=np.float32) # 从bin文件读取预期运算结果
    result = np.abs(real_result - golden) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            # print("\tgolden=",golden, "\treal_result=", real_result);
            resultzero = 0
            goldenzero = 0
            noteql = 0
            # for i in range(real_result.shape[0]):  # 第一维  
            #     if(real_result[i] != golden[i]):
            #         # print(golden[i], "!=", real_result[i])
            #         noteql += 1
            #     # else:
            #         # print(golden[i], "==", real_result[i])
            #     if real_result[i] == 0.0:  # 打印当前元素  
            #         resultzero += 1
            #     if golden[i] == 0.0:
            #         goldenzero += 1
            # print("result-golden=", resultzero-goldenzero)
            print("golden=", golden)
            print("real_result=", real_result)
            # print("not equal=", noteql)
            return False
    print("test pass")
    #print("\tgolden=",golden, "\treal_result=", real_result);
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
