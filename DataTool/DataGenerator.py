#! 生成假数据使用
import numpy as np
class DataGenerator:
    def __init__(self):
        pass

    def info(self):
        print("This is Data Preprocessor")
        
    def multivariate_normal(self, mean, cov, sample_size):
        return np.random.multivariate_normal(mean, cov, sample_size)
    
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_generator = DataGenerator()
    mean = [0, 3]
    cov = [[1, 0], [0, 3]]
    sample_size = [10, 50]
    x, y = data_generator.multivariate_normal(mean, cov, sample_size).T
    
    
    
    
    
