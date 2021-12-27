from pandas import DataFrame
from pandas import Series
from numpy  import ndarray
class DataTransformer:
    def __init__(self):
        pass
    
    def info(self):
        print("This is DataTransformer")


    def pd_data_frame_to_ndarray(self, data:DataFrame) -> ndarray :
        return data
    
    def pd_data_frame_to_list(self, data:DataFrame) -> list:
        return data

    def pd_series_to_ndarray(self, data: Series) -> ndarray:
        return data

    def pd_series_to_list(self, data: Series) -> list:
        return data
    
    
    def ndarray_to_pd_series(self, data: ndarray) -> Series:
        return data

    def ndarray_to_list(self, data: ndarray) -> list:
        return data
    
    def ndarray_to_pd_data_frame(self, data: ndarray) -> DataFrame:
        return data
    
    def list_to_pd_data_frame(self, data: list) -> DataFrame:
        return data
    
    def list_to_pd_series(self, data: list) -> Series:
        return data
    
    def list_to_ndarray(self, data: list) -> ndarray:
        return data