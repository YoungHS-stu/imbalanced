from pandas import DataFrame
from pandas import Series
from numpy  import ndarray
class DataTransformer:
    def __init__(self):
        pass
    
    def info(self):
        print("This is DataTransformer")

    def pd_data_frame_to_ndarray(self, data:DataFrame) -> ndarray :
        return data.to_numpy()
    
    def pd_data_frame_to_2d_list(self, data:DataFrame) -> list:
        return data.values.tolist()

    def pd_series_to_ndarray(self, data: Series) -> ndarray:
        return data.to_numpy()

    def pd_series_to_list(self, data: Series) -> list:
        return data.tolist()
    
    def ndarray_to_pd_series(self, data: ndarray, rows=[]) -> Series:
        return Series(data, index=rows)

    def ndarray_to_list(self, data: ndarray) -> list:
        return data.tolist()
    
    def ndarray_to_pd_data_frame(self, data: ndarray, columns=[], rows=[]) -> DataFrame:
        if len(columns) != data.shape[1]:
            raise ValueError("columns.length must equal to column number in [data]")
        if len(rows) != data.shape[0]:
            raise ValueError("rows.length must equal to rows number in [data]")
        return DataFrame(data, columns=columns, index=rows)
    
    def list_to_pd_data_frame(self, data: list,  columns=[], rows=[]) -> DataFrame:
        if len(columns) != len(data[0]):
            raise ValueError("columns.length must equal to column number in [data]")
        if len(rows) != len(data):
            raise ValueError("rows.length must equal to rows number in [data]")
        return DataFrame(data, columns=columns, index=rows)
    
    def list_to_pd_series(self, data: list, rows=[]) -> Series:
        if len(rows) != len(data):
            raise ValueError("rows.length must equal to rows number in [data]")
        return Series(data, index=rows)
    
    def list_to_ndarray(self, data: list) -> ndarray:
        from numpy import array
        return array(data)