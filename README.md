# News Recommendation System

This repository contains the implementation of a news recommendation system using deep learning models. The system utilizes decoders for both news and user encoding, and employs a click predictor to recommend news articles.

## Project Structure
![image](https://github.com/user-attachments/assets/cb19ce40-422c-47c1-bc5f-633c3589bbe1)


## Issue Encountered
When running the train.py file, an error occurs during the data loading process. The error traceback indicates an issue with the conversion of data to tensors. Below is the detailed error message:


Traceback (most recent call last):
  File "c:\Users\trong\Documents\News_Recommend_System\train.py", line 101, in <module>
    for batch in train_loader:
  File "C:\Users\trong\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\dataloader.py", line 631, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "C:\Users\trong\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\dataloader.py", line 675, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\trong\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\_utils\fetch.py", line 54, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\trong\Documents\News_Recommend_System\train.py", line 55, in collate_fn
    raise e
  File "c:\Users\trong\Documents\News_Recommend_System\train.py", line 49, in collate_fn
    clicked_news_tensors = convert_to_tensor_list([news for user_news in clicked_news_padded for news in user_news])
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\trong\Documents\News_Recommend_System\train.py", line 34, in convert_to_tensor_list
    tensor_list.append(torch.tensor(sub_list, dtype=dtype))
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: not a sequence


![Error Screenshot](https://github.com/cloudysman/News-Recommendation/blob/master/error.jpg)
