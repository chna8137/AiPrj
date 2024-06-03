import pickle
import tensorflow as tf
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 학습모델 로딩하기
loaded_model = tf.keras.models.load_model("model/naver_movie.h5")

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

with open("model/naver_movie_tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

okt = Okt()

max_len = 30

def sentiment_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩

    score = float(loaded_model.predict(pad_new)) # 예측
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))

sentiment_predict('이 영화 재미 없다.')

sentiment_predict('이 영화 재미 있다.')

sentiment_predict('배우 연기가 별로다.')

sentiment_predict('배우 연기가 좋다.')

sentiment_predict('망한 영화인가?')