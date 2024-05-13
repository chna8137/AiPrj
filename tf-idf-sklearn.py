from sklearn.feature_extraction.text import TfidfVectorizer

# 설치한 konlpy 외부라이브러리로부터 Hannanum 기능 사용하도록 설정
from konlpy.tag import Hannanum

# 문자열(문장) 수정을 위한 파이썬 기본 기능 추가
import re

# 형태소 분석기 사용
myHannanum = Hannanum()

org_docs = [
    "학생들은 빅데이터와 인공지능 기술을 배우고 있다.",
    "빅데이터 기술은 방대한 데이터를 처리한다. 빅데이터는 많은 데이터를 저장한다.",
    "빅데이터 기술을 많이 어렵다. 특히 하둡이 어렵다.",
    "나의 목표는 빅데이터 기술을 활용하는 빅데이터 소프트웨어 개발자이다.",
    "소프트웨어 개발은 코딩이 필수이다. 나는 소프트웨어 개발자가 되고 싶다. 소프트웨어 개발자 화이팅!",
    "인공지능 기술에서 자연어 처리는 재미있다. 자연어는 사람이 사용하는 일반적인 언어이다."

]

# 형태소 분석을 통해 변경된 문서 데이터
docs = []

# 형태소 분석기로 문서별 명사 추출하기
for org_doc in org_docs:
    replace_doc = re.sub("[!@#$%^&*()_+]", " ", org_doc)
    docs.append(" ".join(myHannanum.nouns(replace_doc)))

# 변경된 문서 출력해보기
print(docs)

tfidfv = TfidfVectorizer().fit(docs)
print(tfidfv.transform(docs).toarray())
print(tfidfv.vocabulary_)